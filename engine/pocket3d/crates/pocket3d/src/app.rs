//! Windowed application loop (winit 0.30) with fixed-step simulation.
//!
//! Games implement [`Game`]; the same object can also be driven headlessly
//! by calling `tick`/`compose` directly (see OpenStrike's script mode).

use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use anyhow::Context;
use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
#[cfg(target_os = "windows")]
use winit::platform::windows::WindowAttributesExtWindows;
#[cfg(target_os = "windows")]
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{CursorGrabMode, Window, WindowId, WindowLevel};

#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{BOOL, HWND};
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
};
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
#[cfg(target_os = "windows")]
use windows::core::Interface;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::hud::Hud;
use crate::input::Input;
use crate::renderer::{Renderer, RendererConfig};
use crate::scene::Scene;
use crate::time::FixedTimestep;

pub struct AppConfig {
    pub title: String,
    pub size: (u32, u32),
    pub tick_hz: f32,
    /// Grab + hide the cursor for mouse look (Esc toggles).
    pub capture_mouse: bool,
    /// Alpha-composited window: the surface picks a non-opaque alpha mode
    /// and pixels the scene leaves transparent show the desktop behind the
    /// window. Pair with [`crate::scene::Scene::transparent_clear`].
    pub transparent: bool,
    /// Window chrome (title bar, borders). Off for widget-style windows.
    pub decorations: bool,
    /// Float above normal windows.
    pub always_on_top: bool,
    pub resizable: bool,
    /// Cap the render rate. `None` renders every vsync, which on a 120 Hz
    /// display means 120 fps; long-lived widgets should cap (and the loop
    /// then sleeps between frames instead of spinning on redraws).
    pub max_fps: Option<f32>,
    /// Left-mouse press starts an OS window drag (widget-style move;
    /// the press still reaches [`Input`] first, so clicks keep working).
    pub drag_window: bool,
    /// Generic requested MSAA sample count. Renderer capability negotiation
    /// determines the effective count at startup.
    pub requested_sample_count: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "Pocket3D".into(),
            size: (1600, 900),
            tick_hz: 64.0,
            capture_mouse: true,
            transparent: false,
            decorations: true,
            always_on_top: false,
            resizable: true,
            max_fps: None,
            drag_window: false,
            requested_sample_count: 1,
        }
    }
}

/// What the app loop needs from a game.
pub trait Game {
    /// Called once after the GPU exists — load assets here.
    fn init(&mut self, gpu: &Gpu, renderer: &mut Renderer) -> Result<()>;
    /// Called once per rendered frame before fixed ticks (mouse look etc.).
    fn frame(&mut self, dt: f32, input: &Input);
    /// Fixed-step simulation.
    fn tick(&mut self, dt: f32, input: &Input);
    /// Prepare renderer state for the next frame, outside any render pass.
    /// Default: nothing.
    fn prepare_render(&mut self, gpu: &Gpu, renderer: &mut Renderer) {
        let (_, _) = (gpu, renderer);
    }
    /// Provide the frame to draw. `time` is seconds since launch.
    fn compose(&mut self, alpha: f32, time: f32, size: (u32, u32)) -> (&Scene, &Camera, &Hud);
    /// Record extra passes over the finished frame (UI overlays, composite
    /// effects) before present. `format` is the target's texture format.
    /// Default: nothing.
    fn overlay(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) {
        let (_, _, _, _, _) = (gpu, encoder, view, format, size);
    }
    /// Return true to quit.
    fn wants_exit(&self) -> bool {
        false
    }
}

pub fn run(config: AppConfig, game: impl Game) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = WinitApp {
        config,
        game,
        state: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

struct WindowState {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    #[cfg(target_os = "windows")]
    direct_composition: Option<DirectCompositionState>,
    gpu: Gpu,
    renderer: Renderer,
    // Keep the HWND alive until after the surface and any composition target.
    window: Arc<Window>,
    input: Input,
    timestep: FixedTimestep,
    start: Instant,
    last_frame: Instant,
    mouse_captured: bool,
}

impl WindowState {
    fn configure_surface(&self) -> Result<()> {
        self.surface
            .configure(&self.gpu.device, &self.surface_config);

        #[cfg(target_os = "windows")]
        if let Some(direct_composition) = &self.direct_composition {
            // wgpu creates or replaces the DirectComposition swapchain and
            // associates it with the visual during surface configuration.
            direct_composition.commit()?;
        }

        Ok(())
    }
}

#[cfg(target_os = "windows")]
struct DirectCompositionState {
    device: IDCompositionDevice,
    // Retain the target and visual for the full lifetime of the surface.
    _target: IDCompositionTarget,
    visual: IDCompositionVisual,
}

#[cfg(target_os = "windows")]
impl DirectCompositionState {
    fn new(window: &Window) -> Result<Self> {
        let window_handle = window.window_handle().context("obtain HWND")?;
        let hwnd = match window_handle.as_raw() {
            RawWindowHandle::Win32(handle) => HWND(handle.hwnd.get() as *mut _),
            _ => anyhow::bail!("obtain HWND: winit returned a non-Win32 window handle"),
        };

        // A null DXGI device asks DirectComposition to create its own device.
        let device: IDCompositionDevice = unsafe { DCompositionCreateDevice(None::<&IDXGIDevice>) }
            .context("create DirectComposition device")?;
        let target = unsafe { device.CreateTargetForHwnd(hwnd, BOOL(1)) }
            .context("create DirectComposition target for HWND")?;
        let visual = unsafe { device.CreateVisual() }.context("create DirectComposition visual")?;
        unsafe { target.SetRoot(&visual) }
            .context("set DirectComposition visual as target root")?;

        Ok(Self {
            device,
            _target: target,
            visual,
        })
    }

    fn create_surface(&self, instance: &wgpu::Instance) -> Result<wgpu::Surface<'static>> {
        // SAFETY: `visual` is a valid IDCompositionVisual. wgpu increments its
        // COM refcount, and WindowState retains this state until after the
        // surface is dropped.
        unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CompositionVisual(
                self.visual.as_raw(),
            ))
        }
        .context("create wgpu DirectComposition surface")
    }

    fn commit(&self) -> Result<()> {
        unsafe { self.device.Commit() }.context("commit DirectComposition device")
    }
}

struct WinitApp<G: Game> {
    config: AppConfig,
    game: G,
    state: Option<WindowState>,
    error: Option<anyhow::Error>,
}

impl<G: Game> WinitApp<G> {
    fn init_state(&mut self, event_loop: &ActiveEventLoop) -> Result<WindowState> {
        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.size.0,
                self.config.size.1,
            ))
            .with_transparent(self.config.transparent)
            .with_decorations(self.config.decorations)
            .with_resizable(self.config.resizable)
            .with_window_level(if self.config.always_on_top {
                WindowLevel::AlwaysOnTop
            } else {
                WindowLevel::Normal
            });

        #[cfg(target_os = "windows")]
        let attrs = if self.config.transparent {
            attrs.with_no_redirection_bitmap(true)
        } else {
            attrs
        };

        let window = Arc::new(event_loop.create_window(attrs)?);

        #[cfg(target_os = "windows")]
        let instance = if self.config.transparent {
            Gpu::new_instance_with_backends(wgpu::Backends::DX12)
        } else {
            Gpu::new_instance()
        };
        #[cfg(not(target_os = "windows"))]
        let instance = Gpu::new_instance();

        #[cfg(target_os = "windows")]
        let (surface, direct_composition) = if self.config.transparent {
            let direct_composition = DirectCompositionState::new(&window)?;
            let surface = direct_composition.create_surface(&instance)?;
            (surface, Some(direct_composition))
        } else {
            (
                instance
                    .create_surface(window.clone())
                    .context("create wgpu window surface")?,
                None,
            )
        };
        #[cfg(not(target_os = "windows"))]
        let surface = instance.create_surface(window.clone())?;

        let gpu = Gpu::from_instance_for_surface(instance, &surface)?;

        #[cfg(target_os = "windows")]
        if self.config.transparent {
            let backend = gpu.adapter.get_info().backend;
            anyhow::ensure!(
                backend == wgpu::Backend::Dx12,
                "transparent Windows DirectComposition surface requires DX12, selected {backend:?}"
            );
        }

        let px = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&gpu.adapter, px.width.max(1), px.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("surface not supported by adapter"))?;
        surface_config.present_mode = wgpu::PresentMode::AutoVsync;

        #[cfg(target_os = "windows")]
        if self.config.transparent {
            configure_windows_transparent_surface(&surface, &gpu.adapter, &mut surface_config)?;
        }
        #[cfg(not(target_os = "windows"))]
        if self.config.transparent {
            surface_config.alpha_mode = pick_alpha_mode(&surface, &gpu.adapter)?;
        }

        surface.configure(&gpu.device, &surface_config);
        #[cfg(target_os = "windows")]
        if let Some(direct_composition) = &direct_composition {
            direct_composition.commit()?;
        }

        let mut renderer = Renderer::new_with_config(
            &gpu,
            surface_config.format,
            RendererConfig {
                requested_sample_count: self.config.requested_sample_count,
            },
        )?;
        self.game.init(&gpu, &mut renderer)?;

        let mut state = WindowState {
            surface,
            surface_config,
            #[cfg(target_os = "windows")]
            direct_composition,
            gpu,
            renderer,
            window,
            input: Input::default(),
            timestep: FixedTimestep::new(self.config.tick_hz),
            start: Instant::now(),
            last_frame: Instant::now(),
            mouse_captured: false,
        };
        if self.config.capture_mouse {
            set_mouse_capture(&mut state, true);
        }
        Ok(state)
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        if self.error.is_some() {
            return;
        }
        let Some(state) = self.state.as_mut() else {
            return;
        };

        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32();
        state.last_frame = now;

        self.game.frame(dt, &state.input);
        let ticks = state.timestep.advance(dt);
        for _ in 0..ticks {
            self.game.tick(state.timestep.step, &state.input);
        }
        state.input.end_frame();

        if self.game.wants_exit() {
            event_loop.exit();
            return;
        }

        self.game.prepare_render(&state.gpu, &mut state.renderer);

        let frame = match state.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                if let Err(error) = state.configure_surface() {
                    self.error = Some(error);
                    event_loop.exit();
                }
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => {
                log::warn!("surface timeout; skipping frame");
                return;
            }
            Err(e @ wgpu::SurfaceError::OutOfMemory) => {
                let error = anyhow::anyhow!("surface error: {e}");
                log::error!("{error}");
                self.error = Some(error);
                event_loop.exit();
                return;
            }
            Err(e) => {
                log::error!("surface error: {e}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let size = (state.surface_config.width, state.surface_config.height);
        let (scene, camera, hud) = self.game.compose(
            state.timestep.alpha(),
            state.start.elapsed().as_secs_f32(),
            size,
        );
        state
            .renderer
            .render(&state.gpu, &view, size, scene, camera, hud);
        let mut encoder = state
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("overlay") });
        self.game.overlay(&state.gpu, &mut encoder, &view, state.surface_config.format, size);
        state.gpu.queue.submit([encoder.finish()]);
        state.window.pre_present_notify();
        frame.present();
    }
}

#[cfg(target_os = "windows")]
fn configure_windows_transparent_surface(
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
    config: &mut wgpu::SurfaceConfiguration,
) -> Result<()> {
    let caps = surface.get_capabilities(adapter);
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let present_mode = wgpu::PresentMode::Fifo;
    let alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;

    anyhow::ensure!(
        caps.formats.contains(&format),
        "select DirectComposition surface format: required {format:?}, supported {:?}",
        caps.formats
    );
    anyhow::ensure!(
        caps.present_modes.contains(&present_mode),
        "select DirectComposition present mode: required {present_mode:?}, supported {:?}",
        caps.present_modes
    );
    anyhow::ensure!(
        caps.alpha_modes.contains(&alpha_mode),
        "select DirectComposition alpha mode: required {alpha_mode:?}, supported {:?}",
        caps.alpha_modes
    );

    config.format = format;
    config.present_mode = present_mode;
    config.alpha_mode = alpha_mode;
    log::info!(
        "transparent Windows presentation: DirectComposition + DX12, format {format:?}, present mode {present_mode:?}, alpha mode {alpha_mode:?}"
    );
    Ok(())
}

/// The scene pass writes premultiplied-style output (opaque pixels carry
/// their own alpha, transparent clear is all-zero), so prefer PreMultiplied
/// and fall back to PostMultiplied — at alpha 0 and 1 they agree. Public so
/// alternative shells (pocket-widget) configure transparent surfaces the
/// same way.
pub fn pick_alpha_mode(
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
) -> Result<wgpu::CompositeAlphaMode> {
    let caps = surface.get_capabilities(adapter);
    for want in [
        wgpu::CompositeAlphaMode::PreMultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied,
        wgpu::CompositeAlphaMode::Inherit,
    ] {
        if caps.alpha_modes.contains(&want) {
            return Ok(want);
        }
    }
    anyhow::bail!(
        "transparent window requested but surface only supports {:?}",
        caps.alpha_modes
    )
}

fn set_mouse_capture(state: &mut WindowState, captured: bool) {
    if captured {
        let grabbed = state
            .window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Confined))
            .is_ok();
        state.window.set_cursor_visible(!grabbed);
        state.mouse_captured = grabbed;
    } else {
        let _ = state.window.set_cursor_grab(CursorGrabMode::None);
        state.window.set_cursor_visible(true);
        state.mouse_captured = false;
        state.input.clear();
    }
}

impl<G: Game> ApplicationHandler for WinitApp<G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match self.init_state(event_loop) {
                Ok(s) => self.state = Some(s),
                Err(e) => {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        state.input.on_window_event(&event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.surface_config.width = size.width.max(1);
                state.surface_config.height = size.height.max(1);
                if let Err(error) = state.configure_surface() {
                    self.error = Some(error);
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { .. } => {
                if self.config.capture_mouse && state.input.key_pressed(KeyCode::Escape) {
                    let captured = !state.mouse_captured;
                    set_mouse_capture(state, captured);
                }
            }
            WindowEvent::MouseInput {
                state: elem_state,
                button,
                ..
            } => {
                // Clicking back into the window recaptures the mouse.
                if self.config.capture_mouse && !state.mouse_captured {
                    set_mouse_capture(state, true);
                }
                if self.config.drag_window
                    && button == winit::event::MouseButton::Left
                    && elem_state.is_pressed()
                {
                    let _ = state.window.drag_window();
                }
            }
            WindowEvent::Focused(false) => set_mouse_capture(state, false),
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut()
            && state.mouse_captured
        {
            state.input.on_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.error.is_some() {
            return;
        }
        let Some(state) = &self.state else { return };
        let Some(max_fps) = self.config.max_fps else {
            state.window.request_redraw();
            return;
        };
        // Frame-paced mode: sleep until the next frame is due instead of
        // redrawing every time the loop wakes.
        let interval = Duration::from_secs_f32(1.0 / max_fps.max(1.0));
        let due = state.last_frame + interval;
        if Instant::now() >= due {
            state.window.request_redraw();
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(due));
        }
    }
}
