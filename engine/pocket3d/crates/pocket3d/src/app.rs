//! Windowed application loop (winit 0.30) with fixed-step simulation.
//!
//! Games implement [`Game`]; the same object can also be driven headlessly
//! by calling `tick`/`compose` directly (see OpenStrike's script mode).

use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use anyhow::Context;
use anyhow::Result;
use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
#[cfg(target_os = "windows")]
use winit::platform::windows::WindowAttributesExtWindows;
#[cfg(target_os = "windows")]
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{CursorGrabMode, CursorIcon, ResizeDirection, Window, WindowId, WindowLevel};

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
#[cfg(target_os = "windows")]
use crate::presentation::{
    LOGICAL_OUTPUT_FORMAT, PHYSICAL_SURFACE_FORMAT, TransparentPresentation,
};
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
    /// The game can veto a press via [`Game::drag_at`] (interactive
    /// overlays); the default hook always allows.
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
    /// Called before [`Game::init`] and before each rendered frame with the
    /// window's physical surface size and OS logical-to-physical scale factor.
    /// Existing games can ignore desktop metrics; UI hosts can use them to
    /// keep logical layout/input coordinates separate from the render target.
    fn window_metrics(&mut self, _physical_size: (u32, u32), _scale_factor: f64) {}
    /// Called once per rendered frame before fixed ticks (mouse look etc.).
    fn frame(&mut self, dt: f32, input: &Input);
    /// Fixed-step simulation.
    fn tick(&mut self, dt: f32, input: &Input);
    /// Prepare renderer state for the next frame, outside any render pass.
    /// Default: nothing.
    fn prepare_render(&mut self, gpu: &Gpu, renderer: &mut Renderer) {
        let (_, _) = (gpu, renderer);
    }
    /// Provide the frame to draw. `time` is seconds since launch. `size` is
    /// the physical surface size in pixels, matching [`Input::cursor`].
    fn compose(&mut self, alpha: f32, time: f32, size: (u32, u32)) -> (&Scene, &Camera, &Hud);
    /// Record extra passes over the finished frame (UI overlays, composite
    /// effects) before present. `format` is the target's texture format and
    /// `size` is the physical surface size in pixels.
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
    /// Left-press drag policy: whether this press starts the OS window
    /// drag. Consulted only when [`AppConfig::drag_window`] is on and the
    /// press missed every borderless resize zone; `cursor` is the pointer
    /// position in the space [`Input::cursor`] reports (physical pixels).
    /// Return false to keep the press with the game (interactive overlays)
    /// instead of moving the window. Default: always allow — today's
    /// behavior.
    fn drag_at(&mut self, cursor: Vec2) -> bool {
        let _ = cursor;
        true
    }
    /// Return true to quit.
    fn wants_exit(&self) -> bool {
        false
    }
}

/// Width of the borderless resize affordance in logical pixels.
const BORDERLESS_RESIZE_HIT_ZONE_LOGICAL: f64 = 7.0;

fn manual_borderless_resize_enabled(resizable: bool, decorations: bool) -> bool {
    resizable && !decorations
}

/// Whether a left press that missed every resize zone starts the OS window
/// drag. The [`Game::drag_at`] veto is consulted only when dragging is on
/// and a cursor position exists; a press before any `CursorMoved` (no
/// position to hand the game) keeps the unconditional drag it had before
/// the hook existed.
fn native_drag_requested(drag_window: bool, cursor: Option<Vec2>, game: &mut impl Game) -> bool {
    drag_window && cursor.is_none_or(|cursor| game.drag_at(cursor))
}

/// Classify a pointer position against the physical client bounds of a
/// borderless window. Winit reports cursor positions and window sizes in
/// physical pixels, so the logical hit zone is scaled before comparison.
/// Corners are checked first because their hit regions overlap the edges.
fn classify_resize_direction(
    position: (f64, f64),
    size: (u32, u32),
    scale_factor: f64,
    resizable: bool,
) -> Option<ResizeDirection> {
    if !resizable
        || size.0 == 0
        || size.1 == 0
        || !scale_factor.is_finite()
        || scale_factor <= 0.0
        || !position.0.is_finite()
        || !position.1.is_finite()
    {
        return None;
    }

    let width = f64::from(size.0);
    let height = f64::from(size.1);
    let (x, y) = position;
    if x < 0.0 || x > width || y < 0.0 || y > height {
        return None;
    }

    let hit_zone = BORDERLESS_RESIZE_HIT_ZONE_LOGICAL * scale_factor;
    let near_left = x <= hit_zone;
    let near_right = x >= width - hit_zone;
    let near_top = y <= hit_zone;
    let near_bottom = y >= height - hit_zone;

    match (near_left, near_right, near_top, near_bottom) {
        (true, _, true, _) => Some(ResizeDirection::NorthWest),
        (_, true, true, _) => Some(ResizeDirection::NorthEast),
        (true, _, _, true) => Some(ResizeDirection::SouthWest),
        (_, true, _, true) => Some(ResizeDirection::SouthEast),
        (true, _, _, _) => Some(ResizeDirection::West),
        (_, true, _, _) => Some(ResizeDirection::East),
        (_, _, true, _) => Some(ResizeDirection::North),
        (_, _, _, true) => Some(ResizeDirection::South),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResizeAction {
    Noop,
    Suspend,
    Reconfigure { size: (u32, u32) },
    Restore { size: (u32, u32) },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResizeState {
    viewport_size: Option<(u32, u32)>,
}

impl ResizeState {
    fn new(size: (u32, u32)) -> Self {
        Self {
            viewport_size: Self::is_non_zero(size).then_some(size),
        }
    }

    fn apply(&mut self, size: (u32, u32)) -> ResizeAction {
        if !Self::is_non_zero(size) {
            return self
                .viewport_size
                .take()
                .map(|_| ResizeAction::Suspend)
                .unwrap_or(ResizeAction::Noop);
        }

        if self.viewport_size == Some(size) {
            return ResizeAction::Noop;
        }

        let action = if self.viewport_size.is_none() {
            ResizeAction::Restore { size }
        } else {
            ResizeAction::Reconfigure { size }
        };
        self.viewport_size = Some(size);
        action
    }

    fn viewport_size(self) -> Option<(u32, u32)> {
        self.viewport_size
    }

    fn is_suspended(self) -> bool {
        self.viewport_size.is_none()
    }

    fn is_non_zero(size: (u32, u32)) -> bool {
        size.0 != 0 && size.1 != 0
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
    #[cfg(target_os = "windows")]
    transparent_presentation: Option<TransparentPresentation>,
    gpu: Gpu,
    renderer: Renderer,
    // Keep the HWND alive until after the surface and any composition target.
    window: Arc<Window>,
    input: Input,
    timestep: FixedTimestep,
    start: Instant,
    last_frame: Instant,
    mouse_captured: bool,
    resize_state: ResizeState,
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

    fn handle_resize(&mut self, size: (u32, u32)) -> Result<ResizeAction> {
        let action = self.resize_state.apply(size);
        match action {
            ResizeAction::Noop => {}
            ResizeAction::Suspend => {
                // Keep the surface at its last valid size. In particular, do
                // not configure a 0x0 swapchain or allocate zero-sized views.
                self.renderer.suspend();
                #[cfg(target_os = "windows")]
                if self.direct_composition.is_some() {
                    self.transparent_presentation = None;
                }
            }
            ResizeAction::Reconfigure { size } | ResizeAction::Restore { size } => {
                self.surface_config.width = size.0;
                self.surface_config.height = size.1;
                self.configure_surface()?;
                // Surface configuration owns the existing DComp visual; this
                // only replaces attachments whose dimensions changed.
                self.renderer.resize(&self.gpu, size);
                #[cfg(target_os = "windows")]
                if self.direct_composition.is_some() {
                    if let Some(presentation) = self.transparent_presentation.as_mut() {
                        presentation.resize(&self.gpu, size);
                    } else {
                        self.transparent_presentation =
                            Some(TransparentPresentation::new(&self.gpu, size));
                    }
                }
                if matches!(action, ResizeAction::Restore { .. }) {
                    self.last_frame = Instant::now();
                }
            }
        }
        Ok(action)
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
        self.game
            .window_metrics((px.width, px.height), window.scale_factor());
        self.game.init(&gpu, &mut renderer)?;

        #[cfg(target_os = "windows")]
        let transparent_presentation = if self.config.transparent && px.width != 0 && px.height != 0
        {
            Some(TransparentPresentation::new(&gpu, (px.width, px.height)))
        } else {
            None
        };

        let mut state = WindowState {
            surface,
            surface_config,
            #[cfg(target_os = "windows")]
            direct_composition,
            #[cfg(target_os = "windows")]
            transparent_presentation,
            gpu,
            renderer,
            window,
            input: Input::default(),
            timestep: FixedTimestep::new(self.config.tick_hz),
            start: Instant::now(),
            last_frame: Instant::now(),
            mouse_captured: false,
            resize_state: ResizeState::new((px.width, px.height)),
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
        if state.resize_state.is_suspended() {
            return;
        }

        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32();
        state.last_frame = now;

        let physical_size = state
            .resize_state
            .viewport_size()
            .expect("active window must have a non-zero viewport");
        self.game
            .window_metrics(physical_size, state.window.scale_factor());
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
        #[cfg(target_os = "windows")]
        let surface_view = if self.config.transparent {
            frame.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("transparent physical surface unorm view"),
                format: Some(PHYSICAL_SURFACE_FORMAT),
                ..Default::default()
            })
        } else {
            frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        #[cfg(not(target_os = "windows"))]
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        #[cfg(target_os = "windows")]
        let logical_view = if self.config.transparent {
            state
                .transparent_presentation
                .as_ref()
                .expect("active transparent window must have a logical output")
                .view()
        } else {
            &surface_view
        };
        #[cfg(not(target_os = "windows"))]
        let logical_view = &surface_view;
        let size = state
            .resize_state
            .viewport_size()
            .expect("active window must have a non-zero viewport");
        let (scene, camera, hud) = self.game.compose(
            state.timestep.alpha(),
            state.start.elapsed().as_secs_f32(),
            size,
        );
        state
            .renderer
            .render(&state.gpu, logical_view, size, scene, camera, hud);
        let mut encoder =
            state
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("overlay"),
                });
        #[cfg(target_os = "windows")]
        let logical_format = if self.config.transparent {
            LOGICAL_OUTPUT_FORMAT
        } else {
            state.surface_config.format
        };
        #[cfg(not(target_os = "windows"))]
        let logical_format = state.surface_config.format;
        self.game
            .overlay(&state.gpu, &mut encoder, logical_view, logical_format, size);
        #[cfg(target_os = "windows")]
        if self.config.transparent {
            state
                .transparent_presentation
                .as_ref()
                .expect("active transparent window must have a logical output")
                .pack(&mut encoder, &surface_view);
        }
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
    add_transparent_surface_view_format(&mut config.view_formats);
    log::info!(
        "transparent Windows presentation: DirectComposition + DX12, format {format:?}, present mode {present_mode:?}, alpha mode {alpha_mode:?}"
    );
    Ok(())
}

#[cfg(target_os = "windows")]
fn add_transparent_surface_view_format(view_formats: &mut Vec<wgpu::TextureFormat>) {
    if !view_formats.contains(&PHYSICAL_SURFACE_FORMAT) {
        view_formats.push(PHYSICAL_SURFACE_FORMAT);
    }
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
                if let Err(error) = state.handle_resize((size.width, size.height)) {
                    self.error = Some(error);
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { .. } => {
                if self.config.capture_mouse && state.input.key_pressed(KeyCode::Escape) {
                    let captured = !state.mouse_captured;
                    set_mouse_capture(state, captured);
                }
                #[cfg(target_os = "windows")]
                if self.config.transparent
                    && state.direct_composition.is_some()
                    && state.input.key_pressed(KeyCode::F9)
                {
                    if let Some(presentation) = state.transparent_presentation.as_mut() {
                        let use_exact_load = presentation.toggle();
                        log::info!(
                            "presentation pack: {}",
                            if use_exact_load { "exact" } else { "filtered" }
                        );
                        state.window.request_redraw();
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let manual_resize = manual_borderless_resize_enabled(
                    self.config.resizable,
                    self.config.decorations,
                );
                if manual_resize {
                    let size = state.window.inner_size();
                    let resize_direction = classify_resize_direction(
                        (position.x, position.y),
                        (size.width, size.height),
                        state.window.scale_factor(),
                        manual_resize,
                    );
                    state.window.set_cursor(
                        resize_direction
                            .map(CursorIcon::from)
                            .unwrap_or(CursorIcon::Default),
                    );
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if manual_borderless_resize_enabled(self.config.resizable, self.config.decorations)
                {
                    state.window.set_cursor(CursorIcon::Default);
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
                if button == MouseButton::Left && elem_state == ElementState::Pressed {
                    let manual_resize = manual_borderless_resize_enabled(
                        self.config.resizable,
                        self.config.decorations,
                    );
                    let resize_direction = if manual_resize {
                        state.input.cursor().and_then(|position| {
                            let size = state.window.inner_size();
                            classify_resize_direction(
                                (f64::from(position.x), f64::from(position.y)),
                                (size.width, size.height),
                                state.window.scale_factor(),
                                manual_resize,
                            )
                        })
                    } else {
                        None
                    };
                    if let Some(direction) = resize_direction {
                        // Resize hit zones take precedence over the normal
                        // interior drag gesture. Native resizing owns the
                        // pointer loop until the button is released.
                        let _ = state.window.drag_resize_window(direction);
                    } else if native_drag_requested(
                        self.config.drag_window,
                        state.input.cursor(),
                        &mut self.game,
                    ) {
                        let _ = state.window.drag_window();
                    }
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
        if state.resize_state.is_suspended() {
            return;
        }
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

#[cfg(test)]
mod tests {
    use super::{
        Game, ResizeAction, ResizeDirection, ResizeState, classify_resize_direction,
        manual_borderless_resize_enabled, native_drag_requested,
    };
    #[cfg(target_os = "windows")]
    use super::{PHYSICAL_SURFACE_FORMAT, add_transparent_surface_view_format};
    use crate::camera::Camera;
    use crate::gpu::Gpu;
    use crate::hud::Hud;
    use crate::input::Input;
    use crate::renderer::Renderer;
    use crate::scene::Scene;
    use glam::Vec2;

    const WINDOW: (u32, u32) = (450, 600);

    /// Games that exercise only the drag policy: the required trait
    /// methods are never called by [`native_drag_requested`].
    struct DefaultGame;
    struct RejectingGame;

    impl Game for DefaultGame {
        fn init(&mut self, _: &Gpu, _: &mut Renderer) -> anyhow::Result<()> {
            Ok(())
        }
        fn frame(&mut self, _: f32, _: &Input) {}
        fn tick(&mut self, _: f32, _: &Input) {}
        fn compose(&mut self, _: f32, _: f32, _: (u32, u32)) -> (&Scene, &Camera, &Hud) {
            unreachable!("only drag_at is exercised")
        }
    }

    impl Game for RejectingGame {
        fn init(&mut self, _: &Gpu, _: &mut Renderer) -> anyhow::Result<()> {
            Ok(())
        }
        fn frame(&mut self, _: f32, _: &Input) {}
        fn tick(&mut self, _: f32, _: &Input) {}
        fn compose(&mut self, _: f32, _: f32, _: (u32, u32)) -> (&Scene, &Camera, &Hud) {
            unreachable!("only drag_at is exercised")
        }
        fn drag_at(&mut self, _cursor: Vec2) -> bool {
            false
        }
    }

    #[test]
    fn default_hook_preserves_native_drag_at_every_cursor() {
        // No override → every press drags, with or without a cursor
        // position, exactly as before the hook existed.
        let mut game = DefaultGame;
        assert!(game.drag_at(Vec2::new(120.0, 300.0)));
        assert!(native_drag_requested(true, None, &mut game));
        assert!(native_drag_requested(
            true,
            Some(Vec2::new(120.0, 300.0)),
            &mut game
        ));
    }

    #[test]
    fn game_rejection_blocks_native_drag() {
        let mut game = RejectingGame;
        assert!(!native_drag_requested(
            true,
            Some(Vec2::new(120.0, 300.0)),
            &mut game
        ));
        // No cursor position means there is nothing to reject against; the
        // legacy unconditional drag stands (see `native_drag_requested`).
        assert!(native_drag_requested(true, None, &mut game));
    }

    #[test]
    fn drag_window_disabled_never_starts_native_drag() {
        assert!(!native_drag_requested(
            false,
            Some(Vec2::new(120.0, 300.0)),
            &mut DefaultGame
        ));
        assert!(!native_drag_requested(false, None, &mut RejectingGame));
    }

    #[test]
    fn manual_borderless_resize_requires_resizable_undecorated_window() {
        assert!(manual_borderless_resize_enabled(true, false));
        assert!(!manual_borderless_resize_enabled(false, false));
        assert!(!manual_borderless_resize_enabled(true, true));
    }

    #[test]
    fn resize_hit_testing_classifies_each_edge() {
        assert_eq!(
            classify_resize_direction((3.0, 300.0), WINDOW, 1.0, true),
            Some(ResizeDirection::West)
        );
        assert_eq!(
            classify_resize_direction((447.0, 300.0), WINDOW, 1.0, true),
            Some(ResizeDirection::East)
        );
        assert_eq!(
            classify_resize_direction((225.0, 3.0), WINDOW, 1.0, true),
            Some(ResizeDirection::North)
        );
        assert_eq!(
            classify_resize_direction((225.0, 597.0), WINDOW, 1.0, true),
            Some(ResizeDirection::South)
        );
    }

    #[test]
    fn resize_hit_testing_gives_corners_precedence_over_edges() {
        assert_eq!(
            classify_resize_direction((3.0, 3.0), WINDOW, 1.0, true),
            Some(ResizeDirection::NorthWest)
        );
        assert_eq!(
            classify_resize_direction((447.0, 3.0), WINDOW, 1.0, true),
            Some(ResizeDirection::NorthEast)
        );
        assert_eq!(
            classify_resize_direction((3.0, 597.0), WINDOW, 1.0, true),
            Some(ResizeDirection::SouthWest)
        );
        assert_eq!(
            classify_resize_direction((447.0, 597.0), WINDOW, 1.0, true),
            Some(ResizeDirection::SouthEast)
        );
    }

    #[test]
    fn resize_hit_testing_uses_logical_hit_zone_and_ignores_interior() {
        assert_eq!(
            classify_resize_direction((14.0, 300.0), WINDOW, 2.0, true),
            Some(ResizeDirection::West)
        );
        assert_eq!(
            classify_resize_direction((14.1, 300.0), WINDOW, 2.0, true),
            None
        );
        assert_eq!(
            classify_resize_direction((225.0, 300.0), WINDOW, 1.0, true),
            None
        );
        assert_eq!(
            classify_resize_direction((-1.0, 300.0), WINDOW, 1.0, true),
            None
        );
    }

    #[test]
    fn resize_hit_testing_is_disabled_when_window_is_not_resizable() {
        for position in [(3.0, 3.0), (3.0, 300.0), (225.0, 3.0), (225.0, 300.0)] {
            assert_eq!(
                classify_resize_direction(position, WINDOW, 1.0, false),
                None
            );
        }
    }

    #[test]
    fn zero_size_resize_suspends_without_changing_surface_viewport() {
        let mut state = ResizeState::new((450, 600));

        assert_eq!(state.apply((0, 0)), ResizeAction::Suspend);
        assert!(state.is_suspended());
        assert_eq!(state.apply((0, 0)), ResizeAction::Noop);
    }

    #[test]
    fn non_zero_resize_updates_viewport_and_same_size_is_a_noop() {
        let mut state = ResizeState::new((450, 600));

        assert_eq!(
            state.apply((900, 600)),
            ResizeAction::Reconfigure { size: (900, 600) }
        );
        assert_eq!(state.viewport_size(), Some((900, 600)));
        assert_eq!(state.apply((900, 600)), ResizeAction::Noop);
    }

    #[test]
    fn restore_after_minimize_requires_one_non_zero_reconfigure() {
        let mut state = ResizeState::new((450, 600));

        assert_eq!(state.apply((0, 600)), ResizeAction::Suspend);
        assert_eq!(
            state.apply((700, 500)),
            ResizeAction::Restore { size: (700, 500) }
        );
        assert_eq!(state.apply((700, 500)), ResizeAction::Noop);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn transparent_surface_view_format_policy_is_additive_and_idempotent() {
        let mut view_formats = vec![wgpu::TextureFormat::Rgba8Unorm];
        add_transparent_surface_view_format(&mut view_formats);
        add_transparent_surface_view_format(&mut view_formats);

        assert_eq!(
            view_formats,
            vec![wgpu::TextureFormat::Rgba8Unorm, PHYSICAL_SURFACE_FORMAT]
        );
    }
}
