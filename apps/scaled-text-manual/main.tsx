// Diagnostic-only scaled baked text fixture for the native uihost.
// Build with --density=1 or --density=2; uihost's --scale only sizes the window.
import { createSignal } from "solid-js";
import { mount } from "@pocketjs/framework";
import { Text, View } from "@pocketjs/framework/components";
import { onButtonPress, onFrame } from "@pocketjs/framework/lifecycle";
import { BTN } from "@pocketjs/framework/input";

const INITIAL_CLIP_WIDTH = 62;
const INITIAL_VIEWPORT_OFFSET = -12;

function ScaledTextManual() {
  const [clipWidth, setClipWidth] = createSignal(INITIAL_CLIP_WIDTH);
  const [viewportOffset, setViewportOffset] = createSignal(INITIAL_VIEWPORT_OFFSET);

  onFrame((buttons) => {
    if (buttons & BTN.LEFT) setClipWidth((width) => Math.max(8, width - 1));
    if (buttons & BTN.RIGHT) setClipWidth((width) => Math.min(200, width + 1));
    if (buttons & BTN.UP) setViewportOffset((offset) => Math.max(-48, offset - 1));
    if (buttons & BTN.DOWN) setViewportOffset((offset) => Math.min(20, offset + 1));
  });
  onButtonPress(BTN.CROSS, () => {
    setClipWidth(INITIAL_CLIP_WIDTH);
    setViewportOffset(INITIAL_VIEWPORT_OFFSET);
  });

  return (
    <View class="relative w-full h-full bg-[#101820]">
      <Text class="absolute left-[16] top-[8] text-xs text-[#90a8bb]">
        SCALED TEXT / LEFT-RIGHT CLIP / UP-DOWN EDGE / Z RESET
      </Text>

      <Text class="absolute left-[16] top-[34] text-xs text-[#90a8bb]">IDENTITY</Text>
      <Text class="absolute left-[104] top-[31] text-sm text-white">MMMM</Text>

      <Text class="absolute left-[16] top-[65] text-xs text-[#90a8bb]">LOGICAL 2X</Text>
      <Text
        class="absolute left-[16] top-[82] text-sm text-[#ffe09b]"
        style={{ width: 180, height: 18, scale: 2, originX: -0.5, originY: -0.5 }}
      >
        MMMM
      </Text>

      <Text class="absolute left-[16] top-[130] text-xs text-[#90a8bb]">MOVING INNER CLIP</Text>
      <View
        class="absolute left-[16] top-[150] h-[38] overflow-hidden bg-[#243847]"
        style={{ width: clipWidth() }}
      >
        <Text
          class="absolute left-0 top-0 text-sm text-[#ffe09b]"
          style={{ width: 180, height: 18, scale: 2, originX: -0.5, originY: -0.5 }}
        >
          MMMM
        </Text>
      </View>
      <Text class="absolute left-[250] top-[162] text-xs text-white">
        {`clip width: ${clipWidth()}`}
      </Text>

      <Text class="absolute left-[16] top-[204] text-xs text-[#90a8bb]">VIEWPORT LEFT EDGE</Text>
      <Text
        class="absolute left-0 top-[223] text-sm text-[#8ce7d0]"
        style={{ width: 80, height: 18, scale: 2, originX: -0.5, originY: -0.5, translateX: viewportOffset() }}
      >
        M
      </Text>
      <Text class="absolute left-[250] top-[232] text-xs text-white">
        {`viewport offset: ${viewportOffset()}`}
      </Text>
    </View>
  );
}

mount(() => <ScaledTextManual />);
