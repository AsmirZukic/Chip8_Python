import pygame
import sys
from chip8.core.types import Chip8State, VIDEO_WIDTH, VIDEO_HEIGHT

SCALE = 10
OFF_COLOR = (0, 0, 0)
ON_COLOR = (255, 255, 255)

class PygameWindow:
    def __init__(self):
        pygame.init()
        # Audio Setup
        self.audio_enabled = False
        self.fallback_audio = False
        self.sound = None
        self.is_playing = False
        self.fallback_proc = None # For paplay process

        # Try initializing Pygame Mixer (Suppress warnings about missing libs)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                pygame.mixer.init(44100, -16, 1, 512)
                self.audio_enabled = True
            except Exception:
                # Silent failure, check fallback
                pass

        if not self.audio_enabled:
            # Check for fallback (paplay)
            import shutil
            if shutil.which("paplay"):
                print("Audio: Using PulseAudio (paplay) fallback.")
                self.fallback_audio = True
                self.audio_enabled = True
            else:
                 print("Audio: Sound disabled (No mixer or paplay found).")
        
        self.screen = pygame.display.set_mode((VIDEO_WIDTH * SCALE, VIDEO_HEIGHT * SCALE))
        pygame.display.set_caption("Chip8 Emulator (Python/uv)")
        self.surface = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # Generate Pygame Sound if mixed is working
        if self.audio_enabled and not self.fallback_audio:
            try:
                self.sound = self._generate_beep()
            except Exception as e:
                print(f"Warning: Audio generation failed ({e}). Sound disabled.")
                self.audio_enabled = False

    def _generate_beep(self):
        if not self.audio_enabled or self.fallback_audio:
            return None
            
        import numpy as np
        # Generate 440Hz Square Wave
        sample_rate = 44100
        duration = 0.1 # 100ms buffer (looped)
        frequency = 440
        
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples)
        waveform = np.sign(np.sin(2 * np.pi * frequency * t / sample_rate))
        waveform = (waveform * 3276).astype(np.int16) # 0.1 volume
        
        return pygame.mixer.Sound(buffer=waveform)

    def update_sound(self, timer_value: int):
        """Play/Stop sound based on DT > 0."""
        if not self.audio_enabled:
            return
            
        if self.fallback_audio:
            # Fallback (paplay) logic
            # Just fire-and-forget for now to avoid blocking. 
            # Ideally we kill the process when timer stops, but paplay length is fixed by wav.
            import subprocess
            import os
            
            if timer_value > 0:
                if not self.is_playing:
                    # Start playing
                    # beep.wav expected in CWD
                    if os.path.exists("beep.wav"):
                        # We use Popen so it doesn't block
                        self.fallback_proc = subprocess.Popen(["paplay", "beep.wav"], 
                                                              stdout=subprocess.DEVNULL, 
                                                              stderr=subprocess.DEVNULL)
                        self.is_playing = True
            else:
                if self.is_playing:
                    # Stop playing (if we could, but paplay plays a file).
                    # If we made beep.wav long (infinite), we could kill it.
                    # Since it's 0.5s, just let it drain or kill it.
                    if self.fallback_proc:
                        self.fallback_proc.terminate()
                        self.fallback_proc = None
                    self.is_playing = False
        else:
            # Normal Pygame Mixer logic
            if self.sound is None: return
            
            if timer_value > 0:
                if not self.is_playing:
                    self.sound.play(-1) # Loop indefinitely
                    self.is_playing = True
            else:
                if self.is_playing:
                    self.sound.stop()
                    self.is_playing = False
        
    def handle_input(self, state: Chip8State) -> bool:
        """
        Polls events. Returns False if quit requested.
        Updates state.keys.
        """
        # Chip8 Keypad Layout:
        # 1 2 3 C
        # 4 5 6 D
        # 7 8 9 E
        # A 0 B F
        
        # Mapping to QWERTY:
        # 1 2 3 4 -> 1 2 3 C
        # q w e r -> 4 5 6 D
        # a s d f -> 7 8 9 E
        # z x c v -> A 0 B F
        
        KEY_MAP = {
            pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0xC,
            pygame.K_q: 0x4, pygame.K_w: 0x5, pygame.K_e: 0x6, pygame.K_r: 0xD,
            pygame.K_a: 0x7, pygame.K_s: 0x8, pygame.K_d: 0x9, pygame.K_f: 0xE,
            pygame.K_z: 0xA, pygame.K_x: 0x0, pygame.K_c: 0xB, pygame.K_v: 0xF
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in KEY_MAP:
                    state.keys |= (1 << KEY_MAP[event.key])
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    state.keys &= ~(1 << KEY_MAP[event.key])
            
        return True

    def render(self, state: Chip8State):
        """
        Renders the VRAM to the Pygame window.
        """
        import numpy as np
        
        # Convert VRAM (bytearray of 0/1) to a numpy array
        # Shape: (32, 64) -> Transpose to (64, 32) if needed by surfarray? 
        # Pygame surfarray is (width, height).
        vram_np = np.frombuffer(state.vram, dtype=np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH))
        
        # Transpose to (Width, Height) for Pygame
        vram_t = vram_np.T
        
        # Map 0/1 to Colors
        # Create an (W, H, 3) array
        # This is a bit manual. Alternatively, we can use a plette.
        
        # Faster approach: Blit a pre-created scalable surface.
        # Or `pygame.surfarray.blit_array`.
        
        # Let's map 0->(0,0,0) and 1->(255,255,255)
        # We can construct the uint32 packed color array for blit_array (2d)
        
        # If we use 8-bit surface with palette, it's fastest.
        # But let's stick to simple RGB for now.
        
        rgb_array = np.zeros((VIDEO_WIDTH, VIDEO_HEIGHT, 3), dtype=np.uint8)
        # where mask
        mask = (vram_t == 1)
        rgb_array[mask] = ON_COLOR
        
        pygame.surfarray.blit_array(self.surface, rgb_array)
        
        # Scale up
        scaled = pygame.transform.scale(self.surface, (VIDEO_WIDTH * SCALE, VIDEO_HEIGHT * SCALE))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()
        
    def cleanup(self):
        pygame.quit()
