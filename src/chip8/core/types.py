from dataclasses import dataclass, field
from typing import List

VIDEO_WIDTH = 64
VIDEO_HEIGHT = 32
MEMORY_SIZE = 4096
REGISTER_COUNT = 16

@dataclass
class Chip8State:
    """
    Mutable state of the Chip8 CPU.
    Optimized for performance (using bytearrays where possible).
    """
    # System Memory (4KB)
    memory: bytearray = field(default_factory=lambda: bytearray(MEMORY_SIZE))
    
    # Registers V0-VF
    registers: bytearray = field(default_factory=lambda: bytearray(REGISTER_COUNT))
    
    # Index Register (I)
    index: int = 0
    
    # Program Counter (PC)
    pc: int = 0x200
    
    # Stack (16 levels usually, but list is fine for Python)
    stack: List[int] = field(default_factory=list)
    
    # Timers (60Hz)
    delay_timer: int = 0
    sound_timer: int = 0
    
    # Video Memory (64x32 monochrome)
    # Stored as 0 or 1 bytes for easier digestion by the shell
    vram: bytearray = field(default_factory=lambda: bytearray(VIDEO_WIDTH * VIDEO_HEIGHT))
    
    # Keypad State
    # 16 keys, stored as a bitmask (1 = pressed, 0 = released) to save space/copying
    keys: int = 0
    
    # Key Wait State (for Fx0A)
    # If not None, we are waiting for this specific key (0-15) to be released.
    wait_key: int | None = None

    def reset(self):
        """Reset state to initial values."""
        self.memory = bytearray(MEMORY_SIZE)
        self.registers = bytearray(REGISTER_COUNT)
        self.index = 0
        self.pc = 0x200
        self.stack.clear()
        self.delay_timer = 0
        self.sound_timer = 0
        self.vram = bytearray(VIDEO_WIDTH * VIDEO_HEIGHT)
        self.keys = 0
