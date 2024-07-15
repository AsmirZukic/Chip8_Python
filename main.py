import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

class Chip8: 
    MEMORY_SIZE = 4096
    VIDEO_WIDTH = 64
    VIDEO_HEIGHT = 32
    VIDEO_SIZE = VIDEO_WIDTH * VIDEO_HEIGHT
    FONTSET_START_ADDRESS = 0x50
    ROM_START_ADDRESS = 0x200
    
    def __init__(self):
        self.memory = np.zeros(self.MEMORY_SIZE, dtype=np.uint16)
        self.video_memory = np.zeros(self.VIDEO_SIZE, dtype=np.bool_)
        self.registers = np.zeros(16, dtype=np.uint8)
        self.stack = []
        self.program_counter = self.ROM_START_ADDRESS
        self.stack_counter = 0
        self.index_register = 0 
        self.delay_counter = 0
        self.sound_counter = 0
        self.fontset = [
            0xF0, 0x90, 0x90, 0x90, 0xF0, # 0
            0x20, 0x60, 0x20, 0x20, 0x70, # 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0, # 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0, # 3
            0x90, 0x90, 0xF0, 0x10, 0x10, # 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0, # 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0, # 6
            0xF0, 0x10, 0x20, 0x40, 0x40, # 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0, # 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0, # 9
            0xF0, 0x90, 0xF0, 0x90, 0x90, # A
            0xE0, 0x90, 0xE0, 0x90, 0xE0, # B
            0xF0, 0x80, 0x80, 0x80, 0xF0, # C
            0xE0, 0x90, 0x90, 0x90, 0xE0, # D
            0xF0, 0x80, 0xF0, 0x80, 0xF0, # E
            0xF0, 0x80, 0xF0, 0x80, 0x80  # F
        ]
        self.opcode_map = {
            0x0: self.handle_0x0,
            0x1: self.op1NNN,
            0x6: self.op6xnn,
            0xA: self.opAnnn,
            0xD: self.opDxyn,
        }

    def _load_fontset(self): 
        self.memory[self.FONTSET_START_ADDRESS:self.FONTSET_START_ADDRESS + len(self.fontset)] = self.fontset

    def _load_rom_data(self, rom_path): 
        with open(rom_path, "rb") as file: 
            rom_data = file.read()
            start = self.ROM_START_ADDRESS
            self.memory[start:start + len(rom_data)] = np.frombuffer(rom_data, dtype=np.uint8)

    def load_rom(self, rom_path): 
        self._load_fontset()
        self._load_rom_data(rom_path)

    def fetch(self): 
        if self.program_counter + 1 >= self.MEMORY_SIZE:
            raise IndexError("Program counter out of bounds.")
        first_byte = self.memory[self.program_counter]
        second_byte = self.memory[self.program_counter + 1]
        self.program_counter += 2
        return (first_byte << 8) | second_byte
    
    def decode_and_execute(self):
        instruction = self.fetch()
        opcode = (instruction & 0xF000) >> 12
        if opcode in self.opcode_map:
            self.opcode_map[opcode](instruction)
        else:
            raise ValueError(f"Unknown opcode: {hex(opcode)}")

    def run(self):
        try:
            while True:
                self.decode_and_execute()
        except (IndexError, ValueError) as e:
            print(f"Error encountered: {e}")
            # Handle error or exit

    def handle_0x0(self, instruction):
        if instruction == 0x00E0:
            self.op00E0()
        # Handle other 0x0 opcodes if necessary

    def op00E0(self):
        self.video_memory = np.zeros(self.VIDEO_SIZE, dtype=np.bool_)

    def op6xnn(self, instruction):
        X = (instruction & 0x0F00) >> 8
        NN = instruction & 0x00FF
        self.registers[X] = NN 

    def opAnnn(self, instruction):
        self.index_register = instruction & 0x0FFF

    def op1NNN(self, instruction): 
        self.program_counter = instruction & 0x0FFF

    def opDxyn(self, instruction):
        x = self.registers[(instruction & 0x0F00) >> 8]
        y = self.registers[(instruction & 0x00F0) >> 4]
        height = instruction & 0x000F
        self.registers[0xF] = 0  # Reset collision flag
        for row in range(height):
            sprite = self.memory[self.index_register + row]
            for col in range(8):
                if (sprite & (0x80 >> col)) != 0:
                    vx = (x + col) % self.VIDEO_WIDTH
                    vy = (y + row) % self.VIDEO_HEIGHT
                    index = vy * self.VIDEO_WIDTH + vx
                    if self.video_memory[index]:
                        self.registers[0xF] = 1
                    self.video_memory[index] ^= True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chip8.py <path_to_rom>")
        sys.exit(1)
    
    emulator = Chip8()
    emulator.load_rom(sys.argv[1])
    emulator.run()
