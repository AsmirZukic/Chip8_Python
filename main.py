import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

class Chip8: 
    def __init__(self):
        self.memory = np.zeros( 4096 , dtype=np.uint8 )
        self.video_memory = np.zeros( 32*64, dtype=np.bool )
        self.registers = np.zeros( 16, dtype=np.uint8 )
        self.stack = []
        self.program_counter = 0x200
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

    def _load_fontset(self): 
        for i in range(len(self.fontset)):
            self.memory[0x50 + i] = self.fontset[i]

    def _load_rom_data(self, rom_path): 
        index = 0x200
        with open(rom_path, "rb") as file: 
            while True:
                bit = file.read(1)
                if not bit: 
                    break 
                self.memory[index] = ord(bit)  # Store byte value in memory
                index += 1

    def load_rom(self): 
        self._load_fontset()
        self._load_rom_data(sys.argv[1])
        print(self.memory)

    def fetch(self): 
        #Instructions are two bytes but the memory only stores 8bits (1 byte)
        #We need to fetch the two bytes to get the full instruction and then combine them together
        first_byte = self.memory[self.program_counter] 
        second_byte = self.memory[self.program_counter + 1]

        instruction = (first_byte << 8) | second_byte

        #We're going to increment the program counter by 2 here
        self.program_counter += 2

        return instruction
    
    def decode_and_execute(self):
        instruction = self.fetch()
        first_nibble = instruction >> 12

        if first_nibble == 0x0:
            if instruction == 0x00E0:
                self.op00E0()
            # Add other 0x0--- opcodes if needed
        elif first_nibble == 0x6:
            self.op6xnn(instruction)
        elif first_nibble == 0xA:
            self.opAnnn(instruction)
        elif first_nibble == 0xD:
            self.opDxyn(instruction)
        # Add other opcode handlers as needed


    #Clear screen
    def op00E0(self):
        self.video_memory = np.zeros( 32*64, dtype=np.bool )

    def op6xnn(self, instruction):
        #Get X
        X = (instruction & 0x0F00) >> 8
        #Get NN
        NN = instruction & 0x00FF
        #set the value of Vx to be NN 
        self.registers[X] = NN 

    #Set index register to value NNN
    def opAnnn(self, instruction):
        self.index_register = instruction & 0x0FFF

    #Jump to NNN
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
                    vx = (x + col) % 64
                    vy = (y + row) % 32
                    if self.video_memory[vy, vx]:
                        self.registers[0xF] = 1
                    self.video_memory[vy, vx] ^= True
        

if __name__ == "__main__":
    emulator = Chip8()
    emulator.load_rom()
    while True:
        emulator.decode_and_execute()
