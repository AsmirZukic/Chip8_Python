import sys
import numpy as np
import pygame

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
        self.video_memory = np.zeros(shape=[self.VIDEO_WIDTH, self.VIDEO_HEIGHT], dtype=np.bool_)
        self.registers = np.zeros(16, dtype=np.uint8)
        self.stack = []
        self.program_counter = self.ROM_START_ADDRESS
        self.stack_counter = 0
        self.index_register = 0
        self.delay_counter = 0
        self.sound_counter = 0
        self.fontset = [
            0xF0,0x90,0x90,0x90,0xF0,  # 0
            0x20,0x60,0x20,0x20,0x70,  # 1
            0xF0,0x10,0xF0,0x80,0xF0,  # 2
            0xF0,0x10,0xF0,0x10,0xF0,  # 3
            0x90,0x90,0xF0,0x10,0x10,  # 4
            0xF0,0x80,0xF0,0x10,0xF0,  # 5
            0xF0,0x80,0xF0,0x90,0xF0,  # 6
            0xF0,0x10,0x20,0x40,0x40,  # 7
            0xF0,0x90,0xF0,0x90,0xF0,  # 8
            0xF0,0x90,0xF0,0x10,0xF0,  # 9
            0xF0,0x90,0xF0,0x90,0x90,  # A
            0xE0,0x90,0xE0,0x90,0xE0,  # B
            0xF0,0x80,0x80,0x80,0xF0,  # C
            0xE0,0x90,0x90,0x90,0xE0,  # D
            0xF0,0x80,0xF0,0x80,0xF0,  # E
            0xF0,0x80,0xF0,0x80,0x80,  # F
        ]

        self.opcode_map = {
            0x0: {
                0x00E0: self.op00E0,
                0x00EE: self.op00EE
            },
            0x1: self.op1NNN,
            0x2: self.op2NNN,
            0x3: self.op3XNN,
            0x4: self.op4XNN,
            0x5: self.op5XY0,
            0x6: self.op6xnn,
            0x7: self.op7XNN,
            0x8: {
                0x0: self.op8XY0,
                0x1: self.op8XY1,
                0x2: self.op8XY2,
                0x3: self.op8XY3,
                0x4: self.op8XY4,
                0x5: self.op8XY5,
                0x6: self.op8XY6,
                0x7: self.op8XY7,
                0xE: self.op8XYE
            },
            0x9: self.op9XY0,
            0xA: self.opAnnn,
            0xD: self.opDxyn,
            0xF: {
                0x65: self.opFX65,
                0x55: self.opFX55,
                0x33: self.opFX33,
                0x1E: self.opFX1E
            }
        }


    def _load_fontset(self):
        self.memory[
            self.FONTSET_START_ADDRESS : self.FONTSET_START_ADDRESS + len(self.fontset)
        ] = self.fontset

    def _load_rom_data(self, rom_path):
        with open(rom_path, "rb") as file:
            rom_data = file.read()
            start = self.ROM_START_ADDRESS
            self.memory[start : start + len(rom_data)] = np.frombuffer(
                rom_data, dtype=np.uint8
            )

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
        sub_opcode = instruction & 0x00FF
        sub_opcode_8 = instruction & 0x000F  # For 0x8 instructions

        # Retrieve the handler for the main opcode
        handler = self.opcode_map.get(opcode)
        
        if handler:
            if isinstance(handler, dict):
                # Handle nested dictionary for specific opcodes
                if opcode == 0x8:
                    # Special handling for 0x8 instructions
                    sub_handler = handler.get(sub_opcode_8, lambda instr: print(f"Unknown 0x8 sub-opcode: {hex(sub_opcode_8)}"))
                    sub_handler(instruction)  # Call with instruction
                elif opcode == 0xF:
                    # Special handling for 0xF instructions
                    sub_handler = handler.get(sub_opcode, lambda instr: print(f"Unknown 0xF sub-opcode: {hex(sub_opcode)}"))
                    sub_handler(instruction)  # Call with instruction
                else:
                    # Handle other nested dictionaries
                    sub_handler = handler.get(sub_opcode, lambda instr: print(f"Unknown sub-opcode: {hex(sub_opcode)}"))
                    sub_handler(instruction)  # Call with instruction
            else:
                # Direct function call with instruction
                handler(instruction)
        else:
            print(f"Unknown opcode: {hex(opcode)}")

        return self.get_video_memory()

    def run(self):
        try:
            while True:
                self.decode_and_execute()

        except (IndexError, ValueError) as e:
            print(f"Error encountered: {e}")
            # Handle error or exit

    def get_X(self, instruction):
        return (instruction & 0x0F00) >> 8

    def get_Y(self, instruction):
        return (instruction & 0x00F0) >> 4

    def get_NN(self, instruction):
        return instruction & 0x00FF

    def get_NNN(self, instruction):
        return instruction & 0x0FFF

    def op00E0(self, instruction):
        self.video_memory.fill(0)

    def op3XNN(self, instruction):
        if self.registers[self.get_X(instruction)] == self.get_NN(instruction):
            self.program_counter += 2
        else:
            pass

    def op4XNN(self, instruction):
        if self.registers[self.get_X(instruction)] != self.get_NN(instruction):
            self.program_counter += 2
        else:
            pass

    def op5XY0(self, instruction):
        if (self.registers[self.get_X(instruction)] == self.registers[self.get_Y(instruction)]):
            self.program_counter += 2
        else:
            pass

    def op9XY0(self, instruction):
        if (self.registers[self.get_X(instruction)] != self.registers[self.get_Y(instruction)]):
            self.program_counter += 2
        else:
            pass

    def op6xnn(self, instruction):
        X = (instruction & 0x0F00) >> 8
        NN = instruction & 0x00FF
        self.registers[X] = NN

    def op7XNN(self, instruction):
        X = (instruction & 0x0F00) >> 8
        NN = instruction & 0x00FF

        self.registers[X] += NN

    def opAnnn(self, instruction):
        self.index_register = instruction & 0x0FFF

    def op1NNN(self, instruction):
        self.program_counter = instruction & 0x0FFF

    def op2NNN(self, instruction):
        self.stack.append(self.program_counter)
        self.program_counter = instruction & 0x0FFF

    def op00EE(self, instruction):
        self.program_counter = self.stack.pop()

    def op8XY0(self, instruction):
        self.registers[self.get_X(instruction)] = self.registers[self.get_Y(instruction)]

    def op8XY1(self, instruction):
        self.registers[self.get_X(instruction)] |= self.registers[self.get_Y(instruction)]

    def op8XY2(self, instruction):
        self.registers[self.get_X(instruction)] &= self.registers[self.get_Y(instruction)]

    def op8XY3(self, instruction):
        self.registers[self.get_X(instruction)] ^= self.registers[self.get_Y(instruction)]

    def op8XY4(self, instruction):
        x = self.get_X(instruction)
        y = self.get_Y(instruction)
        
        # Cast the registers to uint16 to handle overflow properly
        vx = np.uint16(self.registers[x])
        vy = np.uint16(self.registers[y])
        
        sum_value = vx + vy
    
        # Store only the lower 8 bits of the sum in VX
        self.registers[x] = sum_value & 0xFF

        self.registers[0xF] = 1 if sum_value > 0xFF else 0
        

    def op8XY5(self, instruction):
        if self.registers[self.get_X(instruction)] < self.registers[self.get_Y(instruction)] :
            self.registers[self.get_X(instruction)] -= self.registers[self.get_Y(instruction)]
            self.registers[0xF] = 0
        else:
            self.registers[self.get_X(instruction)] -= self.registers[self.get_Y(instruction)]
            self.registers[0xF] = 1

    def op8XY6(self, instruction):
        original_x = self.registers[self.get_X(instruction)]
        self.registers[self.get_X(instruction)] = self.registers[self.get_X(instruction)] >> 1
        
        if original_x & 0x01:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

    def op8XY7(self, instruction):
        X = self.get_X(instruction)
        Y = self.get_Y(instruction)

        self.registers[X] = self.registers[Y] - self.registers[X]

        if self.registers[Y] >= self.registers[X]:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

    def op8XYE(self, instruction):
        original_x = self.registers[self.get_X(instruction)]

        self.registers[self.get_X(instruction)] = (self.registers[self.get_X(instruction)] << 1) & 0xFF

        if original_x & 0x80:
            self.registers[0xF] = 1
        else:
            self.registers[0xF] = 0

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
                    if self.video_memory[vx, vy]:
                        self.registers[0xF] = 1
                    self.video_memory[vx, vy] ^= True

    def opFX65(self, instruction):
        X = self.get_X(instruction)
        for index in range(0, X + 1):
            self.registers[index] = self.memory[self.index_register + index]

    def opFX55(self, instruction):
        X = self.get_X(instruction)
        for index in range(0, X + 1):
            self.memory[self.index_register + index] = self.registers[index]

    def opFX33(self, instruction):
        X = self.get_X(instruction)
        value = self.registers[X]

        # Compute hundreds, tens, and ones digits
        hundreds = value // 100
        tens = (value // 10) % 10
        ones = value % 10

        # Store these digits in memory at addresses I, I+1, and I+2
        self.memory[self.index_register] = hundreds
        self.memory[self.index_register + 1] = tens
        self.memory[self.index_register + 2] = ones

    def opFX1E(self, instruction):
        X = self.registers[self.get_X(instruction)]

        self.index_register += X

    def get_video_memory(self):
        return (self.video_memory.astype(np.uint8)) * 255


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chip8.py <path_to_rom>")
        sys.exit(1)

    emulator = Chip8()
    emulator.load_rom(sys.argv[1])

    pygame.init()
    display = pygame.display.set_mode((640, 320))
    pygame.display.set_caption("Chip8")
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        video_memory = emulator.decode_and_execute()
        video_surface = pygame.surfarray.make_surface(video_memory)
        scaled_surface = pygame.transform.scale(video_surface, display.get_size())
        display.blit(scaled_surface, (0, 0))

        pygame.display.update()

    pygame.quit()
