from .types import Chip8State

def execute_opcode(state: Chip8State, opcode: int):
    """
    Decodes and executes a single instruction.
    Mutates the state in place.
    """
    # Simple table-based dispatch or if/else for now.
    # We will expand this as we implement groups.
    
    nibble1 = (opcode & 0xF000) >> 12
    
    if opcode == 0x00E0:
        op_00E0(state, opcode)
    elif opcode == 0x00EE:
        op_00EE(state, opcode)
    elif nibble1 == 0x1:
        op_1NNN(state, opcode)
    elif nibble1 == 0x2:
        op_2NNN(state, opcode)
    elif nibble1 == 0x3:
        op_3XNN(state, opcode)
    elif nibble1 == 0x4:
        op_4XNN(state, opcode)
    elif nibble1 == 0x5:
        op_5XY0(state, opcode)
    elif nibble1 == 0x6:
        op_6XNN(state, opcode)
    elif nibble1 == 0x7:
        op_7XNN(state, opcode)
    elif nibble1 == 0x8:
        op_8XYN(state, opcode)
    elif nibble1 == 0x9:
        op_9XY0(state, opcode)
    elif nibble1 == 0xA:
        op_ANNN(state, opcode)
    elif nibble1 == 0xB:
        op_BNNN(state, opcode)
    elif nibble1 == 0xC:
        op_CXNN(state, opcode)
    elif nibble1 == 0xD:
        op_DXYN(state, opcode)
    elif nibble1 == 0xE:
        op_Exxx(state, opcode)
    elif nibble1 == 0xF:
        op_Fxxx(state, opcode)
    # ... placeholders
    else:
        # Unknown opcode dispatch
        pass

# --- Opcode Implementations ---

import random


def op_BNNN(state: Chip8State, opcode: int):
    """JP V0, addr: Jump to location nnn + V0."""
    # Standard Behavior: Jump to nnn + V0
    nnn = opcode & 0x0FFF
    state.pc = (nnn + state.registers[0]) & 0xFFFF 

def op_CXNN(state: Chip8State, opcode: int):
    """RND Vx, byte: Set Vx = random byte AND kk."""
    x = (opcode & 0x0F00) >> 8
    kk = opcode & 0x00FF
    state.registers[x] = random.randint(0, 255) & kk

# ... (Previous functions)

def op_8XYN(state: Chip8State, opcode: int):
    """Arithmetic and Logic operations."""
    x = (opcode & 0x0F00) >> 8
    y = (opcode & 0x00F0) >> 4
    sub = opcode & 0x000F
    
    if sub == 0x0: # LD Vx, Vy
        state.registers[x] = state.registers[y]
    elif sub == 0x1: # OR Vx, Vy
        state.registers[x] |= state.registers[y]
        state.registers[0xF] = 0 # Quirk: VF Reset
    elif sub == 0x2: # AND Vx, Vy
        state.registers[x] &= state.registers[y]
        state.registers[0xF] = 0 # Quirk: VF Reset
    elif sub == 0x3: # XOR Vx, Vy
        state.registers[x] ^= state.registers[y]
        state.registers[0xF] = 0 # Quirk: VF Reset
    elif sub == 0x4: # ADD Vx, Vy
        # Carry flag is VF
        val_x = state.registers[x]
        val_y = state.registers[y]
        res = val_x + val_y
        state.registers[x] = res & 0xFF
        state.registers[0xF] = 1 if res > 255 else 0
    elif sub == 0x5: # SUB Vx, Vy
        # VF is NOT borrow (1 if x >= y)
        val_x = state.registers[x]
        val_y = state.registers[y]
        state.registers[x] = (val_x - val_y) & 0xFF
        state.registers[0xF] = 1 if val_x >= val_y else 0
    elif sub == 0x6: # SHR Vx
        # VF is LSB. Shift right by 1.
        # Legacy: Vy is shifted into Vx.
        val_x = state.registers[y] # Use Vy
        sb = val_x & 0x1
        state.registers[x] = val_x >> 1
        state.registers[0xF] = sb
    elif sub == 0x7: # SUBN Vx, Vy
        # Vx = Vy - Vx. VF is NOT borrow (1 if y >= x)
        val_x = state.registers[x]
        val_y = state.registers[y]
        state.registers[x] = (val_y - val_x) & 0xFF
        state.registers[0xF] = 1 if val_y >= val_x else 0
    elif sub == 0xE: # SHL Vx
        # VF is MSB. Shift left by 1.
        # Legacy: Vy is shifted into Vx.
        val_x = state.registers[y] # Use Vy
        sb = 1 if (val_x & 0x80) else 0
        state.registers[x] = (val_x << 1) & 0xFF
        state.registers[0xF] = sb

def op_Exxx(state: Chip8State, opcode: int):
    """Skip next instruction if key is pressed/not pressed."""
    x = (opcode & 0x0F00) >> 8
    sub = opcode & 0x00FF
    key = state.registers[x]
    
    if sub == 0x9E: # SKP Vx
        if (state.keys & (1 << key)):
             state.pc += 2
    elif sub == 0xA1: # SKNP Vx
        if not (state.keys & (1 << key)):
             state.pc += 2

def op_Fxxx(state: Chip8State, opcode: int):
    """Miscellaneous instructions."""
    x = (opcode & 0x0F00) >> 8
    sub = opcode & 0x00FF
    
    if sub == 0x07: # LD Vx, DT
        state.registers[x] = state.delay_timer
    elif sub == 0x0A: # LD Vx, K (Wait for key)
        # Check if we are already waiting for a release
        if state.wait_key is not None:
             # Check if the specific key is still pressed
            current_pressed = (state.keys >> state.wait_key) & 1
            if current_pressed:
                # Still pressed, keep waiting (don't advance)
                state.pc -= 2
            else:
                # Released! Store key and finish.
                state.registers[x] = state.wait_key
                state.wait_key = None
        else:
            # Not yet waiting for a specific key. Check for ANY key press.
            if state.keys == 0:
                state.pc -= 2 # Keep waiting
            else:
                # Key pressed! Find which one.
                for k in range(16):
                    if (state.keys >> k) & 1:
                        # Found a key. Start waiting for its release.
                        state.wait_key = k
                        state.pc -= 2 # Stay on this instruction to check release next cycle
                        break
    elif sub == 0x15: # LD DT, Vx
        state.delay_timer = state.registers[x]
    elif sub == 0x18: # LD ST, Vx
        state.sound_timer = state.registers[x]
    elif sub == 0x1E: # ADD I, Vx
        state.index = (state.index + state.registers[x]) & 0xFFFF
    elif sub == 0x29: # LD F, Vx
        digit = state.registers[x]
        state.index = 0x50 + (digit * 5)
    elif sub == 0x33: # LD B, Vx
        val = state.registers[x]
        state.memory[state.index]     = val // 100
        state.memory[state.index + 1] = (val // 10) % 10
        state.memory[state.index + 2] = val % 10
    elif sub == 0x55: # LD [I], Vx
        for i in range(x + 1):
            state.memory[state.index + i] = state.registers[i]
        # Quirk: Increment I
        state.index = (state.index + x + 1) & 0xFFFF
    elif sub == 0x65: # LD Vx, [I]
        for i in range(x + 1):
            state.registers[i] = state.memory[state.index + i]
        # Quirk: Increment I
        state.index = (state.index + x + 1) & 0xFFFF



def op_ANNN(state: Chip8State, opcode: int):
    """LD I, addr: Set I = nnn."""
    state.index = opcode & 0x0FFF

def op_2NNN(state: Chip8State, opcode: int):
    """CALL addr: Call subroutine at nnn."""
    state.stack.append(state.pc)
    state.pc = opcode & 0x0FFF

def op_3XNN(state: Chip8State, opcode: int):
    """SE Vx, byte: Skip next instruction if Vx == kk."""
    x = (opcode & 0x0F00) >> 8
    kk = opcode & 0x00FF
    if state.registers[x] == kk:
        state.pc += 2

def op_4XNN(state: Chip8State, opcode: int):
    """SNE Vx, byte: Skip next instruction if Vx != kk."""
    x = (opcode & 0x0F00) >> 8
    kk = opcode & 0x00FF
    if state.registers[x] != kk:
        state.pc += 2

def op_5XY0(state: Chip8State, opcode: int):
    """SE Vx, Vy: Skip next instruction if Vx == Vy."""
    x = (opcode & 0x0F00) >> 8
    y = (opcode & 0x00F0) >> 4
    if state.registers[x] == state.registers[y]:
        state.pc += 2

def op_9XY0(state: Chip8State, opcode: int):
    """SNE Vx, Vy: Skip next instruction if Vx != Vy."""
    x = (opcode & 0x0F00) >> 8
    y = (opcode & 0x00F0) >> 4
    if state.registers[x] != state.registers[y]:
        state.pc += 2

# --- Opcode Implementations ---
# ... (Previous functions)

from .types import VIDEO_WIDTH, VIDEO_HEIGHT 

def op_DXYN(state: Chip8State, opcode: int):
    """DRW Vx, Vy, nibble: Display n-byte sprite starting at memory location I at (Vx, Vy)."""
    vx = (opcode & 0x0F00) >> 8
    vy = (opcode & 0x00F0) >> 4
    n = opcode & 0x000F
    
    x_coord = state.registers[vx] % VIDEO_WIDTH
    y_coord = state.registers[vy] % VIDEO_HEIGHT
    
    state.registers[0xF] = 0
    
    for row in range(n):
        # Stop drawing if we go off the bottom edge (standard behavior for some interpreters, 
        # though wrapping behavior varies. Using clipping for Y as per common spec)
        if y_coord + row >= VIDEO_HEIGHT:
            break
            
        sprite_byte = state.memory[state.index + row]
        
        for col in range(8):
            if x_coord + col >= VIDEO_WIDTH:
                break
                
            if (sprite_byte & (0x80 >> col)) != 0:
                pixel_index = (y_coord + row) * VIDEO_WIDTH + (x_coord + col)
                
                # Check for collision
                if state.vram[pixel_index] == 1:
                    state.registers[0xF] = 1
                
                # XOR
                state.vram[pixel_index] ^= 1

def op_00E0(state: Chip8State, opcode: int):
    """CLS: Clear the display."""
    # Fast clear
    state.vram[:] = b'\x00' * len(state.vram)

def op_00EE(state: Chip8State, opcode: int):
    """RET: Return from a subroutine."""
    if not state.stack:
        return # Error handling?
    state.pc = state.stack.pop()

def op_1NNN(state: Chip8State, opcode: int):
    """JP addr: Jump to location nnn."""
    state.pc = opcode & 0x0FFF

def op_6XNN(state: Chip8State, opcode: int):
    """LD Vx, byte: Set Vx = kk."""
    x = (opcode & 0x0F00) >> 8
    kk = opcode & 0x00FF
    state.registers[x] = kk

def op_7XNN(state: Chip8State, opcode: int):
    """ADD Vx, byte: Set Vx = Vx + kk."""
    x = (opcode & 0x0F00) >> 8
    kk = opcode & 0x00FF
    # Handle wrapping (uint8)
    state.registers[x] = (state.registers[x] + kk) & 0xFF
