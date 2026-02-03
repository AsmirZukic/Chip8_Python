from chip8.core.types import Chip8State
from chip8.core.opcodes import execute_opcode

def test_fx0a_wait_release():
    """Test Fx0A (Get Key) wait-for-release logic."""
    state = Chip8State()
    state.pc = 0x200
    opcode = 0xF00A # LD V0, K
    
    # 1. No keys pressed -> Wait
    state.keys = 0
    execute_opcode(state, opcode)
    assert state.pc == 0x1FE # Decremented (200 - 2)
    assert state.wait_key is None
    
    # Reset PC for next step simulation
    state.pc = 0x200 
    
    # 2. Key 5 pressed -> Start waiting for release
    state.keys = (1 << 5)
    execute_opcode(state, opcode)
    assert state.pc == 0x1FE # Decremented
    assert state.wait_key == 5
    assert state.registers[0] == 0 # Not set yet
    
    # Reset PC
    state.pc = 0x200
    
    # 3. Key 5 still pressed -> Keep waiting
    state.keys = (1 << 5)
    execute_opcode(state, opcode)
    assert state.pc == 0x1FE
    assert state.wait_key == 5
    
    # Reset PC
    state.pc = 0x200
    
    # 4. Key 5 released -> Finish
    state.keys = 0
    execute_opcode(state, opcode)
    assert state.pc == 0x200 # Advanced (did not decrement) -> Actually execute_opcode doesn't auto-increment, 
                             # but usually the loop does. 
                             # Wait, execute_opcode implementation for Fx0A sets pc -= 2 if waiting.
                             # If finished, it does nothing to PC (so it stays 0x200, effectively "advancing" relative to the loop's increment)
                             # Wait, my test runner usually increments PC *before* execution?
                             # Let's check test_opcodes.py context.
                             # In the app, PC is incremented, THEN execute is called? Or Execute, then increment?
                             # Usually Fetch (PC+=2), Decode, Execute.
                             # If Execute does PC-=2, effective change is 0.
                             # If Execute does nothing, effective change is +2.
                             # In this unit test, I manually set PC. execute_opcode modifies it relative to current.
                             # If success, execute_opcode does NOT modify PC. So PC remains 0x200.
    assert state.pc == 0x200
    assert state.wait_key is None
    assert state.registers[0] == 5
