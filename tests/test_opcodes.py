import json
import pytest
from pathlib import Path
from chip8.core.types import Chip8State, VIDEO_WIDTH, VIDEO_HEIGHT
from chip8.core.opcodes import execute_opcode

def load_test_cases():
    path = Path(__file__).parent / "data/test_cases.json"
    with open(path, "r") as f:
        return json.load(f)

def parse_value(v):
    if isinstance(v, str) and v.startswith("0x"):
        return int(v, 16)
    return v

def setup_state(state_dict):
    state = Chip8State()
    # Apply initial overrides
    for key, value in state_dict.items():
        value = parse_value(value)
        
        if key == "vram":
            if value == "FILL:1":
                state.vram[:] = b'\x01' * len(state.vram)
            elif value == "FILL:0":
                 state.vram[:] = b'\x00' * len(state.vram)
        elif key == "registers" and isinstance(value, dict):
            # Patch specific registers
            for idx, val in value.items():
                state.registers[int(idx)] = parse_value(val)
        elif key == "memory" and isinstance(value, dict):
            # Patch memory at specific addresses
            for addr, bytes_list in value.items():
                start_addr = parse_value(addr)
                for i, byte_val in enumerate(bytes_list):
                   state.memory[start_addr + i] = int(byte_val)
        elif key == "stack":
            state.stack = [parse_value(x) for x in value]
        elif hasattr(state, key):
             setattr(state, key, value)
    return state

@pytest.mark.parametrize("test_case", load_test_cases())
def test_opcode_execution(test_case):
    state = setup_state(test_case.get("initial", {}))
    opcode = parse_value(test_case["opcode"])

    # Simulate fetch step
    state.pc += 2
    
    execute_opcode(state, opcode)
    
    expected = test_case.get("expected", {})
    for key, value in expected.items():
        if key == "vram":
            # Simple check for now
            if value == "FILL:0":
                assert all(b == 0 for b in state.vram)
        elif key == "registers" and isinstance(value, dict):
             for idx, val in value.items():
                 val = parse_value(val)
                 actual = state.registers[int(idx)]
                 assert actual == val, f"Register V{idx} mismatch. Expected {val}, got {actual}"
        elif key == "memory" and isinstance(value, dict):
             for addr, bytes_list in value.items():
                start_addr = parse_value(addr)
                for i, expected_byte in enumerate(bytes_list):
                   actual_byte = state.memory[start_addr + i]
                   assert actual_byte == expected_byte, f"Memory mismatch at {hex(start_addr+i)}. Expected {expected_byte}, got {actual_byte}"
        elif key == "stack":
            # Stack is a list of ints
            expected_stack = [parse_value(x) for x in value]
            assert state.stack == expected_stack, "Stack mismatch"
        else:
            value = parse_value(value)
            actual = getattr(state, key)
            if key == "pc" and "pc" not in test_case.get("expected", {}):
                 # Opcode execution usually increments PC by 2, but our test runner doesn't have auto-increment logic.
                 # The Opcode function mutates PC directly for Jumps/Skips.
                 pass
            assert actual == value, f"Failed {test_case['name']}: {key} expected {value}, got {actual}"
