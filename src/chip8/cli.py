import sys
import time
import argparse
from chip8.core.types import Chip8State
from chip8.core.opcodes import execute_opcode

# Clock speed: 500Hz
CLOCK_SPEED = 500
MS_PER_CYCLE = 1.0 / CLOCK_SPEED

def load_rom(state: Chip8State, path: str):
    with open(path, "rb") as f:
        data = f.read()
        # Load into memory starting at 0x200
        start = 0x200
        state.memory[start:start+len(data)] = data
    # Loading fontset (not implemented in this minimal version yet, but needed for text)
    # We should add fontset loading to State init or here.
    
    # Simple hardcoded fontset for now
    FONTSET = [
        0xF0,0x90,0x90,0x90,0xF0, # 0
        0x20,0x60,0x20,0x20,0x70, # 1
        0xF0,0x10,0xF0,0x80,0xF0, # 2
        0xF0,0x10,0xF0,0x10,0xF0, # 3
        0x90,0x90,0xF0,0x10,0x10, # 4
        0xF0,0x80,0xF0,0x10,0xF0, # 5
        0xF0,0x80,0xF0,0x90,0xF0, # 6
        0xF0,0x10,0x20,0x40,0x40, # 7
        0xF0,0x90,0xF0,0x90,0xF0, # 8
        0xF0,0x90,0xF0,0x10,0xF0, # 9
        0xF0,0x90,0xF0,0x90,0x90, # A
        0xE0,0x90,0xE0,0x90,0xE0, # B
        0xF0,0x80,0x80,0x80,0xF0, # C
        0xE0,0x90,0x90,0x90,0xE0, # D
        0xF0,0x80,0xF0,0x80,0xF0, # E
        0xF0,0x80,0xF0,0x80,0x80  # F
    ]
    # Fontset usually at 0x50
    state.memory[0x50:0x50+len(FONTSET)] = bytearray(FONTSET)


def main():
    parser = argparse.ArgumentParser(description="Chip8 Emulator")
    parser.add_argument("rom", help="Path to ROM file")
    parser.add_argument("-p", "--perun", action="store_true", 
                        help="Use Perun for rendering (streams to perun-client)")
    parser.add_argument("--tcp", type=str, default="127.0.0.1:8080",
                        help="Perun server TCP address (default: 127.0.0.1:8080)")
    parser.add_argument("--unix", type=str, default=None,
                        help="Perun server Unix socket path (overrides --tcp)")
    parser.add_argument("--no-delta", action="store_true", help="Disable delta-frame optimization (send all frames)")
    args = parser.parse_args()

    rom_path = args.rom
    
    # 1. Init Core
    state = Chip8State()
    load_rom(state, rom_path)
    
    # 2. Init Shell
    if args.perun:
        from chip8.shell.perun_shell import PerunWindow
        if args.unix:
            window = PerunWindow(address=args.unix, use_tcp=False, delta_only=not args.no_delta)
        else:
            window = PerunWindow(address=args.tcp, use_tcp=True, delta_only=not args.no_delta)
    else:
        from chip8.shell.window import PygameWindow
        window = PygameWindow()
    
    # 3. Game Loop
    running = True
    last_time = time.time()
    accumulator = 0.0
    timer_accumulator = 0.0
    
    try:
        while running:
            # Time delta
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Input (Imperative Shell)
            running = window.handle_input(state)
            if not running:
                break
            
            # CPU Cylces (Functional Core)
            # Accumulator pattern for stable clock
            accumulator += dt
            timer_accumulator += dt

            while accumulator >= MS_PER_CYCLE:
                # Fetch
                # Safety check
                if state.pc >= 4096:
                    break
                    
                opcode = (state.memory[state.pc] << 8) | state.memory[state.pc + 1]
                state.pc += 2
                
                # Decode & Execute
                execute_opcode(state, opcode)
                
                accumulator -= MS_PER_CYCLE

            # Update timers (60Hz)
            should_render = False
            while timer_accumulator >= 1.0 / 60.0:
                timer_accumulator -= 1.0 / 60.0
                
                # Decrement timers directly
                if state.delay_timer > 0:
                    state.delay_timer -= 1
                if state.sound_timer > 0:
                    state.sound_timer -= 1
                
                # Audio Control
                window.update_sound(state.sound_timer)
                
                should_render = True
            
            if should_render:
                window.render(state)
            
            # Sleep briefly when we have no work to do
            # This prevents CPU spinning and allows I/O to be processed
            # Without this sleep, the tight loop starves socket operations,
            # causing severe input latency (7+ seconds)
            if accumulator < MS_PER_CYCLE and timer_accumulator < 1.0 / 60.0:
                time.sleep(0.001)  # 1ms sleep when idle
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Crash at PC={hex(state.pc)}: {e}")
        raise
    finally:
        window.cleanup()

if __name__ == "__main__":
    main()

