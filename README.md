# Modern Chip-8 Emulator

A precise, test-driven Chip-8 emulator built with modern Python engineering practices.

## Overview

This project is a complete rewrite of a basic Chip-8 emulator I created years ago when I first started dabbling in emulation. Returning to the project with more experience, I wanted to rebuild it from the ground up to demonstrate how much better it could be done using modern software engineering practices.

The result is a highly maintainable, testable emulator that replaces the original ad-hoc implementation with a professional architecture:

*   **Functional Core**: The CPU logic (`src/chip8/core`) is pure and side-effect free, receiving a state and mutation instructions. This allows for 100% deterministic unit testing.
*   **Imperative Shell**: The outer layer (`src/chip8/shell`) handles I/O, rendering (Pygame), and audio, bridging the gap between the pure core and the user's machine.
*   **Data-Driven Testing**: We use `pytest` with a massive JSON dataset to verify every opcode against known-good behaviors, ensuring regression-free development.
*   **Modern Tooling**: Built with `uv` for lightning-fast dependency management and environment isolation.

## Features

- **High Accuracy**: Passes **all** Timendus test ROMs, including:
    - `1-chip8-logo` (Graphics)
    - `2-ibm-logo` (Standard Opcode set)
    - `3-corax+` (Opcode Edge Cases)
    - `4-flags` (Math/Carry Flags)
    - `5-quirks` (Strict Logic/Memory/Shift compliance)
    - `6-keypad` (Input State Machine)
    - `7-beep` (Audio Timing)
- **Robust Audio**: Uses `pygame.mixer` with a graceful PulseAudio (`paplay`) fallback for systems with missing libraries.
- **Configurable Quirks**: Currently tuned for strict COSMAC VIP compatibility (Legacy Shifts, Standard Jumps).
- **Controls**: Full QWERTY mapping to 4x4 Hex Keypad.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# Clone the repo
git clone <url>
cd Chip8_Python

# Install dependencies
uv sync
```

## Usage

To run a ROM:

```bash
uv run chip8 path/to/rom.ch8
```

### Controls

| Chip-8 Keypad | QWERTY Keyboard |
|---------------|-----------------|
| 1 2 3 C       | 1 2 3 4         |
| 4 5 6 D       | Q W E R         |
| 7 8 9 E       | A S D F         |
| A 0 B F       | Z X C V         |

## Development

The project is architected for testability. To run the full suite of data-driven tests:

```bash
PYTHONPATH=src uv run pytest
```

## License

MIT License. See `LICENSE` for details.