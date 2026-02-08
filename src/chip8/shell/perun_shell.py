"""Perun-based shell for Chip8 emulator.

Streams video to Perun server and receives input from connected clients.
"""

import sys
sys.path.insert(0, '/home/asmir/Projects/Perun/sdk/python')

from chip8.core.types import Chip8State, VIDEO_WIDTH, VIDEO_HEIGHT
from perun_sdk import PerunConnection, VideoFramePacket, InputEventPacket, PacketType


class PerunWindow:
    """
    Perun-based shell that mirrors PygameWindow interface.
    
    Connects to a Perun server and:
    - Sends video frames (64x32 RGBA)
    - Receives input events (16-bit button bitmask)
    """
    
    def __init__(self, address: str = "127.0.0.1:8080", use_tcp: bool = True, delta_only: bool = True):
        self._conn = PerunConnection()
        self._connected = False
        self._last_vram = None
        self._delta_only = delta_only
        
        if use_tcp:
            # Parse host:port
            if ':' in address:
                host, port_str = address.rsplit(':', 1)
                port = int(port_str)
            else:
                host = address
                port = 8080
            self._connected = self._conn.connect_tcp(host, port)
        else:
            self._connected = self._conn.connect_unix(address)
        
        if not self._connected:
            raise ConnectionError(f"Failed to connect to Perun server at {address}")
        
        print(f"[PerunWindow] Connected to Perun server at {address} (delta_only={delta_only})")
    
    def handle_input(self, state: Chip8State) -> bool:
        """
        Poll for input packets from Perun server.
        Updates state.keys bitmask.
        Returns False if connection closed.
        """
        if not self._connected:
            return False
        
        # Non-blocking receive loop
        while True:
            result = self._conn.receive_packet_header()
            if not result:
                break
            
            header, payload = result
            if header.type == PacketType.InputEvent:
                pkt = InputEventPacket.deserialize(payload)
                # Update Chip8 keys from button bitmask
                state.keys = pkt.buttons
        
        return self._conn.is_connected()
    
    def render(self, state: Chip8State):
        """
        Convert VRAM (64x32 monochrome) to RGBA and send as VideoFramePacket.
        Only sends if VRAM changed or delta_only is disabled.
        """
        if not self._connected:
            return
        
        # Delta-only optimization: check if VRAM changed
        if self._delta_only:
            if self._last_vram is not None and self._last_vram == state.vram:
                return
            self._last_vram = list(state.vram) # Clone vram
        
        # Convert VRAM to RGBA (64x32 * 4 bytes)
        rgba = bytearray(VIDEO_WIDTH * VIDEO_HEIGHT * 4)
        
        for i, pixel in enumerate(state.vram):
            offset = i * 4
            if pixel:
                # White (on)
                rgba[offset] = 255
                rgba[offset + 1] = 255
                rgba[offset + 2] = 255
                rgba[offset + 3] = 255
            else:
                # Black (off)
                rgba[offset] = 0
                rgba[offset + 1] = 0
                rgba[offset + 2] = 0
                rgba[offset + 3] = 255
        
        packet = VideoFramePacket(
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            compressed_data=bytes(rgba)
        )
        # Use async sending - returns immediately, never blocks
        self._conn.send_video_frame_async(packet)
    
    def update_sound(self, timer_value: int):
        """
        Audio handling stub.
        
        TODO: Implement audio streaming via AudioChunkPacket when
        the native client supports audio playback.
        """
        pass
    
    def cleanup(self):
        """Close the connection."""
        if self._conn:
            self._conn.close()
            self._connected = False
        print("[PerunWindow] Connection closed")
