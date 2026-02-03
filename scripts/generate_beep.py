import wave
import math
import struct

# Audio params
sample_rate = 44100
duration = 0.5 # 0.5 second beep
frequency = 440.0

n_samples = int(sample_rate * duration)
audio = []

for i in range(n_samples):
    # Square wave
    t = float(i) / sample_rate
    # 2 * pi * f * t
    val = 1.0 if math.sin(2.0 * math.pi * frequency * t) > 0 else -1.0
    # Scale to 16-bit
    packed_val = struct.pack('<h', int(val * 32767.0 * 0.5))
    audio.append(packed_val)

with wave.open('beep.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(b''.join(audio))
