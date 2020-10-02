import pyaudio
import numpy as np

p = pyaudio.PyAudio()

fs = 44100       # sampling rate, Hz, must be integer
duration = 100   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float
channels = 2

# generate samples, note conversion to float32 array
t = 2*np.pi*np.arange(fs*duration)/fs
data = np.stack(
    [
        16 * np.sin(f * t) ** 3,
        13 * np.cos(f * t) - 5 * np.cos(2 * f * t)  - 2 *np.cos(3 * f * t) - np.cos(4 * f * t)
    ],
    axis=-1
)
data = (1 + 0.1 * np.sin(1 * t))[:, np.newaxis] * data
data /= data.max()
data = data.reshape(-1)
samples = (data).astype(np.float32)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=channels,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively) 
stream.write(samples)

stream.stop_stream()
stream.close()

p.terminate()