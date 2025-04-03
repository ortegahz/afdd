import matplotlib.pyplot as plt
import numpy as np

# Parameters
sampling_rate = 22325.0  # Sampling rate
N = int(sampling_rate / 50)  # Signal length, converts to number of samples for one period of 50 Hz
t = np.arange(N) / sampling_rate  # Time axis

# Construct the signal with frequency components at 50 Hz and 100 Hz
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 8000 * t)

# Perform FFT
fft_result = np.fft.fft(signal)

# Frequency axis
freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

# Amplitude spectrum
amplitude_spectrum = np.abs(fft_result)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)  # Plot the entire signal as it represents one cycle
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(freqs[:N // 2], amplitude_spectrum[:N // 2])  # Take the first half
plt.title('Frequency Spectrum (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
