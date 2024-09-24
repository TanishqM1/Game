import pyaudio
import numpy as np
from scipy.fftpack import fft
import time

# Set up constants
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Format for PyAudio input
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate in Hz (44.1 kHz is standard for audio)
SECONDS = 1  # Accumulate data for 1 second
MOLOTOV_FREQ_LOW = 500  # Lower bound of frequency for Molotov fire sound (adjust based on testing)
MOLOTOV_FREQ_HIGH = 5000  # Upper bound of frequency for Molotov fire sound (adjust based on testing)
THRESHOLD_FACTOR = 2  # Multiplier for background noise threshold (adjust based on testing)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Calibrating noise levels...")

# Step 1: Measure baseline background noise
background_energy = 0
calibration_time = 5  # Calibrate for 5 seconds
for _ in range(0, int(RATE / CHUNK * calibration_time)):
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    fft_data = np.abs(fft(audio_data))[:len(audio_data) // 2]
    frequencies = np.fft.fftfreq(len(audio_data), 1 / RATE)
    molotov_indices = np.where((frequencies >= MOLOTOV_FREQ_LOW) & (frequencies <= MOLOTOV_FREQ_HIGH))
    background_energy += np.sum(fft_data[molotov_indices])

background_energy /= (RATE / CHUNK * calibration_time)  # Average background energy over calibration time
print(f"Background noise energy level: {background_energy}")

# Main loop to detect Molotov sound
try:
    print("Listening for Molotov sound...")
    while True:
        start_time = time.time()
        audio_frames = []  # To accumulate data for 1 second

        # Capture multiple chunks to form 1 second of audio
        for _ in range(0, int(RATE / CHUNK * SECONDS)):
            data = stream.read(CHUNK)
            audio_frames.append(np.frombuffer(data, dtype=np.int16))

        # Combine all the chunks into one large array for FFT
        audio_data = np.hstack(audio_frames)
        
        # Perform FFT on the accumulated audio data
        fft_data = np.abs(fft(audio_data))[:len(audio_data) // 2]  # Only keep positive frequencies
        
        # Find the frequency spectrum
        frequencies = np.fft.fftfreq(len(audio_data), 1 / RATE)
        positive_frequencies = frequencies[:len(frequencies) // 2]  # Keep positive frequencies
        
        # Find if the frequencies in the Molotov range are prominent
        molotov_indices = np.where((positive_frequencies >= MOLOTOV_FREQ_LOW) & (positive_frequencies <= MOLOTOV_FREQ_HIGH))
        molotov_energy = np.sum(fft_data[molotov_indices])

        # Set threshold based on the background noise level and a factor
        threshold = background_energy * THRESHOLD_FACTOR

        # Check if the energy in the Molotov frequency range exceeds the threshold
        if molotov_energy > threshold:
            print(f"Molotov detected! 🔥 Energy: {molotov_energy}")
        else:
            print(f"No Molotov sound detected. Energy: {molotov_energy}")
        
        # Sleep for the remaining time in the second, if any
        time.sleep(max(0, SECONDS - (time.time() - start_time)))

except KeyboardInterrupt:
    print("Stopped listening...")

finally:
    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
