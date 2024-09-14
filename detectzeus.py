import numpy as np
import sounddevice as sd
from detecthealth import detect
from pydub import AudioSegment
from scipy.signal import correlate

# Helper function to convert audio into NumPy arrays
def audio_to_numpy(segment):
    return np.array(segment.get_array_of_samples())

# Loads and stores raw audio of the Zeus in "RawZeus".
ZeusSample = audio_to_numpy(AudioSegment.from_file("zeus_sound.wav", format="wav"))

# Calculates the correlation between the two samples, based on the given threshold
def detectzeus(segment):
    incomingsample = audio_to_numpy(segment)  
    correlation = correlate(incomingsample, ZeusSample, mode='valid')

    return np.max(correlation) > 1000000

# ChatGPT code to get incoming audio (NOT TESTED)
def audiocallback(indata, frames, time, status):
    
    if status:
        print(status)
        
    audioclip = AudioSegment(
        data=indata.tobytes(),
        sample_width=indata.dtype.itemsize,
        framerate = frames,  # usually 48000
        channels = 1
    )
    
    if detectzeus(audioclip):
        print("Zeus Sound Detected!")

    # Double checks that health is being dropped after Zeus sound is made
    health = detect()  # Call your health detection function
    if health < previous_health:  # Compare with previously stored health
        print("Player is likely being damaged by the Zeus.")
    else:
        print("No health drop detected yet.")

# Initialize previous health for comparison
previous_health = detect()  

# Realtime audio callback. Gets audio and calls the "audiocallback" function, which consistently runs "detectzeus"
# and outputs if we have a match or not.
with sd.InputStream(callback=audiocallback, channels=1, samplerate=48000):
    print("Listening for audio cue")
    sd.sleep(float('inf'))