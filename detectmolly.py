import numpy as np
import sounddevice as sd
from detecthealth import detect
from pydub import AudioSegment
from scipy.signal import correlate


#helper function to convert audio into NumPy files
def audio_to_numpy(segment):
    return np.array(segment.get_array_of_samples())


#Loads and stores raw audio of molly in "RawMolotov".
#incase audio doesn't load, try: mollyaudio AudioSegment.from_file("molotov_sound.wav", format="wav")
MolotovSample = audio_to_numpy(AudioSegment.from_file("molotov_sound.wav", format="wav"))

#calculatues the correlation between the two samples, based on the given threshold
def detectmolotov(segment):
    incomingsample = audio_to_numpy(segment)  
    correlation = correlate(incomingsample, MolotovSample, mode='valid')

    return np.max(correlation) > 1000000

#chatgpt code to get incoming audio (NOT TESTED)
def audiocallback(indata, frames, time, status):
    
    if status:
        print(status)
        
    audioclip = AudioSegment(
        data=indata.tobytes(),
        sample_width=indata.dtype.itemsize,
        framerate = frames, #usually 48000 ?
        channels = 1
    )
    
    if detectmolotov(audioclip):
        print("Molotov Sound Detected!")
 
    

#realtime audio callback. Gets audio and calls the "audiocallback" function, which consistently runs "detectmolotov"
#and outputs if we have a match or not.
with sd.InputStream(callback=audiocallback, channels=1, samplerate=48000):
    print("Listening for audio cue")
    sd.sleep(float('inf'))
    










