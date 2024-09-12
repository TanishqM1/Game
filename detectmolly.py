import cv2
import numpy as np
from detecthealth import detect
from pydub import AudioSegment
from scipy.signal import correlate

#helper function to convert audio into NumPy files
def audio_to_numpy(segment):
    return np.array(segment.get_array_of_samples())


#Loads and stores raw audio of molly in "RawMolotov".
#incase audio doesn't load, try: mollyaudio AudioSegment.from_file("molotov_sound.wav", format="wav")
RawMolotov = audio_to_numpy(AudioSegment.from_file("molotov_sound.wav", format="wav"))

#calculatues the correlation between the two samples, based on the given threshold
def detectmolotov(segment, threshold):
    incomingsample = audio_to_numpy(segment)  
    correlation = correlate(incomingsample, RawMolotov, mode='valid')

    if np.max(correlation)>threshold:
        return True
    return False

#chatgpt code to get incoming audio (NOT TESTED)
def callback(indata, frames, time, status):

    audioclip = AudioSegment(
        data=indata.tobytes(),
        sample_width=indata.dtype.itemsize,
        framerate = frames, #usually 48000 ?
        channels = 1
    )

    if detectmolotov(audioclip, RawMolotov, 1000000):
        print("Molly Detected")
        #need sd stream in a main method to run continously, and we will stop it here.


#NEEDED: 1) test file upload to this script, 2) need to test incoming audio, 3) need to test compare method.








