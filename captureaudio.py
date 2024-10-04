import soundcard as sc
import soundfile as sf

with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=48000) as mic:
   #saves a numpy array of audio in "data".
    data = mic.record(numframes=48000*3)
    
  


