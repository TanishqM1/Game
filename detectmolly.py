import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
import soundcard as sc
import soundfile as sf

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#pathways to positive and negative clips
POS = os.path.join('data', 'Updated_Molotov_Clips_1sec')
NEG = os.path.join('data', 'Updated_Non_Molotov_Clips_1sec')


#filtering .wav files in each folder
pos2 = tf.data.Dataset.list_files(POS+'\*.wav')
neg2 = tf.data.Dataset.list_files(NEG+'\*.wav')

#idk
positives = tf.data.Dataset.zip((pos2, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos2)))))
negatives = tf.data.Dataset.zip((neg2, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg2)))))
data = positives.concatenate(negatives)

#finding lengths of all data files
lengths = []
for file in os.listdir(os.path.join('data', 'Updated_Molotov_Clips_1sec')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Updated_Molotov_Clips_1sec', file))
    lengths.append(len(tensor_wave))

tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

print("check")
print(tf.math.reduce_mean(lengths))
print(tf.math.reduce_min(lengths))
print(tf.math.reduce_max(lengths))

#preprocessinng files to convert to spectogram
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

#tweak frame_length, frame_step & wav[6500] based on testing

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
print(spectrogram.shape)

#preprocessing data strings
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

print(len(data))
#train machine using 70% of clips and test on the renaming 30%.

train = data.take(64)
test = data.skip(64).take(28)

#show spectogram shape needed, for a positive match.
samples, labels = train.as_numpy_iterator().next()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#build deep learning model

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# print(model.summary())

#train model

# epochs can be tweaked. Larger = more accurate
hist = model.fit(train, epochs=1, validation_data=test)

X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)

yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

print(tf.math.reduce_sum(yhat))
print(tf.math.reduce_sum(y_test))

print(yhat)
print(y_test)



def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

from itertools import groupby

results = {}
# for file in os.listdir(os.path.join('data', 'test_clips')):
#     FILEPATH = os.path.join('data','test_clips', file)
    
#     wav = load_mp3_16k_mono(FILEPATH)
#     audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=47999, batch_size=1)
#     audio_slices = audio_slices.map(preprocess_mp3)
#     audio_slices = audio_slices.batch(64)
#     print(audio_slices)
    
#     yhat = model.predict(audio_slices)
    
#     results[file] = yhat

# class_preds = {}
# for file, logits in results.items():
#     class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]


# postprocessed = {}
# for file, scores in class_preds.items():
#     postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

import csv

# with open('results.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['recording', 'molly_calls'])
#     for key, value in postprocessed.items():
#         writer.writerow([key, value])
        

# folder_path = "incomingaudio"
# os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

# results2={}
# # Record audio and save to separate files
# for i in range(10):
#     # Define the complete file path for each recording
#     file_path = os.path.join(folder_path, f"TestRecording_{i + 1}.wav")

#     with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=48000) as mic:
#         # Save a numpy array of audio in "data".
#         data = mic.record(numframes=48000)
#         sf.write(file=file_path, data=data[:, 0], samplerate=48000)

# #pre-process our recorded "incomingaudio". Then, store the results of our model prediction in # "results2". (range from 0-1)
# for file in os.listdir(folder_path):

#     FILEPATH = os.path.join(folder_path, file)

#     wav = load_mp3_16k_mono(FILEPATH)
#     audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=15999, batch_size=1)
#     audio_slices = audio_slices.map(preprocess_mp3)
#     audio_slices = audio_slices.batch(64)
    
#     yhat = model.predict(audio_slices)
#     print(yhat)

#     results2[file] = yhat

#     class_preds2 = {}

# # for item in results2.items():
# #     print(item)
# # print(results2)
# for file, logits in results2.items():
#     print(prediction for prediction in logits)

# for file, logits in results2.items():
#     class_preds2[file] = [1 if prediction > 0.5 else 0 for prediction in logits]
# #class_preds2 has our classification reults (0 or 1) for our incoming audio.

# #converts tensorflow array to NumPy array, and prints out results in the format (FILE : RESULT)
# postprocessed2 = {}
# for file, scores in class_preds2.items():
#     postprocessed2[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

# for key, value in postprocessed2.items(): 
#     print(key, value)


# while(True):
#     with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=48000) as mic:
#         # Save a numpy array of audio in "data".
#         data = mic.record(numframes=48000)
    
#     yhat = model.predict()
#     print(yhat)


#//////////////////////////////////////
#create folder path
folder_path = "incomingaudio"
os.makedirs(folder_path, exist_ok=True)

i=0

while True:
    
    i+=1
    
    file_path = os.path.join(folder_path, f"TestRecording_{i}.wav")
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=48000) as mic:
        data = mic.record(numframes=48000)  # Record 1 second of audio
        sf.write(file=file_path, data=data[:, 0], samplerate=48000)  # Save as .wav file
        
    wav = load_wav_16k_mono(file_path)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=15999, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    for prediction in yhat:
        if prediction > 0.5:
            print("Molly detected")
        else:
            print("Nothing")
        