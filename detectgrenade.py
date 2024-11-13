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
POS = os.path.join('data', 'Grenade_Clips')
NEG = os.path.join('data', 'Not_Grenade_Clips')


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
import csv
from player import user

while True:
    
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=16000) as mic:
        data = mic.record(numframes=16000)  # Record 1 second of audio

    # Preprocess the audio directly from 'data'
    # Ensure the waveform is 1 second (16000 samples) by padding/trimming
    wav = tf.convert_to_tensor(data[:, 0], dtype=tf.float32)  # Convert to tensor if not already
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)  # Padding to 16000 samples
    wav = tf.concat([zero_padding, wav], 0)

    # Create the spectrogram
    myspectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    myspectrogram = tf.abs(myspectrogram)  # Get the magnitude of the spectrogram
    myspectrogram = tf.expand_dims(myspectrogram, axis=2)  # Add a channel dimension for model input

    # Prepare the spectrogram for model input (add batch dimension)
    input_data = tf.expand_dims(myspectrogram, axis=0)

    # Make a prediction using the model
    my_prediction = model.predict(input_data)
    hpCheck = user.hpCheck()

    # Output prediction results

    for prediction in my_prediction:
        if prediction > 0.5 and hpCheck:
            print("Molly detected")
