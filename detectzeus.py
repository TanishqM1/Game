import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import soundcard as sc
import soundfile as sf

# Function to load audio as 16k mono
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample from 44100Hz to 16000Hz
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Pathways to positive and negative clips
POS = os.path.join('data', 'zeus_clips')
NEG = os.path.join('data', 'Updated_Non_Molotov_Clips_1sec')

# Filtering .wav files in each folder
pos2 = tf.data.Dataset.list_files(POS + '\*.wav')
neg2 = tf.data.Dataset.list_files(NEG + '\*.wav')

# Labeling data
positives = tf.data.Dataset.zip((pos2, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos2)))))
negatives = tf.data.Dataset.zip((neg2, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg2)))))
data = positives.concatenate(negatives)

# Finding lengths of all data files
lengths = []
for file in os.listdir(os.path.join('data', 'zeus_clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'zeus_clips', file))
    lengths.append(len(tensor_wave))

tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

print("check")
print(tf.math.reduce_mean(lengths))
print(tf.math.reduce_min(lengths))
print(tf.math.reduce_max(lengths))

# Preprocessing files to convert to spectrogram
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Sample spectrogram for positive match
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
print(spectrogram.shape)

# Preprocessing data strings
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

print(len(data))

# Splitting data into training (70%) and testing (30%)
train = data.take(64)
test = data.skip(64).take(28)

# Showing spectrogram shape needed for a positive match
samples, labels = train.as_numpy_iterator().next()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Building deep learning model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(491, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Training model (epochs can be adjusted for accuracy)
hist = model.fit(train, epochs=1, validation_data=test)

X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

print(tf.math.reduce_sum(yhat))
print(tf.math.reduce_sum(y_test))

print(yhat)
print(y_test)

# Function to load MP3 as 16k mono
def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

# Preprocess MP3 sample for spectrogram
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([16000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

from itertools import groupby
import csv
from player import user

while True:
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=16000) as mic:
        data = mic.record(numframes=16000)

    wav = tf.convert_to_tensor(data[:, 0], dtype=tf.float32)
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    myspectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    myspectrogram = tf.abs(myspectrogram)
    myspectrogram = tf.expand_dims(myspectrogram, axis=2)
    input_data = tf.expand_dims(myspectrogram, axis=0)

    my_prediction = model.predict(input_data)
    hpCheck = user.hpCheck()

    for prediction in my_prediction:
        if prediction > 0.5 and hpCheck:
            print("Zeus detected")
