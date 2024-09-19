import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio


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
POS = os.path.join('data', 'Molotov_Clips')
NEG = os.path.join('data', 'Not_Molotov_Clips')


#filtering .wav files in each folder
pos2 = tf.data.Dataset.list_files(POS+'\*.wav')
neg2 = tf.data.Dataset.list_files(NEG+'\*.wav')

#idk
positives = tf.data.Dataset.zip((pos2, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos2)))))
negatives = tf.data.Dataset.zip((neg2, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg2)))))
data = positives.concatenate(negatives)

#finding lengths of all data files
lengths = []
for file in os.listdir(os.path.join('data', 'Molotov_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Molotov_Clips', file))
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
    wav = wav[:6500]
    zero_padding = tf.zeros([6500] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

#tweak frame_length, frame_step & wav[6500] based on testing

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

#preprocessing data strings
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# print(len(data))
#train machine using 70% of clips and test on the renaming 30%.

train = data.take(23)
test = data.skip(23).take(10)

#show spectogram shape needed, for a positive match.
samples, labels = train.as_numpy_iterator().next()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#build deep learning model

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(194, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# print(model.summary())

#train model

# epochs can be tweaked. Larger = more accurate
hist = model.fit(train, epochs=10, validation_data=test)

X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)

yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

print(tf.math.reduce_sum(yhat))
print(tf.math.reduce_sum(y_test))

print(yhat)
print(y_test)

import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000  # 16 kHz
CHUNK_DURATION = 1  # Process audio in 1-second chunks
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # Number of samples in a chunk

# Function to capture a chunk of system audio
def record_system_audio_chunk(sample_rate, chunk_size):
    print("Capturing audio chunk...")
    audio_chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='float32',
                         device=None,  # Use default system output (WASAPI)
                         blocking=True)
    audio_chunk = tf.convert_to_tensor(audio_chunk[:, 0], dtype=tf.float32)  # Remove channel dimension
    return audio_chunk

# Preprocess audio chunk into spectrogram
def preprocess_audio_chunk(audio_chunk):
    audio_chunk = tfio.audio.resample(audio_chunk, rate_in=SAMPLE_RATE, rate_out=16000)
    audio_chunk = audio_chunk[:6500]
    zero_padding = tf.zeros([6500] - tf.shape(audio_chunk), dtype=tf.float32)
    audio_chunk = tf.concat([zero_padding, audio_chunk], 0)
    spectrogram = tf.signal.stft(audio_chunk, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# Real-time audio detection loop
def detect_real_time_audio():
    while True:
        audio_chunk = record_system_audio_chunk(SAMPLE_RATE, CHUNK_SIZE)
        spectrogram = preprocess_audio_chunk(audio_chunk)

        # Make prediction
        prediction = model.predict(tf.expand_dims(spectrogram, axis=0))
        prediction_label = 1 if prediction > 0.5 else 0

        if prediction_label == 1:
            print("Match detected!")
        else:
            print("No match detected.")
        
        # Add a break condition if you want to stop
        # For example, press a key or run for a certain number of seconds
        # To stop the loop, you could add:
        # if stop_condition: 
        #    break

# Start real-time audio detection
detect_real_time_audio()
