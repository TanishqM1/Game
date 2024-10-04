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
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
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

train = data.take(8)
test = data.skip(8).take(3)

#show spectogram shape needed, for a positive match.
samples, labels = train.as_numpy_iterator().next()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#build deep learning model

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# print(model.summary())

#train model

# epochs can be tweaked. Larger = more accurate
hist = model.fit(train, epochs=4, validation_data=test)

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
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

from itertools import groupby

def preprocess_numpy_audio(numpy_array, sample_rate, target_sample_rate=16000):
    # Step 1: Convert NumPy array to a TensorFlow tensor
    tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
    
    # Step 2: Combine stereo channels (assuming the NumPy array has shape [num_samples, num_channels])
    if len(tensor.shape) > 1 and tensor.shape[1] == 2:
        tensor = tf.math.reduce_sum(tensor, axis=1) / 2  # Combine stereo to mono
    
    # Step 3: Cast sample rate (if it's not already an integer)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # Step 4: Resample to 16 kHz (or the target sample rate)
    resampled_tensor = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=target_sample_rate)
    
    return resampled_tensor


while (True):
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=48000) as mic:
   #saves a numpy array of audio in "data".
        incomingdata = mic.record(numframes=48000*3)
        
        processed_audio = preprocess_numpy_audio(incomingdata, 48000)
        audio_slices = tf.keras.utils.timeseries_dataset_from_array(processed_audio, processed_audio, sequence_length=48000, sequence_stride=47999, batch_size=1)
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(64)
        
        yhat = model.predict(audio_slices)
        
        for logits in yhat:
            if max(logits)>0.99:
                print("Detected Molly")
            else:
                print ("No MOlly Detected")

                ##OR

            #     for file, logits in check.items():
            # if (prediction > 0.99 for prediction in logits):
            #     print("Molly Detected")
            # else:
            #     print("No Molly Detected")


# results = {}
# for file in os.listdir(os.path.join('data', 'test_clips')):
#     FILEPATH = os.path.join('data','test_clips', file)
    
#     wav = load_mp3_16k_mono(FILEPATH)
#     print(audio_slices)
    
#     yhat = model.predict(audio_slices)
    
#     results[file] = yhat

# class_preds = {}
# for file, logits in results.items():
#     class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]


# postprocessed = {}
# for file, scores in class_preds.items():
#     postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

# import csv

# with open('results.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['recording', 'molly_calls'])
#     for key, value in postprocessed.items():
#         writer.writerow([key, value])