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

POS = os.path.join('data', 'Molotov_Clips')
NEG = os.path.join('data', 'Not_Molotov_Clips')

pos2 = tf.data.Dataset.list_files(POS+'\*.wav')
neg2 = tf.data.Dataset.list_files(NEG+'\*.wav')

positives = tf.data.Dataset.zip((pos2, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos2)))))
negatives = tf.data.Dataset.zip((neg2, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg2)))))
data = positives.concatenate(negatives)

lengths = []
for file in os.listdir(os.path.join('data', 'Molotov_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Molotov_Clips', file))
    lengths.append(len(tensor_wave))

tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

print(tf.math.reduce_mean(lengths))
