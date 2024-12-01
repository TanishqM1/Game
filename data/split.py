from pydub import AudioSegment
import os

# Load the .wav file
input_file = "data\grenade_sounds_2.wav"
audio = AudioSegment.from_wav(input_file)

# Length of each split in milliseconds (3 seconds = 3000 milliseconds)
split_length_ms = 1000

# Calculate number of splits, including any remaining partial segment
total_duration_ms = len(audio)
splits_count = (total_duration_ms + split_length_ms - 1) // split_length_ms  # Round up

# Create output directory if it doesn't exist
output_dir = "Grenade_Clips"
os.makedirs(output_dir, exist_ok=True)

# Split and export each segment
for i in range(splits_count):
    start_time = i * split_length_ms
    end_time = min(start_time + split_length_ms, total_duration_ms)  # Ensure we don't go past the end
    split_audio = audio[start_time:end_time]
    
    # Export each split as a new .wav file
    output_filename = f"{output_dir}/split_{i+1+61}.wav"
    split_audio.export(output_filename, format="wav")
    print(f"Exported: {output_filename}")

print("Splitting complete!")
