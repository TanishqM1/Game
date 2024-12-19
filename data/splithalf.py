from pydub import AudioSegment
import os

def split_wav_files(input_folder, output_folder):
    """
    Splits all 1-second WAV files in the input folder into two 0.5-second files.

    Args:
        input_folder (str): Path to the folder containing the 1-second WAV files.
        output_folder (str): Path to the folder where the split files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Load the WAV file
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_wav(file_path)

            # Verify the file is 1 second long (1000 ms)
            if len(audio) == 1000:
                # Split the audio into two 0.5-second segments
                first_half = audio[:500]
                second_half = audio[500:]

                # Save the two halves as new files
                base_name = os.path.splitext(filename)[0]
                first_half.export(os.path.join(output_folder, f"{base_name}_part1.wav"), format="wav")
                second_half.export(os.path.join(output_folder, f"{base_name}_part2.wav"), format="wav")

                print(f"Processed: {filename}")
            else:
                print(f"Skipped: {filename} (not 1 second long)")

# Example usage
input_folder = "Updated_Non_Molotov_Clips_1sec"
output_folder = "Updated_Non_Molotov_Clips_05sec"
split_wav_files(input_folder, output_folder)