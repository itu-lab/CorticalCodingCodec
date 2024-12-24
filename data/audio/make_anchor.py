import os
import shutil
from pydub import AudioSegment

# Define the input and output directories
input_dir = "D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\raw"

low_anchor_dir = "D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\low_anchor"
mid_anchor_dir = "D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\mid_anchor"

# Create the output directories if they don't exist
os.makedirs(low_anchor_dir, exist_ok=True)
os.makedirs(mid_anchor_dir, exist_ok=True)

# Get all .wav files from the input directory
wav_files = [file for file in os.listdir(input_dir) if file.endswith('.wav')]

# Process each .wav file
for wav_file in wav_files:
    # Load the audio file
    audio = AudioSegment.from_wav(os.path.join(input_dir, wav_file))
    
    # Generate the low anchor audio by cutting off frequencies
    low_anchor_audio = audio.low_pass_filter(3500)
    # low_anchor_audio = audio.low_pass_filter(5000)
    
    # Generate the mid anchor audio by cutting off frequencies
    # mid_anchor_audio = audio.low_pass_filter(10000).high_pass_filter(5000)
    mid_anchor_audio = audio.low_pass_filter(7000).high_pass_filter(3500)
    
    # Anonymize the anchor audio by mixing with random audio
    # random_audio_files = random.sample(wav_files, 2)  # Select 2 random audio files
    # random_audio = sum([AudioSegment.from_wav(os.path.join(input_dir, file)) for file in random_audio_files])
    # low_anchor_audio = low_anchor_audio.overlay(random_audio)
    # mid_anchor_audio = mid_anchor_audio.overlay(random_audio)

    # Save the low anchor audio to the output directory
    low_anchor_file = os.path.join(low_anchor_dir, wav_file)
    low_anchor_audio.export(low_anchor_file, format='wav')
    
    # Save the mid anchor audio to the output directory
    mid_anchor_file = os.path.join(mid_anchor_dir, wav_file)
    mid_anchor_audio.export(mid_anchor_file, format='wav')
    

# Print a message indicating the process is complete
print('Anchor audio files generated successfully.')