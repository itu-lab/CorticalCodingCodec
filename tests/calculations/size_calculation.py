from corticod.utils import audio
import math
from collections import Counter

def calculate_frequencies(integers):
    """
    Calculate the frequencies (probabilities) of each integer in the list.
    
    Args:
        integers (list): List of integers.

    Returns:
        dict: A dictionary where keys are integers and values are their frequencies.
    """
    total_count = len(integers)
    frequency_dict = {key: value / total_count for key, value in Counter(integers).items()}
    return frequency_dict

def calculate_entropy_bits(frequency_dict):
    """
    Estimate the total bits required to encode the integers using entropy coding.
    
    Args:
        frequency_dict (dict): A dictionary of integer probabilities.
    
    Returns:
        float: Total bits required for encoding.
    """
    total_bits = -sum(prob * math.log2(prob) for prob in frequency_dict.values() if prob > 0)
    return total_bits

WINDOW_SIZE = 8
test_audio_path = r"data\audio\raw\fmi_1_gt.wav"
test_audio = audio.get_audio_data(test_audio_path) 

audio_samples = len(test_audio['audio.data'])
total_bytes = audio_samples * 4 # len(audio):288000*int:4 = ~1,155,072 bytes
audio_duration = audio_samples / test_audio['audio.sr']
print("Duration:", audio_duration)

total_samples = audio_samples // WINDOW_SIZE
print("Samples:", total_samples)

cdbk_len = 27182
sample_bits = math.log2(cdbk_len)
print("Bits:", sample_bits)

encoded_bits = total_samples * sample_bits
print("Encded Bitrate:", (encoded_bits/audio_duration)/1000, "kbps")
encoded_bytes = encoded_bits // 8
print("Total Size:", total_bytes, "Encoded Size:", encoded_bytes, encoded_bytes/1024)
print("Compression: ", encoded_bytes/total_bytes)
print("Size ratio: ", total_bytes/encoded_bytes)

# ffmpeg -i fmi_1_gt.wav -b:a 88k fmi_1_gt.mp3

print('--- ENTROPY APPLIED ---')

import numpy as np
with open('tests/calculations/encoded.npy', 'rb') as f:
    encoded = np.load(f)

print(encoded)
encoded_freq = calculate_frequencies(encoded)
entropy_bits = calculate_entropy_bits(encoded_freq)
print("Total bits required:", entropy_bits)

encoded_bits = total_samples * entropy_bits
print("Encded Bitrate:", (encoded_bits/audio_duration)/1000, "kbps")
encoded_bytes = encoded_bits // 8
print("Total Size:", total_bytes, "Encoded Size:", encoded_bytes, encoded_bytes/1024)
print("Compression: ", encoded_bytes/total_bytes)
print("Size ratio: ", total_bytes/encoded_bytes)
