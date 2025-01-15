from corticod.utils import audio

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
import math
sample_bits = math.log2(cdbk_len)
print("Bits:", sample_bits)

encoded_bits = total_samples * sample_bits
print("Encded Bitrate:", encoded_bits/audio_duration)
encoded_bytes = encoded_bits // 8
print("Total Size:", total_bytes, "Encoded Size:", encoded_bytes, encoded_bytes/1024)
print("Compression: ", encoded_bytes/total_bytes)
print("Size ratio: ", total_bytes/encoded_bytes)

# ffmpeg -i fmi_1_gt.wav -b:a 88k fmi_1_gt.mp3