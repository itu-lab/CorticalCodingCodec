from tqdm import tqdm
import numpy as np

from corticod.utils import audio
from corticod.utils import preprocessing
from corticod.algorithm import cortical_tree

WINDOW_SIZE = 8
EPOCH = 32
RANGE_INIT = 0.1 # 0.3
RANGE_LIMIT = 0.005 # 0.05

# NOTE: run from repo root to reach the paths correctly 
train_audio_path = r"data\audio\raw\fmi_0_gt.wav"
train_audio = audio.get_audio_data(train_audio_path)
train_data = preprocessing.process_audio(train_audio['audio.data'], WINDOW_SIZE).numpy()

tree = cortical_tree.CortexTree(window_size=WINDOW_SIZE, range_init=RANGE_INIT, range_limit=RANGE_LIMIT)
_, cdbk_len = tree.train(train_data, EPOCH)

codebook = tree.complete()
print(f"Codebook shape: {codebook.paths.shape}")
with open('tests/calculations/codebook.npy', 'wb') as f:
    np.save(f, codebook.paths)


test_audio_path = r"data\audio\raw\fmi_1_gt.wav"
test_audio = audio.get_audio_data(test_audio_path)
test_data = preprocessing.process_audio(test_audio['audio.data'], WINDOW_SIZE).numpy()


encoded = []
for w in tqdm(test_data, desc="Window"):
    encoded.append(codebook.encode(w))
encoded = np.asarray(encoded)

print(f"Encoded: {encoded}")
print(f"Encoded shape: {encoded.shape}")
with open('tests/calculations/encoded.npy', 'wb') as f:
    np.save(f, encoded)

# decode
decoded = codebook.decode(encoded)
print(f"Decoded: {decoded[:4]}")
print(f"Decoded shape: {len(decoded)},{len(decoded[0])}:{decoded[0]}")

reconstructed = preprocessing.process_audio_inverse(decoded)
reconstructed = np.asarray(reconstructed)
print(reconstructed)

import matplotlib.pyplot as plt
plt.plot(test_audio['audio.data'])
plt.plot(reconstructed)
plt.show()

import matplotlib.pyplot as plt
plt.plot(test_audio['audio.data'][1000:2000])
plt.plot(reconstructed[1000:2000])
plt.show()

import soundfile as sf
sf.write('./decoded_test.wav', reconstructed, test_audio['audio.sr'])