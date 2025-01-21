### Audio Compression Utilities

The library includes utility functions for analyzing, comparing, and processing audio data. These are particularly useful for testing and evaluating audio compression algorithms.

#### Key Features:
1. **Audio Information Extraction**:
   - `get_audio_info(audio_path)`: Extracts metadata (e.g., sample rate, duration, bit depth) and raw audio data from a given file.

2. **Compression Metrics**:
   - `get_audio_metrics(audio1, audio2)`: Calculates metrics like RMSE, PSNR, NMI, and SSIM between two audio signals to evaluate quality after compression.

3. **Compression Evaluation**:
   - `get_compression_rate(raw_audio_data, encoded_data_size)`: Estimates compression ratio based on raw and encoded audio sizes.
   - `get_results(raw_audio_path, enc_audio_path, dec_audio_path)`: Summarizes compression results, including compression ratio and audio quality metrics.

4. **Supported Formats**:
   - WAV files natively, with FFmpeg integration for other audio formats.

#### Example Usage:
```python
from utils.audio_compression import get_results

raw_audio_path = "input.wav"
encoded_audio_path = "compressed.opus"
decoded_audio_path = "decompressed.wav"

# Evaluate compression results
results = get_results(raw_audio_path, encoded_audio_path, decoded_audio_path)
print("Compression Ratio:", results['compression_ratio'])
print("Audio Quality Metrics:", results['compression_result'])
```


### Haar Wavelet Utilities

This module provides utilities for Haar wavelet decomposition and reconstruction, as well as tools for entropy encoding and decoding. These functions are designed for signal processing and compression tasks, particularly for applications involving audio or time-series data.

#### Key Features

1. **Wavelet Decomposition and Reconstruction**:
   - `haar_wavelet_decomposition(data, levels)`: Performs multi-level Haar wavelet decomposition on signals.
   - `haar_wavelet_reconstruction(data, levels)`: Reconstructs signals from Haar wavelet coefficients.

2. **Entropy Encoding and Decoding**:
   - `entropy_encode(data, probabilities, type='range')`: Encodes data using range or ANS entropy coding.
   - `entropy_decode(compressed, length, probabilities, type='range')`: Decodes entropy-encoded data back to the original.

3. **Signal Processing**:
   - `process_audio(data, ws)`: Decomposes padded audio data into wavelet coefficients.
   - `process_audio_inverse(data)`: Reconstructs audio data from wavelet coefficients.

4. **Performance Metrics**:
   - `score_reconstruction(original_data, reconstructed_data)`: Computes RMSE, PSNR, and SSIM between original and reconstructed signals.

#### Example Usage

```python
import numpy as np
from haar_wavelet_utils import haar_wavelet_decomposition, haar_wavelet_reconstruction, score_reconstruction

# Example signal
data = np.random.rand(1024)  # 1D signal of length 1024

# Decomposition
levels = 3
decomposed = haar_wavelet_decomposition(data, levels=levels)

# Reconstruction
reconstructed = haar_wavelet_reconstruction(decomposed, levels=levels)

# Evaluate Reconstruction Quality
metrics = score_reconstruction(data, reconstructed)
print("Reconstruction Metrics:", metrics)
```

#### Applications
- Audio compression.
- Feature extraction in signal processing.
- Data compression using entropy coding.

#### Dependencies
- Python >= 3.8
- NumPy
- Torch
- scikit-image
- constriction (for entropy coding)
