import torch
import torch.nn.functional as F

def haar_wavelet_decomposition(data, levels:int=None):
    """
    Perform multi-level Haar wavelet decomposition on a batch of signals.

    Args:
    data (Tensor): The input tensor of shape (batch_size, signal_length).
    levels (int): Number of decomposition levels.

    Returns:
    Tensor: The decomposed signals.
    """
    if levels is None:
        levels = int(np.log2(data.shape[1]))

    batch_size, signal_length = data.shape

    # Haar filters
    low_pass_filter = torch.tensor([0.7071, 0.7071]).view(1, 1, 2).float().to(data.device)
    high_pass_filter = torch.tensor([0.7071, -0.7071]).view(1, 1, 2).float().to(data.device)

    batch_stack = 1
    for _ in range(levels):
        # Apply the filters
        approx = F.conv1d(data.view(batch_size*batch_stack, 1, -1), low_pass_filter, stride=2)
        detail = F.conv1d(data.view(batch_size*batch_stack, 1, -1), high_pass_filter, stride=2)

        # Concatenate approximations and details
        data = torch.cat((approx, detail), dim=1)  
        batch_stack *= 2

    # reindex the data 
    data = data.view(batch_size, -1)
    
    return data

def haar_wavelet_reconstruction(data, levels:int=None):
    """
    Perform multi-level inverse Haar wavelet reconstruction on decomposed signals.

    Args:
    data (Tensor): The input tensor containing the decomposed signals.
    levels (int): Number of decomposition levels.

    Returns:
    Tensor: The reconstructed original signals.
    """
    if levels is None:
        levels = int(np.log2(data.shape[1]))

    batch_size, signal_length = data.shape

    # Haar filters
    low_pass_filter = torch.tensor([0.7071, 0.7071]).view(1, 1, 2).float().to(data.device)
    high_pass_filter = torch.tensor([0.7071, -0.7071]).view(1, 1, 2).float().to(data.device)

    current_length = signal_length // (2 ** levels)  # Initial length of the approximation and detail components

    for _ in range(levels):
        data = data.view(-1, 2, current_length)
        d_approx, d_detail = data.unbind(dim=1)
        d_approx = d_approx.view(-1, 1, current_length)
        d_detail = d_detail.view(-1, 1, current_length)
        data = F.conv_transpose1d(d_approx, low_pass_filter, stride=2) + F.conv_transpose1d(d_detail, high_pass_filter, stride=2)
        current_length *= 2

    data = data.view(-1, 8)

    return data


import numpy as np
import constriction

def calculate_probabilities(data, max_value):
    # Count the frequency of each value in the array
    counts = np.bincount(data, minlength=max_value + 1)
    # Normalize to get probabilities
    probabilities = counts / np.sum(counts)
    return probabilities

def entropy_encode(data, probabilities, type='range'):
    model = constriction.stream.model.Categorical(probabilities)
    if type=='ans':
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data, model)
    elif type=='range':
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(data, model)
    compressed = encoder.get_compressed()
    return compressed

def entropy_decode(compressed, length, probabilities, type='range'):
    model = constriction.stream.model.Categorical(probabilities)
    if type=='ans':
        decoder = constriction.stream.stack.AnsCoder(compressed)
    elif type=='range':
        decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded = np.empty(length, dtype=np.uint32) # , dtype=np.int16)
    for i in range(length):
        decoded[i] = decoder.decode(model)
    return decoded


import numpy as np
import torch
def process_audio(data, ws):
    data_padded = np.pad(data, (0, ws - len(data) % ws))
    data_windows = data_padded.reshape(-1, ws)
    data_windows_tensor = torch.tensor(data_windows).float()
    decomposed_data = haar_wavelet_decomposition(data_windows_tensor)
    return decomposed_data

def process_audio_inverse(data):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data).float()
    data_reconstruction = haar_wavelet_reconstruction(data)
    data_reconstruction = data_reconstruction.reshape(-1) 
    return data_reconstruction

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
from scipy.signal import find_peaks
def score_reconstruction(original_data, reconstructed_data):
    if len(original_data) != len(reconstructed_data):
        reconstructed_data = reconstructed_data[:len(original_data)]
    # return np.linalg.norm(original_data - reconstructed_data)
    rmse_score = mse(original_data*2**8, reconstructed_data.numpy()*2**8, squared=False)
    psnr_score = psnr(original_data, reconstructed_data.numpy(), data_range=original_data.max() - original_data.min())
    ssim_score = ssim(original_data, reconstructed_data.numpy(), data_range=original_data.max() - original_data.min())
    # peaks, _ = find_peaks(original_data)
    # snr_score = np.mean(original_data[peaks] / (original_data[peaks] - reconstructed_data.numpy()[peaks]))
    return {"RMSE": rmse_score, "PSNR": psnr_score, "SSIM": ssim_score}


# quantize the cb.paths to get the quantized codebook
def quantize_codebook(codebook, codebook_quantizer):
    original_shape = codebook.paths.shape
    all_codebook_values = codebook.paths.reshape(-1,1)
    codebook_quantized = codebook_quantizer.quantize(all_codebook_values).reshape(-1)
    return codebook_quantized.reshape(original_shape)
    # quantized_cb = Codebook(quantize_codebook(cb, cdbk_cb)[:, :])