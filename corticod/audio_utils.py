from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess
import os

def get_audio_info(audio_path):
    audio_data, samplerate = sf.read(str(audio_path))#, dtype='int16')
    
    # FORCE MONO - TODO: Add support for stereo
    # if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
    if len(audio_data.shape) > 1: audio_data = audio_data.T[0]
    # FORCE 16 BIT - TODO: Add support for 24 bit
    # if audio_data.dtype != np.int16:
    #     audio_data = (audio_data * 32767).astype(np.int16)

    channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
    bytes_per_sample = audio_data.dtype.itemsize
    return {
        'audio.data':audio_data, 
        'audio.sr':samplerate, 
        'audio.channels':channels, 
        'audio.bytes': bytes_per_sample,
        'audio.path': audio_path,
        'audio.duration': len(audio_data) / samplerate,
        'audio.total_bits': os.path.getsize(audio_path) * 8 # len(audio_data) * bytes_per_sample * 8
    }

def get_audio_data(audio_path):
    wav_file = "remove_test_output.wav"
    wav_file = os.path.join(os.getcwd(), wav_file)
    if os.path.exists(wav_file):
        os.remove(wav_file)
    command = f'ffmpeg -i "{audio_path}" "{wav_file}"'
    subprocess.run(command, shell=True)
    # samplerate = sf.info(wav_file).samplerate
    return get_audio_info(wav_file)

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim
# from scipy.signal import butter, filtfilt
metrics = ['rmse', 'psnr', 'nmi', 'ssim'] # ,'mse'
def get_audio_metrics(audio1, audio2):
    """
    get_audio_metrics(raw, raw): {'rmse': 0.0, 'psnr': inf, 'nmi': 2.0 (sometimes nan), 'ssim': 1.000}
    get_audio_metrics(ones, zeros): {'rmse': 1.0, 'psnr': 0.0, 'nmi': nan, 'ssim': 0.0005}
    get_audio_metrics(ones, random): {'rmse': 0.6, 'psnr': 5.0, 'nmi': 1.0, 'ssim': 0.0500}
    get_audio_metrics(zeros, random): {'rmse': 0.6, 'psnr': 5.0, 'nmi': 1.0, 'ssim': 0.0001}
    get_audio_metrics(random, random): {'rmse': 0.4, 'psnr': 7.8, 'nmi': 1.6, 'ssim': 0.0200}
    """
    audio1 = audio1[:min(len(audio1), len(audio2))] # assuming the mismatch is at the end
    audio2 = audio2[:min(len(audio1), len(audio2))]
    rmse_score = np.sqrt(mse(audio1, audio2))
    # rmse_score = nrmse(audio1, audio2)
    psnr_score = psnr(audio1, audio2, data_range=1)
    nmi_score = nmi(audio1, audio2)
    return {
        # 'mse': mse(audio1, audio2),
        'rmse': rmse_score,
        'psnr': psnr_score,
        'nmi' : nmi_score,
        'ssim': ssim(audio1, audio2)
    }

def get_compresion_result(original_audio_data, decoded_audio_data):
    if isinstance(original_audio_data, str) or isinstance(original_audio_data, Path):
        original_audio_data = get_audio_data(original_audio_data)
    if isinstance(decoded_audio_data, str) or isinstance(decoded_audio_data, Path):
        decoded_audio_data = get_audio_data(decoded_audio_data)
    # TODO: sometimes samplerates are different --> opus
    # assert original_audio_data['audio.sr'] == decoded_audio_data['audio.sr'], f"Sample rates should be same: {original_audio_data['audio.sr']} != {decoded_audio_data['audio.sr']}"
    assert original_audio_data['audio.channels'] == decoded_audio_data['audio.channels'], f"Channels should be same: {original_audio_data['audio.channels']} != {decoded_audio_data['audio.channels']}"
    original_audio = original_audio_data['audio.data']
    decoded_audio = decoded_audio_data['audio.data']
    if original_audio_data['audio.channels'] > 1:
        return { # Assuming index 0 is left channel, and index 1 is right channel
            'left': get_audio_metrics(original_audio[0], decoded_audio[0]),
            'right': get_audio_metrics(original_audio[1], decoded_audio[1])
        }
    else:
        # if len(original_audio) < len(decoded_audio):
        #     original_audio = np.pad(original_audio, (0, len(decoded_audio) - len(original_audio)))
        # elif len(original_audio) > len(decoded_audio):
        #     decoded_audio = np.pad(decoded_audio, (0, len(original_audio) - len(decoded_audio)))
        return get_audio_metrics(original_audio, decoded_audio)

def estimate_raw_audio_bytes(audio_data):
    return (len(audio_data) * audio_data.dtype.itemsize) + 45 # added header

def get_compression_rate(raw_audio_data, encoded_data_size):
    return estimate_raw_audio_bytes(raw_audio_data) / encoded_data_size

def get_results(raw_audio_path, enc_audio_path, dec_audio_path):
    entropy_coded_raw_bytes = os.path.getsize(raw_audio_path)
    # get size of the encoded file
    entropy_coded_enc_bytes = os.path.getsize(enc_audio_path)
    compression_ratio = entropy_coded_raw_bytes / entropy_coded_enc_bytes
    return {
        'compression_ratio': compression_ratio,
        'compression_result': get_compresion_result(raw_audio_path, dec_audio_path)
    }