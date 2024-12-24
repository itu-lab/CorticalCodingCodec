from pathlib import Path
import soundfile as sf
import pandas as pd
import torch
from torch import tensor
import torchmetrics.audio as audio_metrics
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

raw_audio_list = list(Path("D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\raw").glob('*.wav'))

# decoded_list = Path(f"D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\A\\A_{bitrate}").glob('*.wav') #corticodec')

results = []

def calculate_rmse(raw_audio, decoded_audio):
    from sklearn.metrics import mean_squared_error
    # return mean_squared_error(raw_audio, decoded_audio)
    return mean_squared_error(raw_audio, decoded_audio, squared=False)


def calculate_psnr(raw_audio, decoded_audio):
    from skimage.metrics import peak_signal_noise_ratio 
    return peak_signal_noise_ratio(raw_audio, decoded_audio, data_range=raw_audio.max() - raw_audio.min())

def calculate_snr(raw_audio, decoded_audio):
    return audio_metrics.SignalNoiseRatio()(tensor(raw_audio), tensor(decoded_audio))

# pip install pesq
def calculate_pesq(raw_audio, decoded_audio, sample_rate=16000, band_mode='nb'): # 16kHz or 8kHz / nb or wb
    pesq = audio_metrics.PerceptualEvaluationSpeechQuality(sample_rate, band_mode)
    return pesq(tensor(raw_audio), tensor(decoded_audio))

def calculate_stoi(raw_audio, decoded_audio, sample_rate=16000): # 16kHz or 8kHz
    stoi = audio_metrics.ShortTimeObjectiveIntelligibility(sample_rate, extended=False)
    return stoi(tensor(raw_audio), tensor(decoded_audio))

def calculate_sdr(raw_audio, decoded_audio):
    return audio_metrics.SignalDistortionRatio()(tensor(raw_audio), tensor(decoded_audio))

def calculate_pit(raw_audio, decoded_audio):
    pit = audio_metrics.PermutationInvariantTraining(
        scale_invariant_signal_noise_ratio,
        mode="speaker-wise", eval_func="max")
    return pit(tensor(raw_audio), tensor(decoded_audio))

def calculate_si_sdr(raw_audio, decoded_audio):
    return audio_metrics.ScaleInvariantSignalDistortionRatio()(tensor(raw_audio), tensor(decoded_audio))

def calculate_si_snr(raw_audio, decoded_audio):
    return audio_metrics.ScaleInvariantSignalNoiseRatio()(tensor(raw_audio), tensor(decoded_audio))

def calculate_c_si_snr(raw_audio, decoded_audio):
    return audio_metrics.ComplexScaleInvariantSignalNoiseRatio()(tensor(raw_audio), tensor(decoded_audio))

def calculate_sa_sdr(raw_audio, decoded_audio):
    return audio_metrics.SourceAggregatedSignalDistortionRatio()(tensor(raw_audio), tensor(decoded_audio))

# pip install git+https://github.com/detly/gammatone
def calculate_srmr(raw_audio, decoded_audio, sample_rate=16000): # 16kHz or 8kHz
    srmr = audio_metrics.SpeechReverberationModulationEnergyRatio(sample_rate)
    return srmr(tensor(raw_audio), tensor(decoded_audio))

# def calculate_ssim(raw_audio, decoded_audio):
#     from skimage.metrics import structural_similarity
#     return structural_similarity(raw_audio, decoded_audio, data_range=raw_audio.max() - raw_audio.min())

def analyze_audio(raw_audio, decoded_audio):
    # Calculate the quality of the decoded audio using the raw audio as a reference
    # You can use any audio quality metrics or algorithms here
    # For example, you can use the mean squared error (MSE) or signal-to-noise ratio (SNR)
    # to compare the raw audio and decoded audio

    # Return the analysis result
    return {
        'RMSE': calculate_rmse(raw_audio, decoded_audio),
        # 'SNR': calculate_snr(raw_audio, decoded_audio),
        'PSNR': calculate_psnr(raw_audio, decoded_audio),
        # 'PESQ': calculate_pesq(raw_audio, decoded_audio),
        # 'STOI': calculate_stoi(raw_audio, decoded_audio),
        # 'SDR': calculate_sdr(raw_audio, decoded_audio),
        # 'PIT': calculate_pit(raw_audio, decoded_audio),
        'SI-SDR': calculate_si_sdr(raw_audio, decoded_audio),
        # 'SI-SNR': calculate_si_snr(raw_audio, decoded_audio),
        # 'C-SI-SNR': calculate_c_si_snr(raw_audio, decoded_audio),
        # 'SA-SDR': calculate_sa_sdr(raw_audio, decoded_audio),
        # 'SRMR': calculate_srmr(raw_audio, decoded_audio)
        # 'SSIM': calculate_ssim(raw_audio, decoded_audio)
    }

# bitrate = 10 # 
# for codec_id in ['A', 'B', 'C', 'D', 'E', 'G']:#, 'F', 'G']:
for codec_id in ['F']:#, 'F', 'G']:
    print(f"Processing files for codec {codec_id}")
    for bitrate in [10, 20, 30, 40, 60, 80, 90]:
        print(f"Processing files for bitrate {bitrate} kbps")
        decoded_path = Path(f"D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\{codec_id}\\{codec_id}_{bitrate}")
        # check if decoded_path exists
        if not decoded_path.exists():
            continue
        for raw_audio in raw_audio_list:
            raw_audio_name = raw_audio.stem
            decoded_audio = decoded_path / f"{raw_audio_name}.wav"
            
            # Load the raw audio data
            raw_audio_data, _ = sf.read(raw_audio)
            decoded_audio_data, _ = sf.read(decoded_audio)

            min_len = min(len(raw_audio_data), len(decoded_audio_data))
            raw_audio_data = raw_audio_data[:min_len]
            decoded_audio_data = decoded_audio_data[:min_len]
            if len(raw_audio_data.shape) < 2:
                raw_audio_data = raw_audio_data.reshape(-1, 1)
            if len(decoded_audio_data.shape) < 2:
                decoded_audio_data = decoded_audio_data.reshape(-1, 1)
            if decoded_audio_data.shape[1] != raw_audio_data.shape[1]:
                raw_audio_data = raw_audio_data[:, 0]
            print(f"Comparing {raw_audio_name} - {raw_audio_data.shape} and {decoded_audio_data.shape}")

            try:
                audio_results = analyze_audio(raw_audio_data, decoded_audio_data)
            except Exception as e:
                continue    
            # Perform your analysis here and store the results
            result = {
                # 'audio_name': raw_audio_name,
                'codec': f"{codec_id}_{bitrate}",
                # 'analysis_result': analyze_audio(raw_audio_data, decoded_audio_data)
                **audio_results
            }
            results.append(result)

# # Write the results to a thesis file
# with open('results.txt', 'w') as f:
#     for result in results:
#         f.write(f"Audio: {result['audio_name']}\n")
#         f.write(f"Analysis Result: {result['analysis_result']}\n")
#         f.write('\n')
# print("Comparative analysis completed. Results are saved in results.txt.")

# Create a DataFrame to store the results
df = pd.DataFrame(results)

# aggregate the results
# df = df.groupby('codec').mean()
# but I also need to see the codec column
df = df.groupby('codec').mean().reset_index()


# Save the DataFrame to a CSV file
df.to_csv('results_aggr_F.csv', index=False)

print("Comparative analysis completed. Results are saved in results.csv.")
