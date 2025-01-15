"""
1. Uygulamanın ortalama CPU - Memory tüketimini ölçen program/tool/library var mı?
2. 1 birim - 10 birim - 100 birim için test et (mesela birim=sn)
    Encode için , Decode için: Execution time vs. cpu usage / memory usage

d = derinlik (window size)
N (Şarkı uzunluğu) 
N/d * d --> N işlem yapılıyor
M = bir düğümün maksimum çocuk seviyesi, codebook'a bağlı
O(N log M) 

3. CPU to watt (electricity usage) 

4.

1 dk music
store:
OPUS     : X sec encoding + CPU usage graph + memory usage graph
Corticod : Y sec encoding + CPU usage graph + memory usage graph 

5 farklı zaman süreci

"""

# ffmpeg
import psutil
import subprocess
import time
import json
import requests
import os

# Configurations
input_file = "input.wav"  # Path to your input file
codecs = {
    "mp3": "-c:a libmp3lame",
    "aac": "-c:a aac",
    "opus": "-c:a libopus",
    "ogg": "-c:a libvorbis"
}
output_folder = "codec_logs"
monitor_interval = 1  # Interval for monitoring in seconds
power_monitor_url = "http://localhost:8085/data.json"  # OpenHardwareMonitor JSON API

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

def monitor_process(proc, task_name):
    """Monitor CPU, memory, and power usage for a process."""
    log_file = os.path.join(output_folder, f"{task_name}_profile.log")
    with open(log_file, "w") as log:
        log.write("Time(s),CPU(%),Memory(MB),Power(W)\n")
        start_time = time.time()

        while proc.poll() is None:  # While the process is running
            elapsed_time = time.time() - start_time
            try:
                # Get process stats
                process = psutil.Process(proc.pid)
                cpu_usage = process.cpu_percent(interval=0.1)
                memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

                # Get power usage (requires OpenHardwareMonitor running)
                power_usage = get_power_usage()
                log.write(f"{elapsed_time:.5f},{cpu_usage:.5f},{memory_usage:.5f},{power_usage:.5f}\n")
                print(f"{task_name} | Time: {elapsed_time:.5f}s, CPU: {cpu_usage:.5f}%, Mem: {memory_usage:.5f} MB, Power: {power_usage:.5f} W")
                time.sleep(monitor_interval)

            except psutil.NoSuchProcess:
                print(f"Process {proc.pid} ended unexpectedly.")
                break
            except Exception as e:
                print(f"Error monitoring process {proc.pid}: {e}")
                break

        log.write("Profiling complete.\n")
    print(f"Profiling for {task_name} complete. Log saved to {log_file}")

def get_power_usage():
    """Get total system power usage from OpenHardwareMonitor."""
    try:
        response = requests.get(power_monitor_url)
        data = response.json()

        # Extract power consumption (customize based on your OpenHardwareMonitor setup)
        power_usage = 0.0
        for sensor in data["Children"][0]["Children"]:
            for item in sensor["Sensors"]:
                if "Power" in item["Name"]:
                    power_usage += item["Value"]
        return power_usage
    except Exception as e:
        print(f"Error fetching power usage: {e}")
        return 0.0

def encode_audio(codec_name, codec_args):
    """Run FFmpeg encoding and profile the process."""
    output_file = os.path.join(output_folder, f"encoded_{codec_name}.{codec_name}")
    command = f"ffmpeg -i {input_file} {codec_args} {output_file} -y"
    print(f"Encoding with {codec_name}: {command}")
    proc = subprocess.Popen(command, shell=True)
    monitor_process(proc, f"{codec_name}_encoding")
    return output_file

def decode_audio(codec_name, encoded_file):
    """Run FFmpeg decoding and profile the process."""
    decoded_output = os.path.join(output_folder, f"decoded_{codec_name}.wav")
    command = f"ffmpeg -i {encoded_file} {decoded_output} -y"
    print(f"Decoding {codec_name}: {command}")
    proc = subprocess.Popen(command, shell=True)
    monitor_process(proc, f"{codec_name}_decoding")
    
# Run profiling for each codec

# Run profiling for each codec
for codec_name, codec_args in codecs.items():
    # Encoding
    encoded_file = encode_audio(codec_name, codec_args)

    # Decoding
    decode_audio(codec_name, encoded_file)

print("All encoding and decoding tasks are complete!")
