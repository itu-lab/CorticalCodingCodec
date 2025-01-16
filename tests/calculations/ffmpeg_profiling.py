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
import datetime

# Configurations
# test_files = ["input_20min.wav"] # Path to the input files
test_files = os.listdir('test_data')

codecs = {
    "mp3": "-c:a libmp3lame",
    "aac": "-c:a aac",
    "opus": "-c:a libopus",
    "ogg": "-c:a libvorbis"
}
output_folder = "codec_logs"
monitor_interval = 0.01  # Interval for monitoring in seconds
power_monitor_url = "http://localhost:8085/data.json"  # OpenHardwareMonitor JSON API

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

def monitor_process(command, task_name, filename):
    """Monitor CPU, memory, and power usage for a process."""
    log_file = os.path.join(output_folder, f"{task_name}_profile.log")
    file_exists = os.path.exists(log_file)
    
    with open(log_file, "a") as log:
        # Write header only if the file is new
        if not file_exists:
            log.write("Timestamp(Date),Time(s),CPU(%),Memory(MB),Power(W)\n")
        log.write(f"Profiling started for {filename}.\n")
        start_time = time.time()
        
        total_cpu = get_overall_cpu_usage()
        proc = subprocess.Popen(command, shell=True)
        print(f"Starting monitoring for {task_name} (PID: {proc.pid})")

        while proc.poll() is None:  # While the process is running
            elapsed_time = time.time() - start_time
            try:
                if proc.poll() is not None:  # Check if the process has exited
                    print(f"Process {proc.pid} completed normally.")
                    break

                # Collect process metrics
                process = psutil.Process(proc.pid)
                # cpu_usage = process.cpu_percent(interval=0.1)
                # cpu_usage = get_cpu_usage_via_wmic(proc.pid)
                cpu_usage = get_overall_cpu_usage() - total_cpu
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                power_usage = get_power_usage()
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"{current_time},{elapsed_time:.5f},{cpu_usage:.5f},{memory_usage:.5f},{power_usage:.5f}\n")
                print(f"{task_name} | Time: {elapsed_time:.5f}s, CPU: {cpu_usage:.5f}%, Mem: {memory_usage:.5f} MB, Power: {power_usage:.5f} W")
                # time.sleep(monitor_interval)
            except psutil.NoSuchProcess:
                print(f"Process {proc.pid} ended unexpectedly.")
                break
            except Exception as e:
                print(f"Error monitoring process {proc.pid}: {e}")
                break

        log.write(f"Profiling complete for {filename}.\n")
    print(f"Profiling for {task_name} complete. Log saved to {log_file}")


def get_cpu_usage_via_typeperf(pid):
    try:
        command = f'typeperf \"\\Process({pid})\\% Processor Time\" -sc 1'
        result = subprocess.check_output(command, shell=True).decode("utf-8")
        usage_line = result.strip().splitlines()[-1]
        usage = usage_line.split(",")[-1].strip().replace("\"", "")
        return float(usage)
    except Exception as e:
        print(f"Error fetching CPU usage: {e}")
        return 0.0

def get_cpu_usage_via_wmic(pid):
    try:
        command = f"wmic path Win32_PerfFormattedData_PerfProc_Process where IDProcess={pid} get PercentProcessorTime"
        result = subprocess.check_output(command, shell=True).decode("utf-8")
        usage = result.strip().splitlines()[-1].strip()
        return float(usage)
    except Exception as e:
        print(f"Error fetching CPU usage: {e}")
        return 0.0


def get_overall_cpu_usage():
    # Get overall CPU usage as a percentage
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"Overall CPU Usage: {cpu_usage}%")
    return cpu_usage


def get_power_usage():
    """Get total system power usage from OpenHardwareMonitor."""
    try:
        response = requests.get(power_monitor_url)
        data = response.json()

        # Extract power consumption (customize based on your OpenHardwareMonitor setup)
        # power_usage = 0.0
        # for sensor in data["Children"][0]["Children"]:
        #     for item in sensor["Sensors"]:
        #         if "Power" in item["Name"]:
        #             power_usage += item["Value"]

        return parse_power_usage(data)
    except Exception as e:
        print(f"Error fetching power usage: {e}")
        return 0.0

def parse_power_usage(json_data):
    """Extract power usage data from the JSON structure."""
    power_data = []

    def traverse(node):
        """Recursive function to traverse JSON nodes."""
        if "Text" in node and "Value" in node and "CPU Core" in node["Text"]: # "Power" is in GPU
            try:
                power_value = float(node["Value"].replace(" W", ""))
                power_data.append(power_value)
            except ValueError:
                pass  # Skip if the value is not a valid number

        for child in node.get("Children", []):
            traverse(child)

    # Start traversal from the root node
    traverse(json_data)

    # Return the total power or a message if no power data is found
    return sum(power_data) if power_data else 0.0


def encode_audio(codec_name, codec_args, file_name):
    """Run FFmpeg encoding and profile the process."""
    output_file = os.path.join(output_folder, f"encoded_{file_name}_{codec_name}.{codec_name}")
    command = f"ffmpeg -i test_data\{file_name} -b:a 75k {codec_args} {output_file} -y" # -loglevel debug"
    print(f"Encoding with {codec_name}: {command}")
    monitor_process(command, f"{codec_name}_encoding", f"{file_name}-20min")
    return output_file

def decode_audio(codec_name, encoded_file, file_name):
    """Run FFmpeg decoding and profile the process."""
    decoded_output = os.path.join(output_folder, f"decoded_{file_name}_{codec_name}.wav")
    command = f"ffmpeg -i {encoded_file} -b:a 75k {decoded_output} -y" # -loglevel debug"
    print(f"Decoding {codec_name}: {command}")
    monitor_process(command, f"{codec_name}_decoding", f"{file_name}-20min")

def bic_encode_audio(codec_args, file_name):
    """Run .BIC encoding and profile the process."""
    output_file = os.path.join(output_folder, f"encoded_{file_name}_corticod.bic")
    command = f"encoder.exe -i test_data\{file_name} -o {output_file} {codec_args}"
    print(f"Encoding with corticod: {command}")
    monitor_process(command, f"corticod_encoding", f"{file_name}-20min")
    return output_file

def bic_decode_audio(codec_args, file_name):
    """Run .BIC decoding and profile the process."""
    decoded_output = os.path.join(output_folder, f"decoded_{file_name}_corticod.wav")
    command = f"decoder.exe -i {file_name} -o {decoded_output} {codec_args}"
    print(f"Decoding with corticod: {command}")
    monitor_process(command, f"corticod_decoding", f"{file_name}-20min")


for file in test_files:
    for codec_name, codec_args in codecs.items():
        # Encoding
        encoded_file = encode_audio(codec_name, codec_args, file)
        # Decoding
        decode_audio(codec_name, encoded_file, file)

print("All encoding and decoding tasks are complete!")
