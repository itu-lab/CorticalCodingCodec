import os 
from pathlib import Path

full_raw_path = "D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\raw"

# target_bitrates = [10, 20, 40, 60, 80]
target_bitrates = [20, 40, 60, 80]

codec_map = {
    ###### "A": "corticodec",
    # "B": "mp3", # mp3
    # "C": "m4a", # aac
    "D": "oga", # vorbis
    # "E": "opus" # opus
}

CODEC_LIB = {
    'mp3': 'libmp3lame',
    'm4a': 'aac', # 'libfdk_aac' # aac
    'opus': 'libopus',
    'oga': 'libvorbis'
}

raw_path = Path(full_raw_path)
# print(list(raw_path.glob('*.wav')))

# import ffmpeg
import subprocess

def run_cmd(command, log=False):
    # subprocess.run(command, shell=True)
    if log: print('Executing:', command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    stdout, stderr = process.communicate()
    if log:
        if stdout is not None:
            output_text = stdout.decode("iso-8859-9")
            if output_text != '':
                print('OUT:')
                print(output_text)
        if stderr is not None:
            error_text = stderr.decode("iso-8859-9")
            if error_text != '':
                print('ERROR:')
                print(error_text)

for target_bitrate in target_bitrates:
    print(f"Processing files for bitrate {target_bitrate} kbps")

    for codec_id, codec in codec_map.items():
        selected_codec_lib = CODEC_LIB[codec]
        print(f"Processing files for codec {codec}")
        encoded_path = f"D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\encoded\\{codec_id}\\{codec_id}_{target_bitrate}"
        decoded_path = f"D:\\WebApps\\audio survey\\audio_survey\\api\\audio\\{codec_id}\\{codec_id}_{target_bitrate}"
        # apply ffmpeg to all the files in the raw folder
        # and save the processed files in the processed folder
        for file in raw_path.glob('*.wav'):
            new_file_name = file.stem + "." + codec 
            # print("\t",new_file_name)

            # create the processed folder if it does not exist
            if not os.path.exists(decoded_path):
                os.makedirs(decoded_path)

            # apply ffmpeg
            input_path = str(file)
            # print("\t",input_path)
            output_path = f"{encoded_path}\\{new_file_name}"
            decoded = f"{decoded_path}\\{file.name}"
            # print("\t",output_path)
            
            # ffmpeg -i input.wav -b:a 128k output.mp3
            # ffmpeg.input(input_path).output(output_path, b='128k').run()
            
            # set codec and bitrate
            # ffmpeg -i input.wav -c:a {selected_codec_lib} -b:a {target_bitrate}k output.mp3
            # ffmpeg.input(input_path).output(output_path, b=f'{target_bitrate}k', acodec=selected_codec_lib).run()
            # command = f'ffmpeg -i \"{input_path}\" -codec:a {selected_codec_lib} -b:a {target_bitrate}k \"{output_path}\"' # -ar {samplerate}
            # command = f'ffmpeg -i \"{input_path}\" -codec:a {selected_codec_lib} -b:a {target_bitrate}k -ar 100 \"{output_path}\"' # -ar {samplerate}
            
            # command to decode the file from the encoded folder
            command = f'ffmpeg -i \"{output_path}\" \"{decoded}\"'
            
            run_cmd(command, log=True)

