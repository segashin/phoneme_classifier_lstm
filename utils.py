import os

# returns list of wav_file_path and phn_file_path tuples [("<wav_file_path>", "<phn_file_path>"), ...]
def create_wav_phn_file_path_pairs(wav_dir, phn_dir):
    ret_list = []
    wavs = [wav for wav in os.listdir(wav_dir) if wav.lower().endswith(".wav")]

    for wav_file in wavs:
        phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
        
        wav_file_path = os.path.join(wav_dir, wav_file)
        phn_file_path = os.path.join(phn_dir, phn_file)

        ret_list.append((wav_file_path, phn_file_path))

    return ret_list