import librosa
import argparse
import torch
import warnings

SAMPLE_RATE = 20480
N_MFCC = 40
FRAMES_PER_SEC = 40
SAMPLE_DURATION = 4
HOP_LENGTH = SAMPLE_RATE // FRAMES_PER_SEC
FRAMES_IN_DURATION = SAMPLE_DURATION * FRAMES_PER_SEC

def split_to_samples(audio_path):
    samples = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sample_rate = librosa.core.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        audio = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    total_num_frames = audio.shape[1]
    num_samples = total_num_frames - FRAMES_IN_DURATION + 1

    for i in range(num_samples):
        samples.append(torch.tensor(audio[:, i:i+FRAMES_IN_DURATION], dtype=torch.float64))

    return torch.stack(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, help='path to audio, which is used to generate video')
    args = parser.parse_args()

    samples = split_to_samples(args.audio_path)

    #TODO: add video generation via model