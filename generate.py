import librosa
import numpy as np
import argparse
import torch
import warnings
from pathlib import Path

from audio.model import AudioLSTMEncoder
from image.model.autoencoder import ConvAutoencoder
from util.video_util import make_video


N_MFCC = 40
DURATION_SEC = 4
SAMPLE_RATE = 20480
FRAMES_PER_SEC = 40
HOP_LENGTH = SAMPLE_RATE // FRAMES_PER_SEC
TOTAL_FRAMES = DURATION_SEC * FRAMES_PER_SEC


def split_to_samples(audio_path: Path) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sample_rate = librosa.core.load(str(audio_path), sr=SAMPLE_RATE, res_type='kaiser_fast')
        audio = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    frames_num = audio.shape[1]
    audio = np.pad(audio, ((0, 0), (TOTAL_FRAMES - 1, 0)), 'constant', constant_values=0)

    samples = torch.tensor([audio[:, i:i+TOTAL_FRAMES] for i in range(frames_num)], dtype=torch.float32)
    return samples.view(frames_num, 1, N_MFCC, TOTAL_FRAMES)


def main(audio_path: Path, video_path: Path):
    samples = split_to_samples(audio_path)

    audio_model = AudioLSTMEncoder(n_mfcc=40, n_hidden=256, n_out=32)
    audio_checkpoint = torch.load(Path('resources', 'models', 'lstm_encoder_cls.bin'), map_location=torch.device('cpu'))
    audio_model.load_state_dict(audio_checkpoint['state_dict'])

    image_checkpoint = torch.load(Path('resources', 'models', 'conv_autoencoder.bin'), map_location=torch.device('cpu'))
    image_model = ConvAutoencoder(hidden_size=32, num_class=10)
    image_model.load_state_dict(image_checkpoint['state_dict'])
    image_model.set_eval()

    audio_out = audio_model(samples)
    image_out = image_model(audio_out)
    make_video(image_out, video_path, fps=40, audio_file=audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, help='path to audio')
    parser.add_argument('--video_path', type=str, help='path to video')
    args = parser.parse_args()
    main(Path(args.audio_path), Path(args.video_path))
