import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import librosa

import sys
import shutil
import warnings
from zipfile import ZipFile
from pathlib import Path
from typing import Optional, Tuple
from util import download_file


class DEAMDataset(Dataset):
    DATASET_PATH = Path('resources', 'datasets', 'DEAM')
    DATASET_URL = 'http://cvml.unige.ch/databases/DEAM/'
    AUDIO_NAME = 'DEAM_audio.zip'
    ANNOTATIONS_NAME = 'DEAM_Annotations.zip'

    def __init__(self,
                 sample_rate: int = 20480,
                 n_mfcc: int = 40,
                 frames_per_sec: int = 40,
                 sample_duration: int = 4):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frames_per_sec = frames_per_sec
        self.sample_duration = sample_duration
        self.hop_length = sample_rate // self.frames_per_sec

        self.random_state: Optional[np.random.RandomState] = None
        self.samples: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None

    @staticmethod
    def _download_audio():
        audio_path = DEAMDataset.DATASET_PATH / DEAMDataset.AUDIO_NAME
        if not audio_path.exists():
            audio_url = DEAMDataset.DATASET_URL + DEAMDataset.AUDIO_NAME
            download_file(audio_url, audio_path)

        dir_to_save = DEAMDataset.DATASET_PATH / 'audio'
        if not dir_to_save.exists():
            dir_to_save.mkdir(parents=True)

        with ZipFile(audio_path, 'r') as audio_zip:
            for file_to_extract in audio_zip.namelist():
                if not file_to_extract.endswith('.mp3'):
                    continue
                file_to_save = dir_to_save / Path(file_to_extract).name
                if file_to_save.exists():
                    continue
                with audio_zip.open(file_to_extract) as src, file_to_save.open('wb') as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _download_annotations():
        annotations_path = DEAMDataset.DATASET_PATH / DEAMDataset.ANNOTATIONS_NAME
        if not annotations_path.exists():
            annotations_url = DEAMDataset.DATASET_URL + DEAMDataset.ANNOTATIONS_NAME
            download_file(annotations_url, annotations_path)

        dir_to_save = DEAMDataset.DATASET_PATH / 'annotations'
        if not dir_to_save.exists():
            dir_to_save.mkdir(parents=True)

        arousal_dynamic = Path('annotations', 'annotations averaged per song', 'dynamic (per second annotations)', 'arousal.csv')
        valence_dynamic = Path('annotations', 'annotations averaged per song', 'dynamic (per second annotations)', 'valence.csv')

        with ZipFile(annotations_path, 'r') as annotations_zip:
            for file_to_extract in [arousal_dynamic, valence_dynamic]:
                file_to_save = dir_to_save / file_to_extract.name
                if file_to_save.exists():
                    continue
                with annotations_zip.open(str(file_to_extract)) as src, file_to_save.open('wb') as dst:
                    shutil.copyfileobj(src, dst)

    def _sample_audio(self, audio: np.ndarray, arousal: np.ndarray, valence: np.ndarray):
        n_targets = len(arousal)
        target_ids = np.arange(n_targets)
        self.random_state.shuffle(target_ids)

        n_targets = min(90, n_targets // 2)
        target_ids = target_ids[:n_targets]

        for part_id in target_ids:
            frame_en = self.frames_per_sec * (30 + part_id) // 2 + 1
            frame_st = frame_en - self.frames_per_sec * self.sample_duration
            self.samples.append(torch.tensor(audio[:, frame_st:frame_en], dtype=torch.float64))
            self.targets.append(torch.tensor([arousal[part_id], valence[part_id]], dtype=torch.float64))

    def _create_dataset(self, torch_samples_path: Path, torch_targets_path: Path):
        if not DEAMDataset.DATASET_PATH.exists():
            DEAMDataset.DATASET_PATH.mkdir(parents=True)
        self._download_audio()
        self._download_annotations()

        self.samples, self.targets = [], []
        arousal_df = pd.read_csv(DEAMDataset.DATASET_PATH / 'annotations' / 'arousal.csv')
        valence_df = pd.read_csv(DEAMDataset.DATASET_PATH / 'annotations' / 'valence.csv')
        audio_dir = DEAMDataset.DATASET_PATH / 'audio'

        for arousal_row, valence_row in zip(arousal_df.itertuples(), valence_df.itertuples()):
            assert arousal_row[1] == valence_row[1]  # song_id
            song_id = arousal_row[1]

            print(f'Processing song {song_id}')

            arousal = np.array(arousal_row[2:])  # arousal values for every .5 seconds since 15th second
            arousal = arousal[np.logical_not(np.isnan(arousal))]

            valence = np.array(valence_row[2:])  # valence values for every .5 seconds since 15th second
            valence = valence[np.logical_not(np.isnan(valence))]

            if len(arousal) != len(valence):
                print(f'song_id={song_id}, arousal_len={len(arousal)}, valence_len={len(valence)}', file=sys.stderr)
                common = min(len(arousal), len(valence))
                arousal = arousal[:common]
                valence = valence[:common]

            audio_path = str(audio_dir / f'{song_id}.mp3')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sample_rate = librosa.core.load(audio_path, sr=self.sample_rate, res_type='kaiser_fast')
                audio = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            self._sample_audio(audio, arousal, valence)

        self.samples = torch.stack(self.samples)
        self.targets = torch.stack(self.targets)

        torch.save(self.samples, torch_samples_path)
        torch.save(self.targets, torch_targets_path)

    def _load_dataset(self, torch_samples_path: Path, torch_targets_path: Path):
        self.samples = torch.load(torch_samples_path)
        self.targets = torch.load(torch_targets_path)

    def load(self):
        torch_samples_path = DEAMDataset.DATASET_PATH / 'samples.pt'
        torch_targets_path = DEAMDataset.DATASET_PATH / 'targets.pt'
        self.random_state = np.random.RandomState(seed=13)

        if not torch_samples_path.exists() or not torch_targets_path.exists():
            self._create_dataset(torch_samples_path, torch_targets_path)
        else:
            self._load_dataset(torch_samples_path, torch_targets_path)
        self.samples = self.samples.reshape(-1, 1, self.n_mfcc, self.frames_per_sec * self.sample_duration)

    def train_test_split(self, test_size=0.2, random_seed=0) -> Tuple[Dataset, Dataset]:
        total_size = len(self)
        train_size = int((1 - test_size) * total_size)

        ids = np.arange(total_size)
        random = np.random.RandomState(random_seed)
        random.shuffle(ids)

        train_ids, test_ids = ids[:train_size], ids[train_size:]

        train_dataset = DEAMDataset(self.sample_rate, self.n_mfcc, self.frames_per_sec, self.sample_duration)
        train_dataset.samples, train_dataset.targets = self[train_ids]

        test_dataset = DEAMDataset(self.sample_rate, self.n_mfcc, self.frames_per_sec, self.sample_duration)
        test_dataset.samples, test_dataset.targets = self[test_ids]

        return train_dataset, test_dataset

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __iter__(self):
        for sample, target in zip(self.samples, self.targets):
            yield sample, target

    def __len__(self):
        return self.samples.shape[0]
