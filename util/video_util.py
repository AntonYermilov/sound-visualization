import numpy as np
import torch
from moviepy.editor import concatenate_videoclips
from moviepy.editor import ImageClip, AudioFileClip
from pathlib import Path
from typing import Optional


def to_rgb(img: np.ndarray) -> np.ndarray:
    channels, rows, cols = img.shape
    img = (img.transpose((1, 2, 0)) * 255).astype(np.int32)

    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def make_video(tensors: torch.Tensor, video_file: Path, fps: int = 40, audio_file: Optional[Path] = None):
    img = [to_rgb(t.detach().numpy()) for t in tensors]

    clips = [ImageClip(m).set_duration(1.0 / fps) for m in img]

    video_clip = concatenate_videoclips(clips, method="compose")
    if audio_file:
        audio_clip = AudioFileClip(str(audio_file))
        video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(video_file), fps=fps)
