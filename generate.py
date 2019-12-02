import librosa
import argparse
import torch
import warnings
import os
import cv2
import moviepy.editor as mpe

SAMPLE_RATE = 20480
N_MFCC = 40
FRAMES_PER_SEC = 40
SAMPLE_DURATION = 4
HOP_LENGTH = SAMPLE_RATE // FRAMES_PER_SEC
FRAMES_IN_DURATION = SAMPLE_DURATION * FRAMES_PER_SEC

IMAGES_PATH = 'images'

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

def create_video(video_path):
    images_dir = os.fsencode(IMAGES_PATH)
    images = [os.fsdecode(el) for el in os.listdir(images_dir)
                        if os.fsdecode(el).endswith(".png")]
    """os.fsdecode(el).isnumeric() and"""
    #images.sort()
    print(len(images))

    frame = cv2.imread(os.path.join(IMAGES_PATH, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(IMAGES_PATH, image)))

    cv2.destroyAllWindows()
    video.release()

def convert_to_mp4(video_path):
    my_clip = mpe.VideoFileClip(video_path)
    my_clip.write_videofile("video.mp4")

def add_music_to_video(video_path, audio_path):
    #TODO: ffmpeg -i video.mp4 -i audio.mp3 -c copy output.mkv
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, help='path to audio, which is used to generate video')
    parser.add_argument('--video_path', type=str, help='path to video, which was generated')
    args = parser.parse_args()

    samples = split_to_samples(args.audio_path)

    #TODO: add video generation via model

    create_video("video.avi")
    convert_to_mp4("video.avi")
    #add_music_to_video("video.mp4", args.audio_path)
