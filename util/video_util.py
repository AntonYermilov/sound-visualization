from moviepy.editor import *

def to_rgb(img):
    res = np.zeros((28, 28, 3))
    rows = img.shape[0]
    cols = img.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            col = int(img[x, y] * 255)
            res[x, y, 0] = col
            res[x, y, 1] = col
            res[x, y, 2] = col
    
    return res

#tensor.Size([1, 28, 28])
def make_video(tensors, file_name="test", fps=40):
    img = [to_rgb(np.squeeze(t.detach().numpy())) for t in range(len(tensors))]

    clips = [ImageClip(m).set_duration(1.0/fps) for m in img]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(file_name + ".mp4", fps=fps)
