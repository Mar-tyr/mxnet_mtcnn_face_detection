from moviepy.editor import VideoFileClip
import cv2
from skimage import color
from skimage.measure import compare_ssim as ssim
from scipy.misc import imsave
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    video_path = '17.mp4'
    clip = VideoFileClip(video_path)
    smls = []
    cmp_imgs = []
    for i, frame in (enumerate(clip.iter_frames())):
        frame = color.rgb2gray(frame)
        if i == 0:
            last_frame = frame
            continue
        sml = ssim(last_frame, frame, data_range=(frame.max() - frame.min()))
        if sml < 0.3:
            cmp_img = np.hstack((last_frame, frame))
            cmp_imgs.append((sml, (cmp_img)))
        last_frame = frame
    cmp_imgs.sort(key=lambda x:x[0])
    smls = [cmp_img[0] for cmp_img in cmp_imgs]
    print len(smls)
    cmp_imgs = [cmp_img[1] for cmp_img in cmp_imgs]
    total_cmp_img = np.vstack(cmp_imgs)
    imsave('cmp.jpg', total_cmp_img)
