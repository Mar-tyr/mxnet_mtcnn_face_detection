import json
import os

from pathlib import Path
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.fx.all import resize
from skimage.measure import compare_ssim as ssim
from scipy.misc import imresize
import cv2
from skimage import color
from tqdm import tqdm
import os.path as osp


def track_video_face(video_dir, subject_id, outdir):
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    video_path = video_dir / 'P{}.mp4'.format(subject_id)
    clip = VideoFileClip(str(video_path))
    clip = resize(clip, (160, 90))
    face_frames = []
    for i, frame in tqdm(enumerate(clip.iter_frames())):
        results = detector.detect_face(frame)
        if results is not None:
            total_boxes = results[0]
            for box in total_boxes:
                if (box[2] - box[0]) > 24 and (box[3] - box[1]) > 24:
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    tmp_x1 = max(tmp_x1, 0)
                    tmp_y1 = max(tmp_y1, 0)
                    tmp_x2 = max(tmp_x2, 0)
                    tmp_y2 = max(tmp_y2, 0)
                    tmp_face = imresize(frame[tmp_y1:tmp_y2, tmp_x1:tmp_x2], (48, 48))
                    if i == 0:
                        x1, y1, x2, y2 = tmp_x1, tmp_y1, tmp_x2, tmp_y2
                        face = tmp_face
                    else:
                        tmp_face_gray = color.rgb2gray(tmp_face)
                        last_face_gray = color.rgb2gray(last_face)
                        sml = ssim(last_face_gray, tmp_face_gray,
                                   data_range=(tmp_face_gray.max() - tmp_face_gray.min()))
                        if sml > 0.3:
                            x1, y1, x2, y2 = tmp_x1, tmp_y1, tmp_x2, tmp_y2
                            face = tmp_face
                        else:
                            face = imresize(frame[y1:y2, x1:x2], (48, 48))
                else:
                    face = imresize(frame[y1:y2, x1:x2], (48, 48))
        else:
            face = imresize(frame[y1:y2, x1:x2], (48, 48))
        face_frames.append(face)
        last_face = face
    res_clip = ImageSequenceClip(face_frames, fps=25)
    face_videodir = str(outdir / 'Video')
    if not osp.isdir(face_videodir):
        os.makedirs(face_videodir)
    res_clip.to_videofile('{}/P{}.mp4'.format(face_videodir, subject_id))


if __name__ == '__main__':
    with open('config/path.json', 'r') as fp:
        config = json.load(fp)
    root_dir = Path(config['windows'])
    video_dir = root_dir / 'RECOLA-Video-recordings'
    anno_dir = root_dir / 'RECOLA-Annotation/emotional_behaviour'
    outdir = root_dir / 'recola_out'
    track_video_face(video_dir, 16, outdir)