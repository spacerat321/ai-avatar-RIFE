import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# Set float32 matmul precision for Torch 2.0+
torch.set_float32_matmul_precision("medium")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0):
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log')
parser.add_argument('--fp16', dest='fp16', action='store_true')
parser.add_argument('--UHD', dest='UHD', action='store_true')
parser.add_argument('--scale', dest='scale', type=float, default=1.0)
parser.add_argument('--skip', dest='skip', action='store_true')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true')
parser.add_argument('--ext', dest='ext', type=str, default='mp4')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()

assert (args.video is not None or args.img is not None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if args.img is not None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# Load model
try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    try:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded ArXiv-RIFE model")

model.eval()
model.device()

# Load input
if args.video:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    fpsNotAssigned = args.fps is None
    args.fps = fps * (2 ** args.exp) if fpsNotAssigned else args.fps
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    if not args.png and fpsNotAssigned:
        print("Audio will be merged after processing.")
    else:
        print("No audio merge due to --fps or --png.")
else:
    videogen = sorted([f for f in os.listdir(args.img) if 'png' in f], key=lambda x: int(x[:-4]))
    tot_frame = len(videogen)
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]

h, w, _ = lastframe.shape
if args.montage:
    left = w // 4
    w = w // 2
    lastframe = lastframe[:, left: left + w]

# Output setup
if args.png:
    os.makedirs('vid_out', exist_ok=True)
else:
    vid_out_name = args.output or f'{video_path_wo_ext}_{2 ** args.exp}X_{int(np.round(args.fps))}fps.{args.ext}'
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
pbar = tqdm(total=tot_frame)

def pad_image(img):
    return F.pad(img, padding).half() if args.fp16 else F.pad(img, padding)

def build_read_buffer(args, read_buffer, videogen):
    try:
        for frame in videogen:
            if args.img:
                frame = cv2.imread(os.path.join(args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def clear_write_buffer(args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if args.png:
            cv2.imwrite(f'vid_out/{cnt:07d}.png', item[:, :, ::-1])
        else:
            vid_out.write(item[:, :, ::-1])
        cnt += 1

def make_inference(I0, I1, n):
    with autocast(enabled=args.fp16):
        middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    return [*first_half, middle, *second_half] if n % 2 else [*first_half, *second_half]

_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None

while True:
    frame = temp or read_buffer.get()
    if frame is None:
        break
    temp = None
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32))
    I1_small = F.interpolate(I1, (32, 32))
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    if ssim > 0.996:
        frame = read_buffer.get()
        if frame is None:
            break
        temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        with autocast(enabled=args.fp16):
            I1 = model.inference(I0, I1, args.scale)
        I1_small = F.interpolate(I1, (32, 32))
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    output = make_inference(I0, I1, 2 ** args.exp - 1) if args.exp else []

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            write_buffer.put(np.concatenate((lastframe, (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            write_buffer.put((mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    pbar.update(1)
    lastframe = frame

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)

write_buffer.put(None)

import time
while not write_buffer.empty():
    time.sleep(0.1)
pbar.close()

if not args.png and fpsNotAssigned and args.video:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        fallback = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(fallback, vid_out_name)
