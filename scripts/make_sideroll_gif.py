# -*- coding: utf-8 -*-
"""Side-roll GIF: two windows of the exam video (roll left, roll right), labelled, real time.
usage: make_sideroll_gif.py <mp4> <out.gif> [fps=25] [width=400] [windows=0.0-2.0:left,6.0-8.0:right]"""
import sys
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

src, dst = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 25
width = int(sys.argv[4]) if len(sys.argv) > 4 else 400
spec = sys.argv[5] if len(sys.argv) > 5 else "0.0-2.0:roll left,6.0-8.0:roll right"
windows = []
for part in spec.split(","):
    rng, label = part.split(":")
    a, b = map(float, rng.split("-"))
    windows.append((int(a * 50), int(b * 50), label))
step = max(1, round(50 / fps))
frames = {}
r = iio.get_reader(src, "ffmpeg")
for i, f in enumerate(r):
    for a, b, label in windows:
        if a <= i < b and (i - a) % step == 0:
            frames[i] = (np.asarray(f), label)
r.close()
out = []
for i in sorted(frames):
    arr, label = frames[i]
    im = Image.fromarray(arr)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, 6 + 8 * len(label) + 12, 26], fill=(20, 30, 50))
    d.text((12, 10), label, fill=(255, 255, 255))
    out.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
out[0].save(dst, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0, optimize=True)
import os
print(f"{dst}: {len(out)} frames, {out[0].size[0]}x{out[0].size[1]}, {os.path.getsize(dst)/1e6:.2f} MB")
