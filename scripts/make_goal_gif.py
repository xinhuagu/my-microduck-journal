# -*- coding: utf-8 -*-
"""Cut the goal-trial video into a README GIF: <mp4> <out.gif> [start_s=0] [end_s=13] [fps=10] [width=480].
Run inside microduck_rl (uv run python ...) so imageio + its ffmpeg plugin are available."""
import sys
import imageio.v3 as iio
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
start_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
end_s = float(sys.argv[4]) if len(sys.argv) > 4 else 13.0
fps_out = int(sys.argv[5]) if len(sys.argv) > 5 else 10
width = int(sys.argv[6]) if len(sys.argv) > 6 else 480

import imageio.v2 as iio2
reader = iio2.get_reader(src, "ffmpeg")           # imageio-ffmpeg plugin (what the recorder used to write it)
src_fps = 50.0                                    # the recorder writes one frame per control step
step = max(1, int(round(src_fps / fps_out)))
i0, i1 = int(start_s * src_fps), int(end_s * src_fps)
out, T = [], 0
for i, frame in enumerate(reader):
    T = i + 1
    if i < i0 or i >= i1 or (i - i0) % step:
        continue
    im = Image.fromarray(np.asarray(frame))
    h = int(im.height * width / im.width)
    out.append(im.resize((width, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128))
reader.close()
out[0].save(dst, save_all=True, append_images=out[1:], duration=int(1000 / fps_out), loop=0, optimize=True)
import os
print(f"{dst}: {len(out)} frames, {out[0].size[0]}x{out[0].size[1]}, {os.path.getsize(dst)/1e6:.2f} MB (source {T} frames)")
