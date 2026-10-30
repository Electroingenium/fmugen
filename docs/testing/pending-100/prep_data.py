"""Input data for the pending models, in ~/cmp/w100/data (run with the driver venv: numpy, pillow, torch).

Downloaded: ultralytics' bus.jpg, the ONNX model zoo's MNIST, whisper's jfk.flac (speech), EPANET's
Net1.inp. Made here: a 224x224 image tensor, a small PNG, a text PNG, a face-like crop of bus.jpg.
"""
import io
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

D = Path.home() / "cmp/w100/data"
D.mkdir(parents=True, exist_ok=True)
URLS = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "mnist-12.onnx": "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
    "jfk.flac": "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
    "Net1.inp": "https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net1.inp",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}
for f, u in URLS.items():
    if not (D / f).exists():
        try:
            urllib.request.urlretrieve(u, D / f)
        except Exception as e:
            print("download failed", f, e)

img = Image.open(D / "bus.jpg").convert("RGB").resize((224, 224))
a = np.asarray(img, dtype="float32") / 255.0
mean, std = np.array([0.485, 0.456, 0.406], "float32"), np.array([0.229, 0.224, 0.225], "float32")
torch.save(torch.from_numpy(((a - mean) / std).transpose(2, 0, 1).copy())[None], D / "img224.pt")
np.save(D / "img224_nhwc.npy", (a[None] * 2.0 - 1.0).astype("float32"))   # MobileNetV2: [-1, 1]
Image.open(D / "bus.jpg").convert("RGB").resize((64, 64)).save(D / "small.png")
Image.open(D / "zidane.jpg").convert("RGB").save(D / "face.jpg")
t = Image.new("RGB", (320, 80), "white")
ImageDraw.Draw(t).text((10, 20), "FMUGEN 2026", fill="black", font=ImageFont.load_default(size=36))
t.save(D / "text.png")

# speech: 16 kHz mono float32, first 3 s of jfk.flac (decoded with ffmpeg if present, else a tone)
try:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-i", str(D / "jfk.flac"), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    s = np.frombuffer(raw, "<i2").astype("float32") / 32768.0
except Exception as e:
    print("no ffmpeg, using a tone:", e)
    s = (0.3 * np.sin(2 * np.pi * 220 * np.arange(48000) / 16000)).astype("float32")
s = s[:48000]
np.save(D / "speech.npy", s)
torch.save(torch.from_numpy(s.copy())[None], D / "speech.pt")
with wave.open(str(D / "speech.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((s * 32767).astype("<i2").tobytes())
print("data in", D, sorted(p.name for p in D.iterdir()))
