"""Input data for the models of models_r4.py, in ~/cmp/wr4/data (run with a venv that has numpy,
pillow, torch and soundfile).

Downloaded (published examples): ultralytics' bus.jpg and zidane.jpg, whisper's jfk.flac, pycalphad's
alzn_mey.tdb, BioModels' BIOMD0000000010 (Kholodenko 2000, MAPK cascade), biosppy's ecg.txt, feedinlib's example weather.csv, flopy's MODFLOW 6 test001a_Tharmonic, CityFlow's 1x1 example, piper's
en_US-lessac-low voice, MediaPipe's pose_landmarker_lite.
MODFLOW 6 for flopy: `get-modflow ~/cmp/wr4/data/bin --subset mf6` in flopy's venv. Made from them: image tensors for the vision models, raw and wave speech.
"""
import json
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly
import torch
from PIL import Image, ImageDraw, ImageFont

D = Path.home() / "cmp/wr4/data"
D.mkdir(parents=True, exist_ok=True)
GH = "https://raw.githubusercontent.com"
URLS = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    "jfk.flac": "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
    "alzn_mey.tdb": f"{GH}/pycalphad/pycalphad/develop/examples/databases/alzn_mey.tdb",
    "BIOMD0000000010.xml": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000010?filename=BIOMD0000000010_url.xml",
    "feedinlib_weather_raw.csv": f"{GH}/oemof/feedinlib/dev/examples/weather.csv",
    "ecg.txt": f"{GH}/PIA-Group/BioSPPy/master/examples/ecg.txt",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "en_US-lessac-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx",
    "en_US-lessac-low.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
}
MF6 = f"{GH}/modflowpy/flopy/develop/examples/data/mf6/test001a_Tharmonic"
for f in ("mfsim.nam", "flow15.dis", "flow15.ic", "flow15.ims", "flow15.nam", "flow15.npf", "flow15.oc", "flow15_constant.chd", "flow15.tdis"):
    URLS[f"mf6_tharmonic/{f}"] = f"{MF6}/{f}"
for f in ("config.json", "roadnet.json", "flow.json"):
    URLS[f"cityflow/{f}"] = f"{GH}/cityflow-project/CityFlow/master/examples/{f}"
for f, u in URLS.items():
    p = D / f
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(u, p)
        except Exception as e:
            print("download failed", f, e)

# CityFlow's example config names its files relative to a dir: point it at this folder
cf = D / "cityflow/config.json"
if cf.exists():
    c = json.loads(cf.read_text())
    c["dir"] = str(D / "cityflow") + "/"
    c["roadnetFile"], c["flowFile"] = "roadnet.json", "flow.json"
    c["saveReplay"] = False
    cf.write_text(json.dumps(c, indent=1))

# feedinlib: the pvlib weather of its simple_feedin.py example (°C, ghi = dirhi + dhi), one day (1 June 2010)
import pandas as pd
w = pd.read_csv(D / "feedinlib_weather_raw.csv", index_col=0)
w.index = pd.to_datetime(w.index, utc=True).tz_convert("Europe/Berlin")
w["temp_air"] = w.temp_air - 273.15
w["ghi"] = w.dirhi + w.dhi
w = w.rename(columns={"v_wind": "wind_speed"})[["wind_speed", "temp_air", "dhi", "ghi"]]
w.loc["2010-06-01"].to_csv(D / "feedinlib_weather.csv")

bus = Image.open(D / "bus.jpg").convert("RGB")
a = np.asarray(bus.resize((224, 224)), dtype="float32") / 255.0
mean, std = np.array([0.485, 0.456, 0.406], "float32"), np.array([0.229, 0.224, 0.225], "float32")
torch.save(torch.from_numpy(((a - mean) / std).transpose(2, 0, 1).copy())[None], D / "img224.pt")
np.save(D / "bus_rgb.npy", np.asarray(bus))
# SAM: longest side 1024, ImageNet-normalized, padded to 1024x1024
w, h = bus.size
s = 1024 / max(w, h)
b = np.asarray(bus.resize((round(w * s), round(h * s))), dtype="float32") / 255.0
b = (b - mean) / std
pad = np.zeros((1024, 1024, 3), "float32")
pad[:b.shape[0], :b.shape[1]] = b
torch.save(torch.from_numpy(pad.transpose(2, 0, 1).copy())[None], D / "sam1024.pt")
Image.open(D / "zidane.jpg").convert("RGB").save(D / "face.jpg")
t = Image.new("RGB", (320, 80), "white")
ImageDraw.Draw(t).text((10, 20), "FMUGEN 2026", fill="black", font=ImageFont.load_default(size=36))
t.save(D / "text.png")
# TrOCR: 384x384, normalized to [-1, 1]
tr = np.asarray(t.resize((384, 384)), dtype="float32") / 255.0
torch.save(torch.from_numpy(((tr - 0.5) / 0.5).transpose(2, 0, 1).copy())[None], D / "trocr384.pt")

# speech: the first 3 s of jfk.flac (44.1 kHz), resampled to 16 kHz mono
s, sr = soundfile.read(D / "jfk.flac", dtype="float32")
if s.ndim > 1:
    s = s.mean(axis=1)
s = resample_poly(s, 16000, sr).astype("float32")[:48000]
np.save(D / "speech.npy", s)
torch.save(torch.from_numpy(s.copy())[None], D / "speech.pt")
pcm = (s * 32767).astype("<i2").tobytes()
(D / "speech.raw").write_bytes(pcm)
with wave.open(str(D / "speech.wav"), "wb") as wv:
    wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000)
    wv.writeframes(pcm)
print("data in", D, sorted(p.name for p in D.iterdir()))
