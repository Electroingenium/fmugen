"""torchaudio: the wav2vec 2.0 ASR bundle (base, 960 h), greedy CTC decoding of the 16 kHz speech
clip given as a file; returns the transcript."""
import torch
import torchaudio


class Wav2Vec2ASR:
    def __init__(self):
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().eval()
        self.labels = self.bundle.get_labels()

    def transcribe(self, path="speech.pt", gain=1.0):
        with torch.inference_mode():
            emission, _ = self.model(torch.load(path) * gain)
        ids = torch.unique_consecutive(emission[0].argmax(-1)).tolist()
        return "".join(self.labels[i] for i in ids if i != 0).replace("|", " ").strip()
