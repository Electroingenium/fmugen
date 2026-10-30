"""uni2ts: Salesforce's Moirai (1.0, small) foundation forecaster, zero-shot through its GluonTS
predictor; each step gives the median of 20 sample paths (seeded) over the next 4 hours of a
48-point hourly series shifted by `level`."""
import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

SERIES = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(48) / 24)


class MoiraiForecaster:
    def __init__(self, prediction_length=4, context_length=48):
        model = MoiraiForecast(module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
                               prediction_length=prediction_length, context_length=context_length, patch_size="auto",
                               num_samples=20, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        self.predictor = model.create_predictor(batch_size=1)

    def predict(self, level=0.0):
        torch.manual_seed(0)
        data = ListDataset([{"start": "2020-01-01", "target": SERIES + level}], freq="h")
        return np.median(next(iter(self.predictor.predict(data))).samples, axis=0)
