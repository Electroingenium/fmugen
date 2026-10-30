"""pytorch-forecasting: a Temporal Fusion Transformer (hidden size 8) trained for one epoch, seeded,
on a 60-point series when the object is made; each step predicts the next 4 values after the series
shifted by `level`."""
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


class TFTForecast:
    def __init__(self, hidden_size=8, epochs=1):
        pl.seed_everything(0, verbose=False)
        t = np.arange(60)
        self.df = pd.DataFrame({"time_idx": t, "group": "s", "y": 20.0 + 5.0 * np.sin(2 * np.pi * t / 24)})
        self.ds = TimeSeriesDataSet(self.df, time_idx="time_idx", target="y", group_ids=["group"],
                                    max_encoder_length=12, max_prediction_length=4,
                                    time_varying_unknown_reals=["y"])
        self.tft = TemporalFusionTransformer.from_dataset(self.ds, hidden_size=hidden_size, attention_head_size=1)
        pl.Trainer(max_epochs=epochs, logger=False, enable_progress_bar=False, enable_model_summary=False,
                   enable_checkpointing=False).fit(self.tft, self.ds.to_dataloader(train=True, batch_size=16))

    def predict(self, level=0.0):
        df = self.df.assign(y=self.df["y"] + level)
        data = TimeSeriesDataSet.from_dataset(self.ds, df, predict=True)
        return self.tft.predict(data, trainer_kwargs={"logger": False, "enable_progress_bar": False}).numpy()[0]
