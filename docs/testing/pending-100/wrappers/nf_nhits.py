"""neuralforecast: an N-HiTS forecaster trained (5 steps, seeded) on a 48-point daily series when the
object is made; each step forecasts h = 4 values after the series shifted by `level`."""
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

SERIES = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(48) / 24)


class NHitsForecast:
    def __init__(self, h=4, input_size=8, max_steps=5):
        self.df = pd.DataFrame({"unique_id": "s", "ds": pd.date_range("2020-01-01", periods=48, freq="D"), "y": SERIES})
        self.nf = NeuralForecast(models=[NHITS(h=h, input_size=input_size, max_steps=max_steps, random_seed=0,
                                              enable_progress_bar=False, logger=False, enable_model_summary=False)], freq="D")
        self.nf.fit(df=self.df)

    def predict(self, level=0.0):
        df = self.df.assign(y=self.df["y"] + level)
        return self.nf.predict(df=df)["NHITS"].to_numpy()
