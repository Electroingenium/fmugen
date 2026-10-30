"""GluonTS: a DeepAR estimator (torch) trained for one epoch, seeded, on a 48-point hourly series when
the object is made; each step gives the mean forecast of the next 4 values after the series shifted by
`level`."""
import lightning
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator

SERIES = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(48) / 24)


class DeepARForecast:
    def __init__(self, prediction_length=4, epochs=1):
        lightning.seed_everything(0, verbose=False)
        est = DeepAREstimator(freq="h", prediction_length=prediction_length, num_batches_per_epoch=5,
                              trainer_kwargs={"max_epochs": epochs, "enable_progress_bar": False, "logger": False,
                                              "enable_model_summary": False})
        self.predictor = est.train(ListDataset([{"start": "2020-01-01", "target": SERIES}], freq="h"))

    def predict(self, level=0.0):
        lightning.seed_everything(0, verbose=False)
        data = ListDataset([{"start": "2020-01-01", "target": SERIES + level}], freq="h")
        return next(iter(self.predictor.predict(data))).mean
