import warnings

warnings.filterwarnings("ignore")

from itertools import islice

import matplotlib.dates as mdates
import pandas as pd
import torch
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from matplotlib import pyplot as plt


class Forecast(dict):
    def __init__(self, dataset_name: str = None):

        super().__init__()

        self.dataset = None
        self.dataset_name = dataset_name
        if dataset_name:
            self.get_dataset(name=dataset_name)

        self.date_formater = mdates.DateFormatter("%Y")

        self.estimator = None
        self.predictor = None
        self.context_length = None

    def build_estimator(
        self,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        num_batches_per_epoch: int = 100,
        context_length: int = 2 * 7 * 24,
    ):
        self.estimator = DeepAREstimator(
            freq=self.dataset.metadata.freq,
            prediction_length=self.dataset.metadata.prediction_length,
            context_length=context_length,
            # TODO: Add other degrees of freedom to Trainer to expand hyperparameter search
            trainer=Trainer(
                ctx="gpu" if torch.cuda.is_available() else "cpu",
                epochs=epochs,
                learning_rate=learning_rate,
                num_batches_per_epoch=num_batches_per_epoch,
            ),
        )
        self.context_length = context_length

    def train(self):
        self.predictor = self.estimator.train(self.dataset.train)

    def eval(self):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.dataset.test, predictor=self.predictor
        )
        forecasts = [f for f in forecast_it]
        tss = list(ts_it)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        metrics, _ = evaluator(
            iter(tss), iter(forecasts), num_series=len(self.dataset.test)
        )
        return (
            forecasts,
            tss,
            pd.DataFrame.from_records(metrics, index=["DeepAR"]).transpose(),
        )

    def get_dataset(self, name: str = "electricity"):
        if name not in dataset_recipes.keys():
            raise KeyError(
                f"The dataset name {name} is not supported. \nPlease select from "
                + "\t".join(dataset_recipes.keys())
            )
        self.dataset = get_dataset(name, regenerate=False)
        self.dataset_name = name

    def visualize_dataset(self):
        if not self.dataset_name or not self.dataset:
            raise Exception("No dataset has been loaded yet.")

        fig = plt.figure(figsize=(12, 8))
        for idx, entry in enumerate(islice(self.dataset.train, 9)):
            ax = plt.subplot(3, 3, idx + 1)
            t = pd.date_range(
                start=entry["start"],
                periods=len(entry["target"]),
                freq=entry["start"].freq,
            )
            plt.plot(t, entry["target"])
            # plt.xticks(pd.date_range(start=pd.to_datetime("2011-12-31"), periods=3, freq="AS"))
            ax.xaxis.set_major_formatter(self.date_formater)
        plt.savefig(f"figs/{self.dataset_name}-visualization.png")

    def visualize_forecast(self, forecast, tss):

        for idx, (forecast, ts) in enumerate(zip(forecast, tss)):
            plt.figure(figsize=(20, 15))
            plt.rcParams.update({"font.size": 15})
            plt.plot(ts[-5 * self.dataset.metadata.prediction_length :], label="target")
            forecast.plot()
            plt.gcf().tight_layout()
            plt.savefig(f"figs/{self.dataset_name}-forecast-{idx}.png")
