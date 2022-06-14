from datetime import datetime, timedelta
from enum import Enum
from typing import List, Mapping

import numpy as np
import pandas as pd


class SalesDataModel(Enum):
    RANDOM = 0
    SINUSOID = 1
    SPARSE = 2


class SalesDataGenerator:
    def __init__(
        self,
        start_date: datetime,
        num_days: int,
        item_to_location: Mapping[str, List[str]],
        item_to_class: Mapping[str, List[str]],
        min_sales_per_day: int = 0,
        max_sales_per_day: int = 1000,
    ):
        self.start_date = start_date
        self.num_days = num_days
        self.item_to_location = item_to_location
        self.item_to_class = item_to_class
        self.min_sales_per_day = min_sales_per_day
        self.max_sales_per_day = max_sales_per_day

    def synthesize(self, data_model: SalesDataModel = SalesDataModel.RANDOM):
        df = pd.DataFrame(columns=["Date", "Item", "Location", "Sales"])
        for item, locations in self.item_to_location.items():
            for location in locations:
                if data_model == SalesDataModel.RANDOM:
                    sales = np.random.randint(
                        self.min_sales_per_day, self.max_sales_per_day, self.num_days
                    )
                elif data_model == SalesDataModel.SINUSOID:
                    alpha = 0.9
                    gain = alpha * (self.max_sales_per_day - self.min_sales_per_day)
                    periods = np.random.randint(1, 6, 1)
                    nn = np.linspace(0, int(periods), self.num_days)
                    base_sales = (
                        gain * np.abs(np.cos(np.pi * nn))
                        + alpha * self.min_sales_per_day
                    )
                    noise = np.random.randint(
                        (1 - alpha) * self.min_sales_per_day,
                        (1 - alpha) * self.max_sales_per_day,
                        self.num_days,
                    )
                    sales = base_sales + noise
                    sales = sales.astype(int).tolist()
                elif data_model == SalesDataModel.SPARSE:
                    thresh = 0.1
                    sales = [
                        0
                        if np.random.rand() > thresh
                        else np.random.randint(
                            self.min_sales_per_day, self.max_sales_per_day, 1
                        )
                        for _ in range(self.num_days)
                    ]
                _df = pd.DataFrame(
                    {
                        "Date": [
                            self.start_date + timedelta(days=diff)
                            for diff in range(self.num_days)
                        ],
                        "Item": [item] * self.num_days,
                        "Location": [location] * self.num_days,
                        "Sales": sales,
                        "Item Class": [self.item_to_class[item]] * self.num_days,
                    }
                )
                df = pd.concat([df, _df], ignore_index=True)
        return df
