# Forecasting with DeepAR

### Overview

This project contains an SDK for building, training, and evaluating the DeepAR model provided by `gluonts`. 

### Installation

To get started, you can clone this project and run 
```bash
python setup.py install
```

### Usage

To initialize a forecasting object, issue the following: 
```python 
from forecast import Forecast
f = Forecast()
```

You can then load one of the supported datasets as: 
```python
dataset_name = "<your-chosen-dataset>"
f.get_dataset(name=dataset_name)
```

Alternatively, you can initiate the forecaster with a dataset. Using the `electricity` 
dataset as an example, exec: 
```python
f = Forecast(dataset_name="electricity")
```
To configure both the DeepAR architecture and trainer, issue the command: 
```python
f.build_estimator(epochs=10, learning_rate=1e-3, num_batches_per_epoch=100, context_length=336)
```

Then, training and evaluation can be completed as: 
```python
f.train()
forecast, tss, metrics = f.eval()
```

To visualize the forecast: 
```python
f.visualize_forecast(forecast=forecast, tss=tss)
```
The forecast visualizations will appear in the `figs` directory.