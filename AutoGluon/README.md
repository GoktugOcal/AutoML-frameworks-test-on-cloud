## Installation

1. Install dependencies

```shell
pip install -U pip
pip install -U setuptools wheel
```

2. Use `pip` to install the AutoGluon Python module.

```shell
pip install autogluon
```

## Demo

```python
from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = TabularPredictor(label='class').fit(train_data, time_limit=120)  # Fit models for 120s
leaderboard = predictor.leaderboard(test_data)
```