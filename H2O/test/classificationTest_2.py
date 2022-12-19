import pandas as pd
from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML
from scipy.io.arff import loadarff
# h2o.connect(
#     url = "http://34.79.223.138:80"
# )

h2o.init()

raw_data = loadarff('data/ElectricDevices_TRAIN.arff')
df_data = pd.DataFrame(raw_data[0])

hf = h2o.H2OFrame(df_data)

y = "target"
x = hf.columns


# ALGORITHMS
#
# DRF 
# GLM
# XGBoost
# GBM
# DeepLearning
# StackedEnsemble

s = time()
aml = H2OAutoML(
    max_models = 10,
    seed = 1,
    include_algos = ["DRF"],
    project_name = "classification_test_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
aml.train(x = x, y = y, training_frame = hf)
print("Execution Time :", time() - s)

lb = aml.leaderboard

print(lb)