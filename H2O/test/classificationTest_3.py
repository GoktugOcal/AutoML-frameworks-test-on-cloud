import pandas as pd
from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML


h2o.connect(
    url = "http://34.79.223.138:80"
)

df_data = pd.read_csv("H2O/test/data/secondary_data.csv", delimiter = ";")
hf = h2o.H2OFrame(df_data)
print(hf.describe())

y = "class"
x = hf.columns

s = time()
aml = H2OAutoML(max_runtime_secs = 300, max_models = 10, seed = 1,  project_name = "classification_test_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
aml.train(x = x, y = y, training_frame = hf)
print("Execution Time :", time() - s)

lb = aml.leaderboard
print(lb)

h2o.remove(aml)