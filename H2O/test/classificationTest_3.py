from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML
h2o.init()

df_data = pd.read_csv("../secondary_data.csv", delimiter = ";")
hf = h2o.H2OFrame(df_data)

y = "class"
x = hf.columns

s = time()
aml = H2OAutoML(max_models = 10, seed = 1,  project_name = "classification_test_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
aml.train(x = x, y = y, training_frame = hf)
print("Execution Time :", time() - s)

lb = aml.leaderboard

h2o.remove(aml)