from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML

h2o.connect(
    url = "http://34.79.223.138:80"
)
# Use local data file or download from GitHub
import os
docker_data_path = "/home/h2o/data/automl/product_backorders.csv"
if os.path.isfile(docker_data_path):
    data_path = docker_data_path
else:
    data_path = "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/product_backorders.csv"


# Load data into H2O
df = h2o.import_file(data_path)
print(df.describe())

y = "went_on_backorder"
x = df.columns
x.remove(y)
x.remove("sku")

s = time()
aml = H2OAutoML(max_models = 10, seed = 1, project_name = "classification_test_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
aml.train(x = x, y = y, training_frame = df)

print("Execution Time :", time() - s)

lb = aml.leaderboard
print(lb)

# h2o.remove(aml)