from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML

h2o.connect(
    url = "http://34.79.223.138:80"
)

# Use local data file or download from GitHub
import os
docker_data_path = "/home/h2o/data/automl/powerplant_output.csv"
if os.path.isfile(docker_data_path):
    data_path = docker_data_path
else:
    data_path = "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/powerplant_output.csv"


# Load data into H2O
df = h2o.import_file(data_path)
print(df.describe())

y = "HourlyEnergyOutputMW"
splits = df.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]

s = time()
aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "regression_test_" + datetime.now().strftime("%Y-%m-%dT%H.%M.%S"))
aml.train(y = y, training_frame = train, leaderboard_frame = test)

print("Execution Time :", time() - s)

print(aml.leaderboard.head())

#Predict
pred = aml.predict(test)

#Performance
perf = aml.leader.model_performance(test)
print(perf)

# h2o.remove(aml)