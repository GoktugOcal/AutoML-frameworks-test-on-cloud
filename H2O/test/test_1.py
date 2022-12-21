from test import *

import pandas as pd
import json
from time import time
from datetime import datetime
import h2o
from h2o.automl import H2OAutoML

h2o.connect(
    url = ADDRESS
)

experiments = test(h2o)

dt = experiments.project_created_time

print("#### Algorithms Test")
for algo in ALGORITHMS:
    print(algo)
    experiments.classification1(max_runtime_secs=1800, include_algos=[algo])

print("# Saving")
experiments.logs_save()