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

print("#### Test 3")
experiments.classification3(max_runtime_secs=3600)

print("# Saving")
experiments.logs_save()