## Installation

1. Install dependencies

```shell
pip install requests
pip install tabulate
pip install future
```

2. Use `pip` or `conda` to install this version of the H2O Python module.

```shell
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

```shell
conda install -c h2oai h2o
```

## Demo

```python
import h2o
h2o.init()
h2o.demo("glm")
```