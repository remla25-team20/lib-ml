# lib_ml

Utilities for preprocessing text data.

## Install

```bash
pip install https://github.com/remla25-team20/lib-ml/releases/download/v0.1.3/lib_ml-0.1.3-py3-none-any.whl
```

## Usage

```py
from lib_ml.preprocessing import preprocess

preprocessed_data = preprocess(path: Path)
```

## Output

Calling `preprocess()` will also write two files into `./output/`:
- preprocessor.joblib
- preprocessed_data.joblib

