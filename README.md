# lib_ml

Utilities for preprocessing text data.

## Usage

```py
from lib_ml.preprocessing import preprocess

preprocessed_data = preprocess(path: Path)
```

## Output

Calling `preprocess()` will also write two files into `./output/`:
- preprocessor.joblib
- preprocessed_data.joblib

