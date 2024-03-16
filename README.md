# GEDI

### Requirements
- [Meta-feature Extractor](https://github.com/gbrltv/process_meta_learning/tree/main/meta_feature_extraction)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Graphviz on your OS e.g.
For MacOS:
```console
brew install graphviz
```

## Installation
- For smac:
```console
conda install pyrfr swig
```
- `conda env create -f .conda.yml`
- Install [Feature Extractor for Event Data (feeed)](https://github.com/lmu-dbs/feeed) in the newly installed conda environment: `pip install feeed`

### Startup
```console
conda activate tag
python main.py -o config_files/options/baseline.json -a config_files/algorithm/experiment_test.json
```
