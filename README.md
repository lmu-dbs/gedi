# GEDI
**G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Installation
### Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Graphviz on your OS e.g.
For MacOS:
```console
brew install graphviz
```

- For smac:
```console
conda install pyrfr swig
```
- `conda env create -f .conda.yml`
- Install [Feature Extractor for Event Data (feeed)](https://github.com/lmu-dbs/feeed) in the newly installed conda environment: `pip install feeed`

### Startup
```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/experiment_test.json
```
## Usage
Our pipeline offers several pipeline steps, which can be run sequentially or partially:
- feature_extraction
- event_logs_generation
- benchmark_test
- evaluation_plotter

We also include two notebooks, which output experimental results as in our paper.

To run different steps of the GEDI pipeline, please adapt the `.json` accordingly.
```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/<pipeline-step>.json
```
For reference of possible keys and values for each step, please see `config_files/algorithm/experiment_test.json`.
To run the whole pipeline please create a new `.json` file, specifying all steps you want to run and specify desired keys and values for each step. 

## References
The framework used by `GEDI` is taken directly from the original paper by [...]. If you would like to discuss the paper, or corresponding research questions on benchmarking process mining tasks please email the authors.
