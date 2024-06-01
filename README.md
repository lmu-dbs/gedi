# GEDI
**G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Graphviz on your OS e.g.
For MacOS:
```console
brew install graphviz
brew install swig
```
- For smac:
```console
conda install pyrfr swig
```
## Installation
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
- generation
- benchmark
- evaluation_plotter

We also include two notebooks, which output experimental results as in our paper.

To run different steps of the GEDI pipeline, please adapt the `.json` accordingly.
```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/<pipeline-step>.json
```
For reference of possible keys and values for each step, please see `config_files/algorithm/experiment_test.json`.
To run the whole pipeline please create a new `.json` file, specifying all steps you want to run and specify desired keys and values for each step. 

## Evaluation real targets
In order to execute the experiments with real targets,we employ the config file `config_files/algorithm/experiment_real_targets.json`. The script's pipeline will output the generated event logs with meta features values being optimized towards meta features of real-world benchmark datasets. Furthermore, it will output the respective feature values in the `\output`folder as well as the benchmark values.

```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/experiment_real_targets.json.json
```

## Result plotting
In the following, we describe the ipynb in the folder `\notebooks` to reproduce the illustrations from our paper. 


### gedi_fig6_benchmark_boxplots.ipynb
This notebook is used to visualize the metric distribution of real event logs compared to the generated ones. It shows 5 different metrics on 3 various process discovery techniques. We use 'fitness,', 'precision', 'fscore', 'size', 'cfc' (control-flow complexity) as metrics and as 'heuristic miner', 'ilp' (integer linear programming), and 'imf' (inductive miner infrequent) as miners. The notebook outputs the visualization shown in Fig.6 in the paper.

### gedi_figs4and5_representativeness.ipynb
In order to visualize the coverage of the feasible meta feature space of synthesied event logs compared to existing real-world benchmark datasets, in this notebook, we condcut a principal compoentn analysis on the meta features of both settings. The first two principal components are utilized to visualize the coverage which is further highlighted by computing a convex hull of the 2D-mapping.  Additionally, we visualize the distribution of each meta feature we used in the paper as a boxplot. Additional features can be extracted with FEEED. Therefore, the notebook contains the figures 4 and 5 in the paper. 

### gedi_figs7and8_benchmarking_statisticalTests.ipynb

This notebook is used to ansewr the question if there is a statistical signifacnt relation between feature similarity and performance metrics for the downstream tasks of process discovery. For that, we compute the pearson coefficient, as well as the kendall's tau coefficient. This elucidates the correlation between the meta features with metric scores being used for process discovery. Each coefficuent is calculated for three different settings: i) real-world datasets; ii) synthesized event log data with real-world targets; iii) synthesized event log data with grid objectives. The figures 7 and 8 shown in the paper refer to this notebook.

### gedi_figs9and10_consistency.ipynb
Likewise to the evaluation on the statistical tests in notebook `gedi_figs7and8_benchmarking_statisticalTests.ipynb`, this notebook is used to compute the differences between two correlation matrices $\Delta C = C_1 - C_2$. This logic is employed to evaluate and visualize the distance of two correlation matrices. Furthermore, we show how significant scores are retained from the correlations being evaluated on real-world datasets coompared to synthesized event log datasets with real-world targets. In Fig. 9 and 10 in the paper, the results of the notebook are shown. 



## References
The framework used by `GEDI` is taken directly from the original paper by [...]. If you would like to discuss the paper, or corresponding research questions on benchmarking process mining tasks please email the authors.
