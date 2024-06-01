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
## General Usage
Our pipeline offers several pipeline steps, which can be run sequentially or partially:
- [Feature Extraction](#feature-extraction)
- [Generation](#generation)
- [Benchmark](#benchmark)
- [Evaluation Plotter](#evaluation_plotter)

We also include two notebooks, which output experimental results as in our paper.

To run different steps of the GEDI pipeline, please adapt the `.json` accordingly.
```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/<pipeline-step>.json
```
For reference of possible keys and values for each step, please see `config_files/algorithm/experiment_test.json`.
To run the whole pipeline please create a new `.json` file, specifying all steps you want to run and specify desired keys and values for each step. 

### Feature Extraction
---
In order to extract the meta features being used for hyperparameter optimization, we employ the following script:
```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/feature_extraction.json
```
The json file consists of the following key-value pairs:

- pipeline_step: denotes the current step in the pipeline (here: feature_extraction)
- input_path: folder to the input files
- feature params: defines a dictionary, where the inner dictionary consists of a key-value pair 'feature_set' with a list of features being extracted from the references files. A list of valid features can be looked up from the FEEED extractor
- output_path: defines the path where plot are saved to
- real_eventlog_path: defines the file with the meta features extracted from the real event logs
- plot_type: defines the style of the output plotting (possible values: violinplot, boxplot)
- font_size: label font size of the output plot
- boxplot_widht: width of the violinplot/boxplot


### Generation
---
After having extracted meta features from the files, the next step is to generate event log data accordingly. Generally, there are two settings on how the targets are defined: i) meta feature targets are defined by the meta features from the real event log data; ii) a configuration space is defined which resembles the feasible meta features space. 

The command to execute the generation step is given by a exemplarily generation.json file:

```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/generation.json
```

In the `generation.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: event_logs_generation)
* output_path: defines the output folder
* generator_params: defines the configuration of the generator itself. For the generator itself, we can set values for the general 'experiment', 'config_space', 'n_trials', and a specific 'plot_reference_feature' being used for plotting

    - experiment: defines the path to the input file which contains the meta features which are used for the optimization step. The 'objectives' defines the specific meta features which are used as optimization criteria.
    - config_space: here, we define the configuration of the generator module (here: process tree generator). The process tree generator can process input information which defines characteristics for the generated data (a more thorough overview of the params can be found [here](https://github.com/tjouck/PTandLogGenerator):

        - mode: most frequent number of visible activities
        - sequence: probability to add a sequence operator to tree
        - choice: probability to add a choice operator to tree
        - parallel: probability to add a parallel operator to tree
        - loop: probability to add a loop operator to tree
        - silent: probability to add silent activity to a choice or loop operator
        - lt_dependency: probability to add a random dependency to the tree
        - num_traces: the number of traces in the event log
        - duplicate: probability to duplicate an activity label
        - or: probability to add an or operator to tree

    - n_trials: the maximum number of trials for the hyperparameter optimization to find a feasible solution to the specific configuration being used as target

    - plot_reference_feature: defines the feature which is used on the x-axis on the output plots, i.e., each feature defined in the 'objectives' of the 'experiment' is plotted against the reference feature being defined in this value


#### Supplementary: Generating data with real targets
In order to execute the experiments with real targets,we employ exemplarily the onfig file `config_files/algorithm/experiment_real_targets.json`. The script's pipeline will output the generated event logs with meta features values being optimized towards meta features of real-world benchmark datasets. Furthermore, it will output the respective feature values in the `\output`folder as well as the benchmark values.

```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/experiment_real_targets.json
```



### Benchmark
The benchmarking defines the downstream task which is used for evaluationg the goodness of the synthesized event log datasets with the metrics of real world datasets. The command to execute a benchmarking is shown in the following script:

```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/benchmark.json
```

In the `benchmark.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: benchmark_test)
* benchmark_test: defines the downstream task. Currently (in v 1.0), only 'discovery' for process discovery is implemented
* input_path: defines the input folder where the synthesized event log data are stored
* output_path: defines the output folder
* miners: defines the miners for the downstream task 'discovery' which are used in the benchmarking. In v 1.0 the miners 'inductive' for inductive miner, 'heuristics' for heuristics miner, 'imf' for inductive miner infrequent, as well as 'ilp' for integer linear programming are implemented


### Evaluation Plotting
The purpose of the evaluation plotting step is used just for visualization. Some examples of how the plotter can be used is shown in the following exemplarily script:


```console
conda activate gedi
python main.py -o config_files/options/baseline.json -a config_files/algorithm/evaluation_plotter.json
```

Generally, in the `evaluation_plotter.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: evaluation_plotter)
* input_path: defines the input file or the input folder which is considered for the visualizations. If a single file is specified, only the meta features in that file are considered whereas in the case of specifying a folder, the framework iterates over all files and use them for plotting
* plot_reference_feature: defines the feature which is used on the x-axis on the output plots, i.e., each feature defined in the input file is plotted against the reference feature being defined in this value
* targets: defines the target values which are also used as reference. Likewise to the input_path, the targets can be specified by single file or by a folder
* output_path: defines where to store the plots

## Further results plotting
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
