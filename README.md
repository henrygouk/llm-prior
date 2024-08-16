# Eliciting Bayesian Logistic Regression Priors with LLMs
This repository contains an implementation for Bayesian logistic regression, where priors are elicited from LLMs.

## Requirements

The code is written in Python 3.10. To install the required packages, run:
```
$ pip install -r requirements.txt
```

## Usage

A vLLM server is used to generate the dataset-specific priors for the Bayesian logistic regression. These priors are cached in `prior_cache`.

All the experiments can be run using a single bash script. To run the experiments, execute the following command:
```
$ ./run_experiments.sh [base_url] [model_repo]
```

Where the optional arguments are as follows:
- `base_url` is the base URL for vLLM API. E.g., `http://localhost:8000/v1`.
- `model_repo` is the name of the model repository on hugging face. E.g., `meta-llama/Meta-Llama-3-8B-Instruct`.

These arguments can be ommitted if the priors are already cached in `prior_cache`.

The script will run the experiments for the following datasets:
- `diabetes`
- `glass`
- `iris`
- `survival`
- `vote`

These datasets are stored in a modified ARFF file format in the `datasets` directory. The results will be saved in the `results` directory.

## Reference
If you use this code in an academic context, please cite the following paper:

```
@inproceedings{gouk2024automated,
  title={Automated Prior Elicitation from Large Language Models for Bayesian Logistic Regression},
  author={Gouk, Henry and Gao, Boyan},
  booktitle={AutoML Conference 2024 (Workshop Track)},
  year={2024}
}
```
