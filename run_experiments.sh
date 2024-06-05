#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name

base_url=${1:-http://localhost:8000/v1}
model=${2:-meta-llama/Meta-Llama-3-8B-Instruct}

datasets=("diabetes" "glass" "iris" "survival" "vote")

for dataset in ${datasets[@]}
do
    mkdir -p results/${dataset}
    mkdir -p prior_cache/

    # Run the baseline Bayesian logistic regression model with a standard Gaussian prior
    python evaluate.py \
      --data-path datasets/${dataset}.arff \
      | tee results/${dataset}/baseline.csv

    # Run the Bayesian logistic regression model with priors elicited from LLMs
    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/${dataset}.arff \
      --prior-cache prior_cache/${dataset}.pkl \
      --prior-samples 80 \
      | tee results/${dataset}/llm.csv

    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/${dataset}.arff \
      --prior-cache prior_cache/${dataset}.pkl \
      --prior-samples 80 \
      --delta 2.0 \
      | tee results/${dataset}/llm_d2_0.csv

    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/${dataset}.arff \
      --prior-cache prior_cache/${dataset}.pkl \
      --prior-samples 80 \
      --delta 0.5 \
      | tee results/${dataset}/llm_d0_5.csv
done
