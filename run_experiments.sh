#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name. The third argument is the number of parallel jobs to run

base_url=${1:-http://localhost:8000/v1}
model=${2:-meta-llama/Meta-Llama-3-8B-Instruct}
N=${3:-3}

datasets=("diabetes" "glass" "iris" "survival" "vote")

run_on_dataset() {
    dataset=$1
    mkdir -p results/${dataset}
    mkdir -p prior_cache/

    # Run the baseline Bayesian logistic regression model with a standard Gaussian prior
    python evaluate.py \
      --data-path datasets/${dataset}.arff \
      | tee results/${dataset}/baseline.csv

    for gamma in 0.5
    do
        for delta in 0.5 1.0 2.0
        do
            # Run the Bayesian logistic regression model with priors elicited from LLMs
            python evaluate.py \
              --base-url ${base_url} \
              --model ${model} \
              --data-path datasets/${dataset}.arff \
              --prior-cache prior_cache/${dataset}.pkl \
              --prior-samples 120 \
              --gamma ${gamma} \
              --delta ${delta} \
              | tee results/${dataset}/llm_gamma-${gamma}_delta-${delta}.csv
        done
    done
}

for dataset in ${datasets[@]}
do
    run_on_dataset ${dataset} &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
      wait
    fi
done

wait
