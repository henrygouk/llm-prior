#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name. The third argument is the number of parallel jobs to run

dataset=$1
base_url=${2:-http://localhost:8000/v1}
model=${3:-meta-llama/Meta-Llama-3.1-8B-Instruct}
N=${4:-4}
runID=${5:-`date +%Y-%m-%d_%H-%M-%S`_rf_${dataset}}

function prepend() {
    while read line;
    do
        echo "[${1}] ${line}"
    done
}

function run_blr() {
    echo "Started running Bayesian Logisitic Regression with random features..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}_rf.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples 128 \
        --eval-method holdout \
        --model blr \
        --use-rand-features \
        --ho-reps 10 \
        2>> exp/${runID}/logs/${model}/${dataset}/blr.log \
        | tee -a exp/${runID}/results/${model}/${dataset}/blr.csv | prepend "BLR"

    echo "Finished running Bayesian Logisitic Regression with random features..."
}

function run_bnn() {
    echo "Started running Bayesian Neural Network with random features..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}_rf.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples 128 \
        --eval-method holdout \
        --model bnn \
        --use-rand-features \
        --ho-reps 10 \
        2>> exp/${runID}/logs/${model}/${dataset}/bnn.log \
        | tee -a exp/${runID}/results/${model}/${dataset}/bnn.csv | prepend "BNN"

    echo "Finished running Bayesian Neural Network with random features..."
}

mkdir -p exp/${runID}/results/${model}/${dataset}
mkdir -p exp/${runID}/logs/${model}/${dataset}
mkdir -p prior_cache/${model}

run_blr
run_bnn
wait