#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name. The third argument is the number of parallel jobs to run

dataset=$1
base_url=${2:-http://localhost:8000/v1}
model=${3:-meta-llama/Meta-Llama-3.1-8B-Instruct}
N=${4:-4}
runID=${5:-`date +%Y-%m-%d_%H-%M-%S`_pke_${dataset}}

function prepend() {
    while read line;
    do
        echo "[${1}] ${line}"
    done
}

function run_blr() {
    echo "Started running Bayesian Logisitic Regression with ${prior_samples} prior samples..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples ${prior_samples} \
        --eval-method holdout \
        --model blr \
        --no-header \
        --ho-reps 10 \
        2>> exp/${runID}/logs/${model}/${dataset}/blr.log \
        | tee -a exp/${runID}/results/${model}/${dataset}/blr.csv | prepend "${dataset} - ${prior_samples}"

    echo "Finished running Bayesian Logisitic Regression with ${prior_samples} prior samples..."
}

function run_bnn() {
    echo "Started running Bayesian Neural Network with ${prior_samples} prior samples..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples ${prior_samples} \
        --eval-method holdout \
        --model bnn \
        --no-header \
        --ho-reps 10 \
        2>> exp/${runID}/logs/${model}/${dataset}/bnn.log \
        | tee -a exp/${runID}/results/${model}/${dataset}/bnn.csv | prepend "${dataset} - ${prior_samples}"

    echo "Finished running Bayesian Neural Network with ${prior_samples} prior samples..."
}

mkdir -p exp/${runID}/results/${model}/${dataset}
mkdir -p exp/${runID}/logs/${model}/${dataset}
mkdir -p prior_cache/${model}

echo "rep,num_prior,num_train,roc_auc" > exp/${runID}/results/${model}/${dataset}/blr.csv
echo "rep,num_prior,num_train,roc_auc" > exp/${runID}/results/${model}/${dataset}/bnn.csv

for prior_samples in 32 64; do
    if (( $(jobs -r | wc -l) >= N )); then
        wait -n
    fi
    
    run_blr &
done

for prior_samples in 4 8 16 32 64; do
    if (( $(jobs -r | wc -l) >= N )); then
        wait -n
    fi

    run_bnn &
done

wait