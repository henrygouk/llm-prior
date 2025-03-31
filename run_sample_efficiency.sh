#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name. The third argument is the number of parallel jobs to run

base_url=${1:-http://localhost:8000/v1}
model=${2:-meta-llama/Meta-Llama-3.1-8B-Instruct}
N=${3:-4}
runID=${4:-`date +%Y-%m-%d_%H-%M-%S`_se}

# Run on all datasets in the ./datasets/ directory
datasets=`find ./datasets/ -maxdepth 1 -type f -name "*.arff" -exec basename {} \; | sed 's/.arff//g'`

function prepend() {
    while read line;
    do
        echo "[${1}] ${line}"
    done
}

function run_baseline_blr() {
    dataset=$1
    mkdir -p exp/${runID}/results/baseline/${dataset}
    mkdir -p exp/${runID}/logs/baseline/${dataset}

    echo "Started running Bayesian Logistic Regression on ${dataset}..."

    python exp_fsl.py \
        --data-path datasets/${dataset}.arff \
        --samples 4 8 16 32 64 128 \
        --eval-method holdout \
        --model blr \
        2> exp/${runID}/logs/baseline/${dataset}/blr.log \
        | tee exp/${runID}/results/baseline/${dataset}/blr.csv | prepend "${dataset}"

    echo "Finished running Bayesian Logistic Regression on ${dataset}..."
}

function run_baseline_bnn() {
    dataset=$1
    mkdir -p exp/${runID}/results/baseline/${dataset}
    mkdir -p exp/${runID}/logs/baseline/${dataset}

    echo "Started running Bayesian Neural Network on ${dataset}..."

    python exp_fsl.py \
        --data-path datasets/${dataset}.arff \
        --samples 4 8 16 32 64 128 \
        --eval-method holdout \
        --model bnn \
        2> exp/${runID}/logs/baseline/${dataset}/bnn.log \
        | tee exp/${runID}/results/baseline/${dataset}/bnn.csv | prepend "${dataset}"

    echo "Finished running Bayesian Neural Network on ${dataset}..."
}

function run_llm_blr() {
    mkdir -p exp/${runID}/results/${model}/${dataset}
    mkdir -p exp/${runID}/logs/${model}/${dataset}
    mkdir -p prior_cache/${model}

    echo "Started running Bayesian Logisitic Regression with LLM (${model}) Prior on ${dataset}..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples 128 \
        --eval-method holdout \
        --model blr \
        2> exp/${runID}/logs/${model}/${dataset}/blr.log \
        | tee exp/${runID}/results/${model}/${dataset}/blr.csv | prepend "${dataset}"

    echo "Finished running Bayesian Logisitic Regression with LLM (${model}) Prior on ${dataset}..."
}

function run_llm_bnn() {
    mkdir -p exp/${runID}/results/${model}/${dataset}
    mkdir -p exp/${runID}/logs/${model}/${dataset}
    mkdir -p prior_cache/${model}

    echo "Started running Bayesian Neural Network with LLM (${model}) Prior on ${dataset}..."

    python exp_fsl.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}.pkl \
        --samples 0 4 8 16 32 64 128 \
        --prior-samples 128 \
        --eval-method holdout \
        --model bnn \
        2> exp/${runID}/logs/${model}/${dataset}/bnn.log \
        | tee exp/${runID}/results/${model}/${dataset}/bnn.csv | prepend "${dataset}"

    echo "Finished running Bayesian Neural Network with LLM (${model}) Prior on ${dataset}..."
}

function run_on_dataset() {
    dataset=$1

    run_baseline_blr ${dataset}
    run_baseline_bnn ${dataset}
    run_llm_blr ${dataset}
    run_llm_bnn ${dataset}
}

for dataset in ${datasets[@]}
do
    run_on_dataset ${dataset} &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait
    fi
done

wait
