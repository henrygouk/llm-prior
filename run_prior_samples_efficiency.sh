#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name

base_url=${1:-http://localhost:8000/v1}
model=${2:-meta-llama/Meta-Llama-3-8B-Instruct}

mkdir -p results/diabetes/prior-samples-efficiency

for m in 5 10 20 40 80:
do
    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/diabetes.arff \
      --prior-cache prior_cache/diabetes.pkl \
      --prior-samples ${m} \
      --delta 0.5 \
      | tee results/diabetes/prior-samples-efficiency/baseline_${m}_d0_5.csv

    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/diabetes.arff \
      --prior-cache prior_cache/diabetes.pkl \
      --prior-samples ${m} \
      | tee results/diabetes/prior-samples-efficiency/llm_${m}_d1_0.csv

    python evaluate.py \
      --base-url ${base_url} \
      --model ${model} \
      --data-path datasets/diabetes.arff \
      --prior-cache prior_cache/diabetes.pkl \
      --prior-samples ${m} \
      --delta 2.0 \
      | tee results/diabetes/prior-samples-efficiency/llm_${m}_d2_0.csv
done
