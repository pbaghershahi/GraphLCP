#!/bin/bash

# Generate the unique Experiment ID
EXP_ID="BATCH_$(date +'%Y-%m-%d-%H-%M-%S')"
echo "Starting Experiment Batch: $EXP_ID"

# Select dataset configs and values that override them.
# datasets=("goodcbas" "goodwebkb" "goodtwitch" "anaheim" "chicagosketch" "twitchptbr" "cora_ml" "dblp" "pubmed" "citeseer" "education" "election" "unemployment" "income" "computers" "photo" "physics" "cs")
datasets=("anaheim")
conformal_eval_methods=("fsc" "wsc" "marginal")

for dataset in "${datasets[@]}"; do
    echo "Run: $dataset"
    python main.py \
        -cff "./config/$dataset.yaml" \
        -eid "$EXP_ID" \
        -cem "${conformal_eval_methods[@]}"
done
