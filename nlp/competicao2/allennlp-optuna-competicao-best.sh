#!/usr/bin/env bash

CONFIG_DIR=tune
OUTPUT_DIR=$1
BATCH_SIZE=$2
EPOCHS=$3
BERT_SIZE=$4
WANDB_NAME=$5

export WANDB_API_KEY=a4f27ffc6ae02f2ad1302e2aacf0eba46dc082b8
export WANDB_WATCH=false
export BATCH_SIZE=${BATCH_SIZE}
export EPOCHS=${EPOCHS}
export BERT_SIZE=${BERT_SIZE}
export WANDB_NAME=${WANDB_NAME}
export TUNE=false

echo -n "dropout: "; read dropout;
echo -n "eps: "; read eps;
echo -n "grad_clipping: "; read grad_clipping;
echo -n "grad_norm: "; read grad_norm;
echo -n "gradient_accumulation_steps: "; read gradient_accumulation_steps;
echo -n "lr: "; read lr;
echo -n "lr_ner: "; read lr_ner;
echo -n "weight_decay: "; read weight_decay;
echo -n "weight_decay_ner: "; read weight_decay_ner;

export dropout=${dropout}
export eps=${eps}
export grad_clipping=${grad_clipping}
export grad_norm=${grad_norm}
export gradient_accumulation_steps=${gradient_accumulation_steps}
export lr=${lr}
export lr_ner=${lr_ner}
export weight_decay=${weight_decay}
export weight_decay_ner=${weight_decay_ner}

if [[ ! -d ${OUTPUT_DIR} ]]
then
    mkdir -p "${OUTPUT_DIR}"
else
    rm -rf "${OUTPUT_DIR}"/*
fi

echo "allennlp train "${CONFIG_DIR}"/optuna_competicao_pass_through_retrain.jsonnet \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer"

allennlp train "${CONFIG_DIR}"/optuna_competicao_pass_through_retrain.jsonnet \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer
