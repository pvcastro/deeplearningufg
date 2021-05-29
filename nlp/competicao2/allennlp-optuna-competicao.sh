#!/usr/bin/env bash

CONFIG_DIR=tune
OUTPUT_DIR=$1
STUDY_NAME="competicao-ner-pt-ngas4-gc1"
N_TRIALS=$2
BATCH_SIZE=$3
EPOCHS=$4

export BATCH_SIZE=${BATCH_SIZE}
export EPOCHS=${EPOCHS}
export BERT_SIZE="base"
export TUNE=true

export grad_clipping=1
export gradient_accumulation_steps=4

allennlp tune "${CONFIG_DIR}"/optuna_competicao_pass_through.jsonnet \
  "${CONFIG_DIR}"/search_space_pass_through_2.json \
  --optuna-param-path "${CONFIG_DIR}"/optuna.json \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer \
  --metrics best_validation_f1-measure-overall \
  --study-name "${STUDY_NAME}" \
  --direction maximize \
  --n-trials ${N_TRIALS} \
  --skip-if-exists