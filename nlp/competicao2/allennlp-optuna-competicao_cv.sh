#!/usr/bin/env bash

CONFIG_DIR=tune
OUTPUT_DIR=$1
N_TRIALS=$2
BATCH_SIZE=$3
EPOCHS=$4

export BATCH_SIZE=${BATCH_SIZE}
export EPOCHS=${EPOCHS}

allennlp tune-cv "${CONFIG_DIR}"/optuna_competicao.jsonnet \
  "${CONFIG_DIR}"/search_space_competicao.json \
  --optuna-param-path "${CONFIG_DIR}"/optuna.json \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer \
  --study-name "competicao-ner-cv" \
  --n-trials ${N_TRIALS} \
  --skip-if-exists