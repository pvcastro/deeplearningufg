#!/usr/bin/env bash

CONFIG_DIR=tune
OUTPUT_DIR=$1
STORAGE_PATH=$2
STUDY_NAME=$3
BATCH_SIZE=$4
EPOCHS=$5

export WANDB_API_KEY=a4f27ffc6ae02f2ad1302e2aacf0eba46dc082b8
export WANDB_WATCH=false
export BATCH_SIZE=${BATCH_SIZE}
export EPOCHS=${EPOCHS}
export FOLD=

if [[ ! -d ${OUTPUT_DIR} ]]
then
    mkdir -p "${OUTPUT_DIR}"
else
    rm -rf "${OUTPUT_DIR}"/*
fi

echo "allennlp retrain "${CONFIG_DIR}"/optuna_competicao_retrain.jsonnet \
  --storage "${STORAGE_PATH}" \
  --study-name "${STUDY_NAME}" \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer"

allennlp retrain "${CONFIG_DIR}"/optuna_competicao_retrain.jsonnet \
  --storage "${STORAGE_PATH}" \
  --study-name "${STUDY_NAME}" \
  --serialization-dir "${OUTPUT_DIR}" \
  --include-package allennlp_datalawyer