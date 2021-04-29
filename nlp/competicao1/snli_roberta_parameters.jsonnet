local transformer_model = "verissimomanoel/RobertaTwitterBR";
local transformer_dim = 768;

{
  "dataset_reader":{
    "type": "text_classification_json",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": "train.json",
  "validation_data_path": "test.json",
  "test_data_path": "test.json",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
    "trainer": {
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "cuda_device": 0,
        "grad_clipping": 1.0,
        "grad_norm": 2.0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.06
        },
        "num_epochs": 10,
        "optimizer": {
            "type": "huggingface_adamw",
            "eps": 1e-06,
            "lr": 2e-05,
            "parameter_groups": [
                [
                    [
                        "^text_field_embedder(?:\\.(?!(LayerNorm|bias))[^.]+)+$"
                    ],
                    {
                        "lr": 1.664381848755544e-05,
                        "weight_decay": 0.1
                    }
                ],
                [
                    [
                        "^text_field_embedder\\.[\\S]+(LayerNorm[\\S]+|bias)$"
                    ],
                    {
                        "weight_decay": 0
                    }
                ],
                [
                    [
                        "feedforward",
                        "classification_layer"
                    ],
                    {
                        "lr": 0.0007472487670611192,
                        "weight_decay": 0
                    }
                ]
            ],
            "weight_decay": 0.1
        },
        "patience": 5,
        "validation_metric": "+accuracy",
        "callbacks": ["tensorboard"]
    },
}
