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
    "type": "bcn",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": 768,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 3],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
