local transformer_model = "neuralmind/bert-base-portuguese-cased";
local transformer_dim = 512;
local cuda_device = 0;

local env_or_default(env_name, default_value) =
    local env_value = std.extVar(env_name);
    if env_value == "" then default_value else env_value;

local batch_size = std.parseInt(env_or_default("BATCH_SIZE", "2"));
local epochs = std.parseInt(env_or_default("EPOCHS", "10"));

local seed = std.parseInt(std.extVar("seed"));
local scheduler = std.extVar("learning_rate_scheduler");

local slanted_triangular_scheduler = {
    "type": "slanted_triangular",
    "cut_frac": 0.06
};

local linear_with_warmup_scheduler = {
    "type": "linear_with_warmup",
    "warmup_steps": std.parseInt(std.extVar("warmup_steps"))
};

local learning_rate_scheduler = if scheduler == "slanted_triangular" then slanted_triangular_scheduler else linear_with_warmup_scheduler;

local weight_decay = std.parseJson(std.extVar("weight_decay"));
local learning_rate = std.parseJson(std.extVar("lr"));
local weight_decay_ner = std.parseJson(std.extVar("weight_decay_ner"));
local learning_rate_ner = std.parseJson(std.extVar("lr_ner"));
local eps = std.parseJson(std.extVar("eps"));
local grad_clipping = std.parseJson(std.extVar("grad_clipping"));
local grad_norm = std.parseJson(std.extVar("grad_norm"));

local fold = std.extVar("FOLD");
local path = if fold == "" then "/" else "/cv/fold-" + fold + "/";

local tokenizer_kwargs = {
        "max_len": transformer_dim
    };

local token_indexer = {
        "type": "pretrained_transformer_mismatched",
        "max_length": transformer_dim,
        "model_name": transformer_model,
        "tokenizer_kwargs": tokenizer_kwargs
    };

{
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "tag_label": "ner",
        "token_indexers": {
            "tokens": token_indexer
        }
    },
    "train_data_path": "/media/discoD/repositorios/deeplearningufg/nlp/competicao2" + path + "train.conll",
    "validation_data_path": "/media/discoD/repositorios/deeplearningufg/nlp/competicao2" + path + "dev.conll",
    "test_data_path": "/media/discoD/repositorios/deeplearningufg/nlp/competicao2" + path + "dev.conll",
    "evaluate_on_test": true,
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "max_length": transformer_dim,
                    "model_name": transformer_model,
                    "tokenizer_kwargs": {
                        "max_len": transformer_dim
                    }
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.5,
            "hidden_size": 200,
            "input_size": 768,
            "num_layers": 2
        },
        "regularizer": {
          "regexes": [
            [
                "scalar_parameters",
                {
                    "type": "l2",
                    "alpha": 0.1
                }
            ]
          ]
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : batch_size,
            "sorting_keys": [
                "tokens"
            ]
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": weight_decay,
            "lr": learning_rate,
            "eps": eps,
            "parameter_groups": [
                [
                    ["^text_field_embedder(?:\\.(?!(LayerNorm|bias))[^.]+)+$"],
                    {"weight_decay": weight_decay, "lr": learning_rate}
                ],
                [
                    ["^text_field_embedder\\.[\\S]+(LayerNorm[\\S]+|bias)$"],
                    {"weight_decay": 0, "lr": learning_rate}
                ],
                [
                    ["encoder._module", "tag_projection_layer", "crf"],
                    {"weight_decay": weight_decay_ner, "lr": learning_rate_ner}
                ]
            ]
        },
        callbacks: [
            {
                type: 'optuna_pruner'
            }
        ],
        "learning_rate_scheduler": learning_rate_scheduler,
        "grad_norm": grad_norm,
        "grad_clipping": grad_clipping,
        "cuda_device": cuda_device,
        "num_epochs": epochs,
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "patience": 5,
        "validation_metric": "+f1-measure-overall"
    }
}