local transformer_model = "neuralmind/bert-base-portuguese-cased";
local transformer_dim = 512;
local transformer_emb_dim = 768;

local cuda_device = 0;

local env_or_default(env_name, default_value) =
    local env_value = std.extVar(env_name);
    if env_value == "" then default_value else env_value;

local batch_size = 6;
local epochs = 10;

local scheduler = std.extVar("learning_rate_scheduler");
//local scheduler = "slanted_triangular";

local slanted_triangular_scheduler = {
    "type": "slanted_triangular",
    "cut_frac": 0.06
};

local linear_with_warmup_scheduler = {
    "type": "linear_with_warmup",
    "warmup_steps": std.parseInt(std.extVar("warmup_steps"))
};

local learning_rate_scheduler = if scheduler == "slanted_triangular" then slanted_triangular_scheduler else linear_with_warmup_scheduler;

//local encoder_type = std.extVar("learning_rate_scheduler");
local encoder_type = "pt_encoder";

local lstm_encoder = {
    "type": "lstm",
    "bidirectional": true,
    "dropout": 0.5,
    "hidden_size": 200,
    "input_size": transformer_emb_dim,
    "num_layers": 2
};

local pt_encoder = {
    "type": "pass_through",
    "input_dim": transformer_emb_dim
};

local encoder = if encoder_type == "pt_encoder" then pt_encoder else lstm_encoder;

local weight_decay = std.parseJson(std.extVar("weight_decay"));
local learning_rate = std.parseJson(std.extVar("lr"));
local eps = std.parseJson(std.extVar("eps"));
local weight_decay_ner = std.parseJson(std.extVar("weight_decay_ner"));
local learning_rate_ner = std.parseJson(std.extVar("lr_ner"));
local grad_clipping = std.parseJson(std.extVar("grad_clipping"));
local grad_norm = std.parseJson(std.extVar("grad_norm"));
//local weight_decay = 0.10;
//local learning_rate = 4e-5;
//local eps = 1e-8;
//local weight_decay_ner = 0.01;
//local learning_rate_ner = 5e-4;
//local grad_clipping = 1.0;
//local grad_norm = 1.0;

local wandb_name = "competicao-ner-best-base-full-final";
local fold = "";
local path = "/";

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
    //"train_data_path": "/home/repositorios/deeplearningufg/nlp/competicao2/full_conll2003.conll",
    //"validation_data_path": ".." + path + "dev.conll",
    //"test_data_path": ".." + path + "dev.conll",
    //"train_data_path": "/media/discoD/repositorios/deeplearningufg/nlp/competicao2/full_conll2003.conll",
    //"test_data_path": "/media/discoD/repositorios/deeplearningufg/nlp/competicao2/test_oficial_conll2003.conll",
    "train_data_path": "/home/repositorios/deeplearningufg/nlp/competicao2/train.conll",
    "validation_data_path": "/home/repositorios/deeplearningufg/nlp/competicao2/dev.conll",
    "test_data_path": "/home/repositorios/deeplearningufg/nlp/competicao2/dev.conll",
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
        "encoder": encoder,
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
        "callbacks": [
            {
                "type": "wandb",
                "name": wandb_name,
                "summary_interval": 1,
                "batch_size_interval": 1,
                "should_log_learning_rate": true,
                "should_log_parameter_statistics": true,
                "project": "competicao-ner",
            },
            {
                "type": "tensorboard",
                "should_log_learning_rate": true
            }
        ],
        "learning_rate_scheduler": learning_rate_scheduler,
        "grad_norm": grad_norm,
        "grad_clipping": grad_clipping,
        "num_gradient_accumulation_steps": 4,
        "cuda_device": cuda_device,
        "num_epochs": epochs,
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "patience": 5,
        "validation_metric": "+f1-measure-overall"
    }
}