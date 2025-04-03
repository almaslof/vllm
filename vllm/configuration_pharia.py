from transformers.configuration_utils import PretrainedConfig


class PhariaConfig(PretrainedConfig):
    model_type = "pharia-v1"
    
    def __init__(
        self,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        #hidden_act="gelu",
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=2048,
        max_position_embeddings=8192,
        model_type="pharia-v1",
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        torch_dtype="bfloat16",
        transformers_version="4.31.0.dev0",
        use_cache=True,
        vocab_size=128000,
        mlp_bias=True,
        attention_bias=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        rope_theta=1000000,  # rotary_embeddingbase,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
