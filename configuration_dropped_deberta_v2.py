from transformers import DebertaV2Config

class DebertaV2DroppedConfig(DebertaV2Config):
    r"""
    This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
    DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 128100):
            Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DebertaV2Model`].
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
            are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 0):
            The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        relative_attention (`bool`, *optional*, defaults to `True`):
            Whether use relative position encoding.
        max_relative_positions (`int`, *optional*, defaults to -1):
            The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
            as `max_position_embeddings`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `False`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (`List[str]`, *optional*):
            The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
            `["p2c", "c2p"]`, `["p2c", "c2p"]`.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        drop_mlp_list (`List[Union[int, bool]]`, *optional*):
            Use [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786) to drop mlp 
            layer in transformers layer. If is int in list, it will drop specify index mlp layer. If is `num_hidden_layers` 
            lenght bool list, it will drop index is True layer.
        drop_attn_list (`List[Union[int, bool]]`, *optional*):
            Use [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786) to drop attn 
            layer in transformers layer. If is int in list, it will drop specify index attn layer. If is `num_hidden_layers` 
            lenght bool list, it will drop index is True layer.
    """
    model_type = "deberta-v2-dropped"
    def __init__(self, vocab_size=128100, hidden_size=1536, num_hidden_layers=24,
                 num_attention_heads=24, intermediate_size=6144, hidden_act="gelu",
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                 type_vocab_size=0, initializer_range=0.02, layer_norm_eps=1e-7, relative_attention=False,
                 max_relative_positions=-1, pad_token_id=0, position_biased_input=True, pos_att_type=None,
                 pooler_dropout=0, pooler_hidden_act="gelu", drop_mlp_list: list[bool | int]=None, drop_attn_list: list[bool | int]=None, **kwargs):
        super().__init__(vocab_size, hidden_size, num_hidden_layers,
                         num_attention_heads, intermediate_size, hidden_act,
                         hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings,
                         type_vocab_size, initializer_range, layer_norm_eps, relative_attention,
                         max_relative_positions, pad_token_id, position_biased_input, pos_att_type,
                         pooler_dropout, pooler_hidden_act, **kwargs)
        # trans bool into int
        new_drop_attn_list = []
        if drop_attn_list is not None:
            for idx in range(len(drop_attn_list)):
                if isinstance(drop_attn_list[idx], bool):
                    if drop_attn_list[idx]:
                        new_drop_attn_list.append(idx)
                elif isinstance(drop_attn_list[idx], int):
                    new_drop_attn_list.append(drop_attn_list[idx])

        new_drop_mlp_list = []
        if drop_mlp_list is not None:
            for idx in range(len(drop_mlp_list)):
                if isinstance(drop_mlp_list[idx], bool):
                    if drop_mlp_list[idx] == True:
                        new_drop_mlp_list.append(idx)
                elif isinstance(drop_mlp_list[idx], int):
                    new_drop_mlp_list.append(drop_mlp_list[idx])

        if new_drop_mlp_list:
            self.drop_mlp_list: list[bool] = []
            for idx in range(self.num_hidden_layers):
                self.drop_mlp_list.append(True if idx in new_drop_mlp_list else False)
        else:
            self.drop_mlp_list = [False] * self.num_hidden_layers

        if new_drop_attn_list:
            self.drop_attn_list: list[bool] = []
            for idx in range(self.num_hidden_layers):
                self.drop_attn_list.append(True if idx in new_drop_attn_list else False)
        else:
            self.drop_attn_list = [False] * self.num_hidden_layers