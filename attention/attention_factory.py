# attention_factory.py
from attention.grouped_attention import GroupedSelfAttention
from attention.retention import MultiScaleRetention
from attention.scaled_dot_product_attention import SelfAttention

def create_attention(config):
    attention_type = config.attention_type
    if attention_type == 'causal':
        return SelfAttention(config)
    elif attention_type == 'grouped':
        return GroupedSelfAttention(config)
    elif attention_type == 'retention':
        return MultiScaleRetention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
