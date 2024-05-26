import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *
import loralib as lora


class BertSelfAttention(nn.Module):
    def __init__(self, config, rank):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = lora.Linear(config.hidden_size, self.all_head_size, r=rank)  # LoRA
        self.key = lora.Linear(config.hidden_size, self.all_head_size, r=rank)  # LoRA
        self.value = lora.Linear(config.hidden_size, self.all_head_size, r=rank)  # LoRA
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
        # Attention scores are calculated by multiplying the key and query to obtain
        # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
        # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
        # token, given by i-th attention head.
        # Before normalizing the scores, use the attention mask to mask out the padding token scores.
        # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
        # and padding tokens (with a value of a large negative number).

        # Make sure to:
        # - Normalize the scores with softmax.
        # - Multiply the attention scores with the value to get back weighted values.
        # - Before returning, concatenate multi-heads to recover the original shape:
        #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

        ### TODO
        key_T = torch.transpose(key, -1, -2)
        attention_QK = (torch.matmul(query, key_T) / math.sqrt(self.attention_head_size)) + attention_mask
        norm_attention = F.softmax(attention_QK, dim=-1)
        weighted_value = torch.matmul(norm_attention, value)

        weighted_value = torch.permute(weighted_value, (0, 2, 1, 3))
        weighted_value = weighted_value.reshape(norm_attention.size(0), norm_attention.size(-1), -1)

        return weighted_value

    def forward(self, hidden_states, attention_mask):
        """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # Calculate the multi-head attention.
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        # Multi-head attention.
        self.self_attention = BertSelfAttention(config, rank)
        # Add-norm for multi-head attention.
        self.attention_dense = lora.Linear(config.hidden_size, config.hidden_size, r=rank)  # LoRA
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = lora.Linear(config.hidden_size, config.intermediate_size, r=rank)  # LoRA
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = lora.Linear(config.intermediate_size, config.hidden_size, r=rank)  # LoRA
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
        # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
        # before it is added to the sub-layer input and normalized with a layer norm.
        ### TODO
        dense = dense_layer(output)
        dropout_dense = dropout(dense)
        result = ln_layer((input + dropout_dense))
        return result

    def forward(self, hidden_states, attention_mask):
        """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
        ### TODO

        mh_attention_out = self.self_attention(hidden_states, attention_mask)
        add_norm_out = self.add_norm(hidden_states, mh_attention_out, self.attention_dense, self.attention_dropout,
                                     self.attention_layer_norm)
        forward_out = self.interm_af(self.interm_dense(add_norm_out))
        out = self.add_norm(add_norm_out, forward_out, self.out_dense, self.out_dropout,
                            self.out_layer_norm)

        return out


class BertModel_LORA(BertPreTrainedModel):
    """
  The BERT model returns the final embeddings for each token in a sentence.

  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """

    def __init__(self, config, rank):
        super().__init__(config)
        self.config = config

        # Embedding layers.
        self.word_embedding = lora.Embedding(config.vocab_size, config.hidden_size, r=rank,
                                             padding_idx=config.pad_token_id)  # LoRA
        self.pos_embedding = lora.Embedding(config.max_position_embeddings, config.hidden_size, r=rank)  # LoRA
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # BERT encoder.
        self.bert_layers = nn.ModuleList([BertLayer(config, rank=rank) for _ in range(config.num_hidden_layers)])

        # [CLS] token transformations.
        self.pooler_dense = lora.Linear(config.hidden_size, config.hidden_size, r=rank)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = None
        ### TODO
        inputs_embeds = self.word_embedding(input_ids)

        # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = None
        ### TODO
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids. Since we are not considering token type, this embedding is
        # just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        ### TODO
        added_embed = inputs_embeds + pos_embeds + tk_type_embeds
        norm_out = self.embed_layer_norm(added_embed)
        out = self.embed_dropout(norm_out)

        return out

    def encode(self, hidden_states, attention_mask):
        """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
        # Get the extended attention mask for self-attention.
        # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
        # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
        # (with a value of a large negative number).
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        # Pass the hidden states through the encoder layers.
        for i, layer_module in enumerate(self.bert_layers):
            # Feed the encoding from the last bert_layer to the next.
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # Get cls token hidden state.
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}