import numpy as np
import torch.nn as nn
import torch
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TimeSeriesNormalizer:
    def __init__(self, epsilon=1e-5):
        """
        Initialize the normalizer with a small constant to avoid division by zero.

        Args:
        epsilon (float): A small constant to ensure numerical stability.
        """
        self.epsilon = epsilon

    def normalize(self, x):
        """
        Normalize the input time series along a specified dimension.

        Args:
        x (torch.Tensor): The input tensor to normalize. Assumes that the time dimension is at index 2.

        Returns:
        tuple: A tuple containing the normalized tensor and the means and standard deviations used for normalization.
        """
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + self.epsilon).detach()
        x /= stdev

        return x, means, stdev

    def denormalize(self, x_normalized, means, stdev):
        """
        Denormalize the input time series using the provided means and standard deviations.

        Args:
        x_normalized (torch.Tensor): The normalized tensor to be denormalized.
        means (torch.Tensor): The means used during the normalization process.
        stdev (torch.Tensor): The standard deviations used during the normalization process.

        Returns:
        torch.Tensor: The denormalized tensor.
        """
        x_denormalized = x_normalized * stdev + means
        return x_denormalized

class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        # Norm and Add
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


def build_transformer(d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:


    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the transformer
    transformer = Transformer(encoder, decoder)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


class ExtractPatches(nn.Module):
    def __init__(self, patch_size, stride_len):
        super().__init__()
        self.patch_size = patch_size
        self.stride_len = stride_len

    def forward(self, x):
        # (batch, channel, past_steps) -->  (batch, channel, patch_size, num_patches)

        # patches = x.unfold(2, self.patch_size, self.stride_len).permute(0, 1, 3, 2)

        num_patches = int(np.floor(((x.shape[-1] - self.patch_size) / self.stride_len)) + 2)
        # Repeat the last slice of the time series to handle edge cases
        last_slice_repeated = x[:, :, -1].unsqueeze(-1).repeat(1, 1, self.stride_len)

        # Concatenate the original time series with the repeated last slice
        time_series_extended = torch.cat((x, last_slice_repeated), dim=2)

        patches = []

        # Extract patches using a sliding window approach
        for i in range(num_patches):
            start_index = i * self.stride_len
            end_index = start_index + self.patch_size
            patch = time_series_extended[:, :, start_index:end_index]
            patches.append(patch)

        # Stack all patches into a tensor
        patches_tensor = torch.stack(patches, dim=-1)
        return patches_tensor


class LinearProjectionLayer(nn.Module):

    def __init__(self, patch_size, d_model) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model, bias=False)

    def forward(self, x) -> None:
        # (batch, channel, patch_size, number_patch) --> (batch, channel, d_model, number_patch)
        B, M, P, N = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, M, N, P)
        x = self.proj(x)
        return x.permute(0, 1, 3, 2)



class LinearHead(nn.Module):

        def __init__(self, d_model: int, num_patch: int, pred_size: int) -> None:
            super().__init__()
            self.linearhead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model*num_patch, pred_size)
            )


        def forward(self, x):
            # (batch, seq_len, d_model) --> (batch, seq_len* d_model) --> (batch, pred_size)
            return self.linearhead(x)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class FutureTST(nn.Module):
    def __init__(self,context_window_size=365, patch_size=16, stride_len=8, d_model=256,
                 num_transformer_layers=2, mlp_size=128, num_heads=8, mlp_dropout=0.2,
                 pred_size=20, embedding_dropout=0.1,input_channels=0):
        super().__init__()


        self.patch_size = patch_size
        self.stride_len = stride_len
        self.pred_size = pred_size
        self.context_window_size = context_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.embedding_dropout = embedding_dropout
        self.input_channels = input_channels

        self.num_endoPatch = int(np.floor((self.context_window_size - self.patch_size) / stride_len) + 2)
        # self.num_exoPatch = int(np.floor((self.context_window_size+self.pred_size - patch_size) / stride_len) + 2)
        self.normalizer = TimeSeriesNormalizer()
        self.extract_patches = ExtractPatches(self.patch_size, self.stride_len)
        self.exogeneous_feature_projection = nn.Linear(self.context_window_size+self.pred_size, self.d_model)
        self.linear_projection_layer = LinearProjectionLayer(self.patch_size, self.d_model)
        self.endo_positional_encoding = PositionalEncoding(self.d_model, self.num_endoPatch, self.embedding_dropout)
        self.exo_positional_encoding = PositionalEncoding(self.d_model,  self.input_channels - 1 , self.embedding_dropout)
        self.transformer = build_transformer(d_model=self.d_model, N=self.num_transformer_layers, h=self.num_heads, dropout=self.mlp_dropout, d_ff=self.mlp_size)

        # Linear layer to project transformer decoder output to the prediction size
        self.linear_head = LinearHead(self.d_model, self.num_endoPatch, self.pred_size)



    def forward(self, x : torch.Tensor):
        # (batch, channel, past_steps+future_steps) -->  (batch, until_last_channel, past_steps+future_steps) and (batch, last_channel, past_steps)

        # sigma = 0.1  # Standard deviation of the Gaussian noise
        # noise = torch.randn(x[:, :-1, :].shape,device=x.device) * sigma
        # exogeneous_data = x[:, :-1, :] + noise # Add noise to the data

        exogeneous_data = x[:, :-1, :]

        endogeneous_data =  torch.unsqueeze(x[:, -1, :self.context_window_size], axis=1)


        endogeneous_data, means, stdev = self.normalizer.normalize(endogeneous_data)

        #print (f'Shape of Exogenous Input is {exogeneous_data.shape} and endogenous Input {endogeneous_data.shape}')

        # Extract patches from the endogeneous & exogeneeous data  (batch, channel, patch_size, num_patches)
        endogeneous_patches = self.extract_patches(endogeneous_data)
        # exogeneous_patches = self.extract_patches(exogeneous_data)

        #print (f'Shape of Patches are {exogeneous_data.shape}, {endogeneous_patches.shape}')

        # Project the patches to the dimension expected by the transformer decoder
        # Endo: (batch, channel, d_model, num_patches)
        endogeneous_patches = self.linear_projection_layer(endogeneous_patches)
        # Exo: (B,channel,past_steps+future_steps) -> (B, channel, d_model) --> (B, 1, channel, d_model) --> (B, 1, d_model, channel)


        exogeneous_patches = self.exogeneous_feature_projection(exogeneous_data).unsqueeze(1).permute(0, 1, 3, 2)

        #print (f'Shape of Projected Patches are {exogeneous_patches.shape}, {endogeneous_patches.shape}')

        # Apply positional encoding to the patches (B, channel, d_model, num_patches) --> (B*channel, num_patches, d_model)
        endogeneous_patches = self.endo_positional_encoding(endogeneous_patches.view(-1, self.d_model, self.num_endoPatch).permute(0, 2, 1))
        exogeneous_patches = self.exo_positional_encoding(exogeneous_patches.view(-1, self.d_model, self.input_channels - 1).permute(0, 2, 1))

        #print (f'Shape of Positional Encoded Patches are {exogeneous_patches.shape}, {endogeneous_patches.shape}')

        encoder_output = self.transformer.encode(exogeneous_patches, None) # (B, seq_len, d_model)
        #print (f'Shape of Encoder Output is {encoder_output.shape}')
        #print(f'Shape of Endogenous Patches is {endogeneous_patches.shape}')
        decoder_output = self.transformer.decode(encoder_output, None, endogeneous_patches, None) # (B, seq_len, d_model)

        #print (f'Shape of Decoder Output is {decoder_output.shape}')
        # Project the transformer decoder output to the prediction size

        predictions = self.linear_head(decoder_output).unsqueeze(1)

        predictions = self.normalizer.denormalize(predictions, means, stdev)

        return predictions