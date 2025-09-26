import torch
import torch.nn as nn
import numpy as np

class DecoderBlock(nn.Module):
    '''
    Class functioning as an decoder block following
    figure 4 in Project 3 description.

    Parameters:
    - dim (Int): Number of dimensions
    - num_heads (Int): Number of heads
    - dropout (Float): Dropout constant
    '''
    def __init__(self, embed_size, num_heads, dropout):
        ################################################################################
        # Inspiration from own implementation of encoder block, tips for section 2.1.2 #
        # and the PyTorch documentation for MultiheadAttention:                        #
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html   #
        ################################################################################
        super().__init__()

        # Initalizing values
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

        # The dimensionality should be divisble by number of heads
        assert embed_size % num_heads == 0, f"Dimension {embed_size} must be divisible by num_heads {num_heads}"

        # Layer normalization
        self.layer_normalization_1 = nn.LayerNorm(embed_size)
        self.layer_normalization_2 = nn.LayerNorm(embed_size)

        # Dropout
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        # Defining the MLP block
        self.MLP_block = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size)
        )

    def forward(self, x, attn_mask, padding_mask):
        '''Perform one forward pass through decoder'''
        # Part 1: Multihead Attention
        residual_1 = x
        x = self.layer_normalization_1(x)
        x, _ = self.multihead_attention(x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask, need_weights=False, is_causal=True)
        x = self.dropout_1(x)
        x += residual_1

        # Part 2: Feed forward through MLP
        x = self.layer_normalization_2(x)
        residual_2 = x
        x = self.MLP_block(x)
        x = self.dropout_2(x)
        x += residual_2

        return x




class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        ####################################################
        # Inspiration from Attention Is All You Need paper #
        # and some help for implementation from ChatGPT    # 
        ####################################################

        # Initialize tensor to hold positional encodings:
        positional_enc = torch.ones(max_len, embed_size)

        # Arange numbers for each position, and make 2d so 
        # pytorch can multiply
        positions = torch.arange(0, max_len).unsqueeze(1)

        # Arange the term the position is going to be divided by
        division_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        
        # Sine for even indices (positions times the division term yields the pos/10000^...)
        positional_enc[:, 0::2] = torch.sin(positions * division_term)

        # Cosine for odd indices (--||--)
        positional_enc[:, 1::2] = torch.cos(positions * division_term)

        # Unsqueeze to make it compatible with batches
        self.register_buffer('positional_enc', positional_enc.unsqueeze(0))

    def forward(self, x):
        # Forward pass, apply same encoding to each batch
        positional_enc = self.positional_enc.to(x.device) # From tips for section 2.1.2
        return x + positional_enc[:, :x.size(1), :]




class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        triangular_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)  # From tips for section 2.1.2
        triangular_mask = triangular_mask == 1 # Convert from 1 and 0 to True and False
        return triangular_mask




if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

