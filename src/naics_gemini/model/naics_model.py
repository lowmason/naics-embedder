import torch
import torch.nn as nn
from .components import TransformerEncoderLayer

class NaicsModel(nn.Module):
    """
    A Transformer-based model for NAICS code classification.
    """
    def __init__(self, vocab_size, num_classes, d_model=512, n_heads=8, 
                 d_ff=2048, num_layers=6, max_seq_length=512, dropout=0.1):
        """
        Initializes the model.
        
        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            num_classes (int): The number of output classes (NAICS codes).
            d_model (int): The dimensionality of the embedding and model.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the feed-forward layer.
            num_layers (int): The number of Transformer encoder layers.
            max_seq_length (int): The maximum sequence length for positional encoding.
            dropout (float): The dropout rate.
        """
        super(NaicsModel, self).__init__()
        
        self.d_model = d_model
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding
        self.positional_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification Head
        # The output layer maps the final transformer output to the number of classes.
        self.classifier_head = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        """Initializes weights for the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Init classifier head biases, e.g. for focal loss if needed,
        # but for standard CE, zero-init is fine.
        if self.classifier_head.bias is not None:
             nn.init.zeros_(self.classifier_head.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tensor of input token IDs. 
                                     Shape: (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): Tensor indicating padding tokens.
                                                     Shape: (batch_size, seq_length)

        Returns:
            torch.Tensor: Logits for each class. Shape: (batch_size, num_classes)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create positional IDs (0, 1, 2, ..., seq_length-1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get token and positional embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.positional_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Create attention mask for the transformer
        # (batch_size, 1, 1, seq_length)
        if attention_mask is not None:
            # Transformer expects mask: 0 for tokens to attend to, 1 for padding
            # But our components.py expects: 1 for tokens to attend to, 0 for padding
            # Let's adjust for the components.py implementation
            # It expects (batch_size, 1, 1, seq_length) for broadcasting
            transformer_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            transformer_mask = None

        # Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=transformer_mask)
            
        # Use the [CLS] token representation for classification
        # Assuming the [CLS] token is always at the beginning (index 0)
        cls_token_output = x[:, 0, :]
        
        # Apply dropout
        cls_token_output = self.dropout(cls_token_output)
        
        # Pass through the classification head
        logits = self.classifier_head(cls_token_output)
        
        return logits
