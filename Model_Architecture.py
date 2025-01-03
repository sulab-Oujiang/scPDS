import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFeatureExtractor(nn.Module):
    """
    The TransformerFeatureExtractor class implements a sophisticated 
    feature extraction mechanism utilizing a Transformer architecture. 
    This module is engineered to process input data, propagate it through 
    multiple Transformer encoder layers, and subsequently produce both 
    the extracted features and the reconstructed input.
    
    Args:
        input_dim (int): The dimensionality of the input data.
        embedding_dim (int): The dimensionality of the embedding space.
        hidden_dim (int): The dimensionality of the feedforward network model in the Transformer.
        num_heads (int): The number of heads in the multi-head attention mechanism.
        num_layers (int): The number of Transformer encoder layers.
        dropout (float): The dropout rate to be applied in the Transformer layers.
    
    Methods:
        forward(x): Processes the input tensor x through the embedding layer, Transformer
                    encoder layers, and decoder. Returns the features extracted by the last
                    encoder layer and the reconstructed input.
    
    Examples:
        >>> extractor = TransformerFeatureExtractor(input_dim=1300, embedding_dim=512, hidden_dim=256, num_heads=4, num_layers=1, dropout=0.3)
        >>> features, reconstructed_input = extractor(input_tensor)
    """
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_heads, num_layers, dropout):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.transformer_encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
            nn.Sigmoid())
    
    def forward(self, x):
        linear1 = self.transformer_encoder.layers[-1].linear1
        activation = {}
        def hook(module, input, output):
            activation[linear1] = output
        handle = linear1.register_forward_hook(hook)
        if self.input_dim != self.embedding:
            x = self.embedding(x)
        x = self.transformer_encoder(x)
        handle.remove()
        features = activation[linear1]
        reconstructed_input = self.decoder(x)
        return features, reconstructed_input


class MLP_Predictor(nn.Module):
    """
    The MLP_Predictor class implements a multi-layer perceptron (MLP) for prediction tasks.
    This module consists of several fully connected layers, batch normalization layers,
    dropout layers, and activation functions designed to process input data and produce
    output predictions.
    
    Args:
        Bottleneck_dim (int): The dimensionality of the extracted features.
        input_dim (int): The dimensionality of the input layer.
        hidden_dim (list of int): A list containing the dimensionalities of the hidden layers.
        output_dim (int): The dimensionality of the output layer.
        dropout (float): The dropout rate applied to the dropout layers.
    
    Methods:
        forward(x): Defines the forward pass of the MLP. Takes an input tensor x, processes it through
                    the network layers, and returns the output predictions.
    
    Examples:
        >>> predictor = MLP_Predictor(Bottleneck_dim=256, input_dim=256, hidden_dim=[128, 64], output_dim=2, dropout=0.3)
        >>> predictions = predictor(input_tensor)
    """
    
    def __init__(self, Bottleneck_dim, input_dim, hidden_dim, output_dim, dropout):
        super(MLP_Predictor, self).__init__()
        self.fc = nn.Linear(Bottleneck_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


