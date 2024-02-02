import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 

class AutoEncoder(nn.Module):
    """
    input : encoder_dims

        encoder = [n_items, dim1, dim2]
        decoder_dims = [dim2, dim1, n_items]
    """
    
    def __init__(self, encoder_dims, decoder_dims=None): 
        super(AutoEncoder, self).__init__()
        self.encoder_dims = encoder_dims
        if decoder_dims:
            assert decoder_dims[0] == encoder_dims[-1], "In and Out dimensions must equal to each other"
            assert decoder_dims[-1] == encoder_dims[0], "Latent dimension for p- and q- network mismatches."
            self.decoder_dims = decoder_dims
        else:
            self.decoder_dims = encoder_dims[::-1]
        
        self.encoder = self.build_layers(self.encoder_dims)
        self.decoder = self.build_layers(self.decoder_dims)

    def build_layers(self, dims):
        """
        Helper function to build layers based on the provided dimensions.

        Parameters:
            - dims (list): List of dimensions for the layers.

        Returns:
            - nn.Sequential: Sequential container for the layers.
        """
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            layers.append(nn.ReLU())  

        return nn.Sequential(*layers)
        
    def forward(self, input):
        h = F.normalize(input)

        h = self.encoder(h)
        h = self.decoder(h)
        
        return h
  
    def get_codes(self, x):
        return self.encoder(x)