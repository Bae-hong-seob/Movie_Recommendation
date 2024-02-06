import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 

class DAE(nn.Module):
    """
    input : encoder_dims

        encoder = [n_items, dim1, dim2]
        decoder_dims = [dim2, dim1, n_items]
    """
    
    def __init__(self, config, encoder_dims, decoder_dims=None, dropout=0.5): 
        super(DAE, self).__init__()
        self.encoder_dims = encoder_dims
        self.drop = nn.Dropout(dropout)
        if decoder_dims:
            assert decoder_dims[0] == encoder_dims[-1], "In and Out dimensions must equal to each other"
            assert decoder_dims[-1] == encoder_dims[0], "Latent dimension for p- and q- network mismatches."
            self.decoder_dims = decoder_dims
        else:
            self.decoder_dims = encoder_dims[::-1]
        
        self.encoder = self.build_layers(self.encoder_dims, config['activate_function'])
        self.decoder = self.build_layers(self.decoder_dims, config['activate_function'])
        

    def build_layers(self, dims, activate_function = 'ReLU'):
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
            if i == len(dims)-1:
                pass
            else:
                if activate_function == 'ReLU':
                    layers.append(nn.ReLU())
                elif activate_function == 'Sigmoid':
                    layers.append(nn.Sigmoid())
                elif activate_function == 'Tanh':
                    layers.append(nn.Tanh())
                else:
                    pass

        return nn.Sequential(*layers)
        
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        h = self.encoder(h)
        h = self.decoder(h)
        
        return h
  
    def get_codes(self, x):
        return self.encoder(x)