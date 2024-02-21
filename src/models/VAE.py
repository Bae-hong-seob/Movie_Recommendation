import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class VAE(nn.Module):
    """
    input : encoder_dims

        encoder = [n_items, dim1, dim2]
        decoder_dims = [dim2, dim1, n_items]
    """
    
    def __init__(self, config, encoder_dims, decoder_dims=None, dropout_rate=0.5): 
        super(VAE, self).__init__()
        self.config = config
        
        self.encoder_dims = encoder_dims
        if decoder_dims:
            assert decoder_dims[0] == encoder_dims[-1], "In and Out dimensions must equal to each other"
            assert decoder_dims[-1] == encoder_dims[0], "Latent dimension for p- and q- network mismatches."
            self.decoder_dims = decoder_dims
        else:
            self.decoder_dims = encoder_dims[::-1]
        
        self.encoder = self.build_layers(self.encoder_dims, config['activate_function'], module='encoder')
        self.decoder = self.build_layers(self.decoder_dims, config['activate_function'], module='decoder')
        
        self.drop = nn.Dropout(dropout_rate)
        
    def build_layers(self, dims, activate_function = 'ReLU', module=None):
        """
        Helper function to build layers based on the provided dimensions.

        Parameters:
            - dims (list): List of dimensions for the layers.

        Returns:
            - nn.Sequential: Sequential container for the layers.
        """
        layers = []
        if module == 'encoder':
            dims[-1] = dims[-1]*2 #last dims for mu and logvar.
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
        if self.config['denoising'] == 'Dropout':
            h = F.normalize(input)
            h = self.drop(h)
            
        elif self.config['denoising'] == 'Gaussian':
            h = self.add_noise(input)
            h = F.normalize(h)

        h = self.encoder(h)
        h, mu, logvar = self.reparameterize(h)
        h = self.decoder(h)
        
        return h, mu, logvar
  
    def get_codes(self, x):
        return self.encoder(x)
    
    def reparameterize(self, h):
        '''
        make mu, logvar using h(vector) linear layer
            Args: 
                decoder_dims -> list
                h : encoder last output -> vector(hidden dim)
            
            Return:
                z : mu + (std*eps) has same dims with encoder output and decoder input
        '''
        mu, logvar = h[:, :h.shape[1]//2], h[:, h.shape[1]//2:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        z = mu + (std*eps)
        return z, mu, logvar