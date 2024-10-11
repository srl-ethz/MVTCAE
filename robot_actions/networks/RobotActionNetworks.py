import torch
import torch.nn as nn
from utils.utils import Flatten, Unflatten

class EncoderAction(nn.Module):
    def __init__(self, flags, input_dim):
        super(EncoderAction, self).__init__()
        self.input_dim = input_dim
        self.class_dim = flags.class_dim
        self.style_dim = flags.style_dim
        self.hidden_dim = flags.dim
        self.factorized_representation = flags.factorized_representation

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.style_dim + self.class_dim),
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.style_dim + self.class_dim, self.class_dim)
        self.class_logvar = nn.Linear(self.style_dim + self.class_dim, self.class_dim)
        
        # optional style branch
        if self.factorized_representation:
            self.style_mu = nn.Linear(self.style_dim + self.class_dim, self.style_dim)
            self.style_logvar = nn.Linear(self.style_dim + self.class_dim, self.style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        if self.factorized_representation:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)

class DecoderAction(nn.Module):
    def __init__(self, flags, output_dim):
        super(DecoderAction, self).__init__()
        self.factorized_representation = flags.factorized_representation
        self.latent_dim = flags.class_dim + flags.style_dim
        self.hidden_dim = flags.dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too