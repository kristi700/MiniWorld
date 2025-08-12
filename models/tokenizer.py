import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize

class VQ_VAE(nn.Module):
    def __init__(self, image_shape, embedding_dim=64, codebook_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

        self.quantizer = VectorQuantize(
            dim = embedding_dim,
            codebook_size = codebook_size,
            accept_image_fmap=True 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        quantized, indicies, cmt_loss = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, indicies, cmt_loss