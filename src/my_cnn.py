"""
Module for training a CNN to generate node embeddings.
"""

import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn.functional import normalize
from math import floor


class CNN(nn.Module):
    """
    CNN based torch.nn module for generating node embeddings.

    Args:
        input_dim (int): Dimension of input embeddings.
        num_filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
    """

    def __init__(self, batch_size: int, kernel_size: int=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=kernel_size, padding=1
        ) #30, 384

        self.conv2 = nn.Conv2d(
            in_channels=6 * 2, out_channels=24, kernel_size=kernel_size, padding=1
        ) # 60, 192

        self.conv3 = nn.Conv2d(
            in_channels=6 * 4, out_channels=48, kernel_size=kernel_size, padding=1
        ) # 120, 96

        self.conv4 = nn.Conv2d(
            in_channels=6 * 8, out_channels=96, kernel_size=kernel_size, padding=1
        ) # 240, 48

        self.conv5 = nn.Conv2d(
            in_channels=6 * 16, out_channels=192, kernel_size=kernel_size, padding=1
        ) # 480, 24

        # self.conv1 = nn.Conv2d(
        #     in_channels=batch_size, out_channels=batch_size * 2, kernel_size=kernel_size, padding=1
        # ) #30, 384


        # self.conv2 = nn.Conv2d(
        #     in_channels=batch_size * 2, out_channels=batch_size * 4, kernel_size=kernel_size, padding=1
        # ) # 60, 192


        # self.conv3 = nn.Conv2d(
        #     in_channels=batch_size * 4, out_channels=batch_size * 8, kernel_size=kernel_size, padding=1
        # ) # 120, 96

        # self.conv4 = nn.Conv2d(
        #     in_channels=batch_size * 8, out_channels=batch_size * 16, kernel_size=kernel_size, padding=1
        # ) # 240, 48

        # self.conv5 = nn.Conv2d(
        #     in_channels=batch_size * 16, out_channels=batch_size * 32, kernel_size=kernel_size, padding=1
        # ) # 480, 24

        # self.fc = nn.Linear(1, input_dim)
        self.max_pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AvgPool2d(2)
        self.avg_pool4 = nn.AvgPool1d(4)
        self.max_pool4 = nn.MaxPool1d(4)
        self.relu = nn.LeakyReLU()

        # conv_output_dim = (N, Cout, 1)
        self.fc0 = nn.Linear(96, 48)
        self.fc01 = nn.Linear(48, 24)
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        """
        Forward pass of the CNN.
        :param x (torch.Tensor): Input tensor with shape (batch_size, 1, input_dim, input_dim).
        :returns x (torch.Tensor): Output tensor with shape (batch_size, input_dim).
        """
        x = torch.unsqueeze(x, dim=1)

        x = self.relu(self.conv1(x))
        x = self.max_pool(x)

        x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        x = self.relu(self.conv3(x))
        x = self.avg_pool(x)

        x = self.relu(self.conv4(x))
        x = self.avg_pool(x)

        x = x.squeeze(2)

        # x = self.relu(self.conv5(x))
        x = self.max_pool4(x)

        # x = self.fc0(x)
        # x = self.fc01(x)
      
        # x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.squeeze(2)

        return x

class CustomLoss(torch.nn.Module):
    """
    Custom loss !
    """

    def __init__(self, margin=0.1, device='cpu'):
        """
        Custom loss function to ensure S_i is similar to T_i
        but dissimilar to T_j where j != i.
        Args:
            margin (float): Margin for dissimilarity penalty. Higher margin encourages more dissimilarity.
        """
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, snes: Tensor, tnes: Tensor) -> Tensor:
        """
        Forward !
        """
        batch_size = snes.shape[0]

        norm_snes = normalize(snes, p=2, dim=1)
        norm_tnes = normalize(tnes, p=2, dim=1)

        pos_loss = 1 - torch.sum(norm_snes * norm_tnes, dim=1)
        # neg_loss = torch.zeros(batch_size, device=self.device)

        # for i in range(batch_size):
        #     negatives = torch.sum(
        #         norm_tnes * norm_snes[i], dim=1
        #     )
        #     negatives[i] = 0
        #     neg_loss[i] = torch.sum(negatives) / batch_size

        negatives = torch.mm(norm_snes, norm_tnes.T)  # Pairwise cosine similarities
        negatives.fill_diagonal_(0)  # Ignore diagonal (self-similarity)

        neg_loss = torch.clamp(self.margin - negatives, min=0).sum(dim=1) / snes.size(0)

        # Total loss
        loss = pos_loss.mean() + neg_loss.mean()

        return loss


class NodeEmbeddingCNN:
    """
    Trainer for the CNN model, optimized for generating node embeddings.

    Args:
        data_loader (dict): Dictionary of data pairs where each key maps to a (input, target) tuple.
        model (nn.Module): CNN model to be trained.
        criterion (nn.Module): Loss function for training.
        optimizer (optim.Optimizer): Optimizer for training.
        device (str): Device for computation, 'cpu' or 'cuda'.
    """

    criterion = CustomLoss()
    cnnConfigPath = "./cnn_config.pth"
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    def __init__(
        self,
        batch_size: int,
        lr: float = 0.0005,
        eps: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.eps = eps
        self.device = device
        self.model = CNN(batch_size=batch_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def train(self, snes: Tensor, tnes: Tensor, max_epochs: int = 100000) -> None:
        """
        Train the CNN model.
        :param epochs (int): Number of training epochs
        """
        for epoch in range(max_epochs):

            # for source_embeddings, target_embedding in data_loader.values():
            self.optimizer.zero_grad()

            source_output = self.model(
                snes.to(self.device)
            )
            target_output = self.model(
               tnes.to(self.device)
            )
            loss = self.criterion(
                source_output,
                target_output
            )
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            if loss.item() < self.eps:
                break

        torch.save(self.model.state_dict(), self.cnnConfigPath)

    def test(self, source_nes: torch.Tensor) -> torch.Tensor:
        """
        Test the CNN model
        :param source_nes, source node embeddings
        """
        self.model.load_state_dict(torch.load(self.cnnConfigPath, weights_only=True))
        output = self.model(source_nes.to(self.device))

        return output
    



# tensor([1.0000], grad_fn=<SumBackward1>)