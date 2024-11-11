"""
Module for training a CNN to generate node embeddings.
"""

import torch
from torch import nn
from torch import optim


class CNN(nn.Module):
    """
    CNN based torch.nn module for generating node embeddings.

    Args:
        input_dim (int): Dimension of input embeddings.
        num_filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
    """

    def __init__(self, input_dim=384, num_filters=64, kernel_size=3):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=1
        )
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            num_filters, num_filters * 2, kernel_size=kernel_size, padding=1
        )
        self.pool2 = nn.MaxPool1d(2)

        conv_output_dim = input_dim // 4

        self.fc = nn.Linear(num_filters * 2 * conv_output_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the CNN.
        :param x (torch.Tensor): Input tensor with shape (batch_size, input_dim).
        :returns x (torch.Tensor): Output tensor with shape (1, input_dim).
        """
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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

    criterion = nn.CosineEmbeddingLoss
    cnnConfigPath = "./cnn_config.pth"
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    def __init__(
        self,
        data_loader,
        device: str = "cpu",
        model: torch.nn.Module = CNN(input_dim=384),
    ) -> None:
        self.data_loader = data_loader
        self.device = device
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train(self, epochs: int = 1000) -> None:
        """
        Train the CNN model.
        :param epochs (int): Number of training epochs
        """
        for epoch in range(epochs):

            for source_embeddings, target_embedding in self.data_loader.values():

                self.optimizer.zero_grad()
                source_output = self.model(
                    torch.tensor(source_embeddings, dtype=torch.float32).to(self.device)
                )
                target_output = self.model(
                    torch.tensor(target_embedding, dtype=torch.float32).to(self.device)
                )
                loss = self.criterion(
                    source_output,
                    target_output
                )
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.save(self.model.state_dict(), self.cnnConfigPath)

    def test(self, source_nes: torch.Tensor) -> torch.Tensor:
        """
        Test the CNN model
        :param source_nes, source node embeddings
        """
        self.model.load_state_dict(torch.load(self.cnnConfigPath, weights_only=True))
        output = self.model(source_nes)

        return output
