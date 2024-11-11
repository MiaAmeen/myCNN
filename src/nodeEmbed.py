"""
This module embeds nodes using CNN model
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import csv
import rdflib

import torch
import torch.nn as nn
import torch.optim as optim

from my_kge import KGE
from my_cnn import NodeEmbeddingCNN


class NodeEmbed:
    """
    Embeds nodes i think
    """
    model_name = "all-MiniLM-L6-v2"

    def __init__(self, file_dir: str):
        self.file_dir = file_dir
        self.model = SentenceTransformer(self.model_name)
        self.pairs = {}

    def train_cnn(self, file: str) -> dict:
        """
        loads data
        """
        # Open the CSV file in read/write mode
        with open(f"{self.file_dir}{file}", "r") as author_pairs:
            reader = csv.reader(author_pairs)
            data_loader = {}

            for row in reader:
                graph = KGE([f"{self.file_dir}{row[0].strip()}"])
                snes = graph.create_neighbor_embeddings(row[1].strip())[0]
                tnes = graph.create_neighbor_embeddings(row[2].strip())[0]

                data_loader[row[0].strip()] = [snes, tnes]
                # sne = self.create_cnn_embeddings(data_loader)
                # pairs[row[0].strip()] = [sne, tne]

            cnn = NodeEmbeddingCNN(data_loader)
            cnn.train()


        # first_key = next(iter(pairs))
        # with open("source.txt", "w") as f:
        #     f.write(np.array2string(pairs[first_key][0], separator=","))
        # with open("target.txt", "w") as f:
        #     f.write(np.array2string(pairs[first_key][1], separator=","))

    # def create_cnn_embeddings(self, data_loader):
    #     cnn = CNN(data_loader)
    #     cnn.train()

    #     first_key = next(iter(data_loader))
    #     return cnn.test(data_loader[first_key][0])


if __name__ == "__main__":
    file_dir = "./files/"
    NE = NodeEmbed(file_dir)
    NE.train_cnn("authorPairs.csv")
