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

    def train_cnn(self, pairs_file: str, node_files: list) -> dict:
        """
        loads data
        """
        # Open the CSV file in read/write mode
        with open(f"{self.file_dir}{pairs_file}", "r") as author_pairs:
            reader = csv.reader(author_pairs)            
            node_files = [self.file_dir + filename for filename in node_files]
            graph = KGE(node_files)

            snes = []
            tnes = []
            for row in reader:
                snes.append(graph.create_attr_val_embeddings(row[1].strip(), minRows = 15))
                tnes.append(graph.create_attr_val_embeddings(row[2].strip(), minRows = 15))

                # data_loader[row[0].strip()] = [snes, tnes]
                # sne = self.create_cnn_embeddings(data_loader)
                # pairs[row[0].strip()] = [sne, tne]

            snes = torch.stack(snes)
            tnes = torch.stack(tnes)
            cnn = NodeEmbeddingCNN(batch_size=snes.shape[1])
            cnn.train(snes, tnes)

    def test_cnn(self, pairs_file: str, node_files: list):
        """
        tests cnn
        """
        # Open the CSV file in read/write mode
        with open(f"{self.file_dir}{pairs_file}", "r") as author_pairs:
            reader = csv.reader(author_pairs)            
            node_files = [self.file_dir + filename for filename in node_files]
            graph = KGE(node_files)

            snes = []
            tnes = []
            for row in reader:
                snes.append(graph.create_attr_val_embeddings(row[1].strip(), minRows = 11))
                tnes.append(graph.create_attr_val_embeddings(row[2].strip(), minRows = 11))

            snes = torch.stack(snes)
            tnes = torch.stack(tnes)
            cnn = NodeEmbeddingCNN(batch_size=snes.shape[1])
            sne = cnn.test(snes)
            tne = cnn.test(tnes)
            diff = torch.cosine_similarity(sne, tne, dim=1)
            print(diff)


if __name__ == "__main__":

    file_dir = "./files/"
    NE = NodeEmbed(file_dir)
    NE.train_cnn("training_pairs.csv", ["semopenalex.ttl", "wikidata.ttl"])
    NE.test_cnn("training_pairs.csv", ["semopenalex.ttl", "wikidata.ttl"])

