"""
Module to make KG node embeddings by averaging neighboring node embeddings
"""

import argparse
import re
import os
import csv
import rdflib
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import networkx as nx
# from torch.nn.functional import normalize
from hdt import HDTDocument


# DATASET PATH
FILE_DIR = "./files/"


class KGE:
    """
    class KGE. Simple knowledge graph embedding computation.
    """

    # Sentence transformer model name
    model_name = "all-MiniLM-L6-v2"

    # Regex to detect RDF URIs
    uri_regex = re.compile(
        r"^(https?:\/\/)?(www\.)?([A-Za-z_0-9.-]+)\.[a-z]{2,}(/[A-Za-z_0-9.-]*)*\/?$"
    )

    def __init__(self, files) -> None:
        """
        Initialization.
        @param files to be parsed.
        """
        self.graph = self.load_kg(files)
        self.model = SentenceTransformer(self.model_name)


    def load_kg(self, files) -> rdflib.Graph:
        """
        Parses the given files to load the graphs.
        @param files List of graph containing files that need to be loaded.
        """
        graph = rdflib.Graph()
        for f in files:
            ext = f.split(".")[-1]

            if ext == "ttl":
                graph.parse(f, format="ttl")

            elif ext == "hdt":
                self.load_hdt(f, graph)

        print(
            f"Graph contains {len(graph)} triples: \
          {len(list(graph.subjects(unique=True)))} subjects, \
          {len(list(graph.predicates(unique=True)))} predicates, \
          {len(list(graph.objects(unique=True)))} objects."
        )

        return graph

    def load_hdt(self, file, graph):
        """
        Loads graphs from hdt files.
        @param graph the graph object to be initialized
        @param file hdt file to be loaded
        """
        def node_type(node: str):
            if self.uri_regex.match(node):
                return rdflib.URIRef(node)
            elif node != "":
                return rdflib.Literal(node)
            return rdflib.BNode()

        document = HDTDocument(file)
        triples, _ = document.search_triples("", "", "")
        for s, p, o in triples:
            graph.add((node_type(s), node_type(p), node_type(o)))

    def load_neighbors(self, iri: str, print_len: bool = False) -> set:
        """
        Loading all node neighbors
        @param iri the iri of the node whose neighbors we want
        @param print_len print the number of neighbors?
        """
        neighbors = set()
        for node in self.graph.objects(subject=rdflib.URIRef(iri), unique=False):
            neighbors.add(node)
        if print_len:
            print(f"{iri} has {len(list(neighbors))} neighbors")
        return neighbors
    
    def create_attr_val_embeddings(self, iri: str, minRows: int = 15) -> torch.Tensor:
        """
        Create tensor-based embeddings for node neighbors.
        :param iri: The IRI of the node.
        :return: A tensor of neighbor embeddings.
        """
        neighbors = []
        
        # Iterate through the predicate-object pairs of the given IRI
        for p, o in self.graph.predicate_objects(subject=rdflib.URIRef(iri)):

            # Extract the meaningful part of the predicate using regex
            p = re.search(r'([^/#]+)$', str(p))

            # Skip if Orcid Id
            if p in ("orcidID", "P496"):
                continue
            
            # Concatenate the predicate and object embeddings
            encoded = torch.cat((torch.tensor(self.model.
                                              encode(p, normalize_embeddings=True), dtype=torch.float32),
                                torch.tensor(self.model.
                                              encode(o.value, normalize_embeddings=True), dtype=torch.float32)))
            neighbors.append(encoded)

        # Stack all neighbor embeddings into a single tensor
        if neighbors:
            neighbors = torch.stack(neighbors)
            rows, cols = neighbors.shape
            if rows < minRows:
                padding = torch.zeros((minRows - rows, cols))
                neighbors = torch.cat((neighbors, padding), dim=0)
            return neighbors

        # Return an empty tensor if there are no neighbors
        return torch.empty((0,), dtype=torch.float32)

    def create_neighbor_embeddings(self, iri: str, level2: bool = False) -> list:
        """
        Return embeddings of node neighbors.
        @param iri the iri of the node whose neighbor's embeddings we want
        @param level2 do we want the neighbors of its neighbors?
        """
        neighbors = self.load_neighbors(iri)
        neighbor_embeddings = []
        neighbor_iris = []

        for node in neighbors:
            if isinstance(node, rdflib.Literal):
                neighbor_embeddings.append(self.model.encode(node.value, normalize_embeddings=True))

            elif level2 and isinstance(node, rdflib.URIRef):
                neighbor_iris.append(node.value)

        return neighbor_embeddings, neighbor_iris

    def create_embedding(self, iri: str, level2: bool = False) -> list:
        """
        Embed one node in the parsed graph.
        @param iri the iri of the node to be embedded
        @param2 do we want to consider the neighbors of its neighbors?
        """
        embedding = 0
        neighbor_embeddings, neighbor_iris = self.create_neighbor_embeddings(iri, level2)

        for node in neighbor_embeddings:
            embedding += node

        if len(neighbor_iris) != 0:
            distant_embeddings = []
            for iri in neighbor_iris:
                distant_embeddings += self.create_neighbor_embeddings(iri, False)[0]
            
            for node in distant_embeddings:
                embedding += 0.5 * distant_embeddings
            
            num_literals = len(distant_embeddings)
            embedding /= num_literals if num_literals != 0 else 0                

        num_literals = len(neighbor_embeddings)
        embedding /= num_literals if num_literals != 0 else 0

        return embedding

    def create_embeddings(self, outfile: str = "embeddings.csv"):
        """
        Embed all nodes in the parsed graph.
        @param outfile the file to write the node embeddings to
        """
        embeddings = {}
        total = len(list(self.graph.subjects(unique=True)))
        # print(f"Graph contains {len(list(self.graph.subjects(unique=True)))} unique triples.")
        for n, s in enumerate(self.graph.subjects(unique=True)):
            val = str(s) if isinstance(s, rdflib.URIRef) else ""
            embedding = self.create_embedding(val)
            if not isinstance(embedding, int):
                embeddings[val] = embedding
                print(f"{s}: {n + 1} / {total}")

        # print(f"Embeddings contain {len(list(embeddings))} unique triple(s).")
        with open(outfile, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["IRI", "Embedding"])  # Write header
            for val, embedding in embeddings.items():
                writer.writerow([val, ",".join(map(str, embedding))])  # Convert embedding to CSV-friendly format

        print(f"Embeddings saved to {outfile}.")
        return embeddings

    def node_match_all(self):
        """
        match a node to node
        """
        node_matches = []
        es = self.create_embeddings()

        labels = list(es.keys())
        embeddings = list(es.values())

        for i, e in enumerate(embeddings):
            for j, e2 in enumerate(embeddings):
                if i != j:
                    node_matches.append(
                        (
                            np.dot(e, e2),
                            labels[i],
                            labels[j],
                        )
                    )

        node_matches = sorted(node_matches, key=lambda x: x[0], reverse=True)
        print(f"Score count: {len(node_matches)}")
        for match in node_matches:
            if match[0] > 0.5:
                print(
                    f"{match[0]}: {match[1].split("/")[-1]}, {match[2].split("/")[-1]}"
                )
            else:
                break

        return node_matches

    def node_match_text(self, text):
        """
        match a node to text
        @param text the text to match
        """
        node_matches = []
        close_matches = []
        text_es = self.model.encode(text, normalize_embeddings=True)
        es = self.create_embeddings()

        labels = list(es.keys())
        embeddings = list(es.values())

        for e, i in enumerate(embeddings):
            node_matches.append(
                (labels[i], np.dot(e, text_es))
            )

        node_matches = sorted(node_matches, key=lambda x: x[1], reverse=True)
        # print(f"Embeddings count: {len(es)} Score count: {len(matches)}")
        print("Node Name,      Similarity Score")
        for match in node_matches:
            if match[1] > 0.4:
                print(f"{match[0].split("/")[-1]}: {match[0]}")
                close_matches.append(match)
            else:
                break

        return close_matches

    @staticmethod
    def cosine_similarity(es1, es2):
        """
        @param es1, es2
        """
        similarity = np.dot(es1, es2) / (np.linalg.norm(es1) * np.linalg.norm(es2))
        return similarity

    def view_graph(self: any, iri: str) -> None:
        """
        :param iri
        """
        G = nx.Graph()
        G.add_node(iri)

        for subj, pred, obj in self.graph.triples((rdflib.URIRef(iri), None, None)):
            G.add_edge(str(subj), str(obj), label=str(pred))

        pos = nx.spring_layout(
            G, k=0.5, iterations=50
        )  # Layout for positioning the nodes
        labels = nx.get_edge_attributes(G, "label")

        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color="gray")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=labels, font_size=10, font_color="red"
        )
        plt.title(f"{iri}")
        plt.axis("off")  # Hide the axis
        plt.show()


if __name__ == "__main__":

    # parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_dir",
                        default=FILE_DIR,
                        help="List of files to be parsed: ")
    parser.add_argument("-t", "--text_match",
                        default=None,
                        help="Text to be matched against graph nodes")
    parser.add_argument("-n", "--node_match",
                        default=None,
                        help="Match all graphs nodes against each other")
    args = parser.parse_args()

    try:
        # List all files in the directory
        FILES = [
            f for f in os.listdir(args.file_dir)
            if os.path.isfile(os.path.join(args.file_dir, f))
        ]
        if len(FILES) == 0:
            raise SystemExit("Error: No files includes in the files/ directory.")
    except Exception as e:
        raise SystemExit("Error parsing graph files in the files/ directory.") from e

    myKGE = KGE(FILES)
    myKGE.create_embeddings()
    
    if args.text_match:
        matches = myKGE.node_match_text(args.text_match)
        if len(matches) > 0:
            myKGE.view_graph(matches[0][1])

    if args.node_match:
        matches = myKGE.node_match_all()
        if len(matches) > 0:
            myKGE.view_graph(matches[0][1])
