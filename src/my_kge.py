"""
Module to make KG node embeddings by averaging neighboring node embeddings
"""

import argparse
import re
import rdflib
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import networkx as nx
from hdt import HDTDocument


class KGE:
    """
    class KGE
    """

    # Sentence transformer model name
    model_name = "all-MiniLM-L6-v2"

    # Regex to detect RDF URIs
    uri_regex = re.compile(
        r"^(https?:\/\/)?(www\.)?([A-Za-z_0-9.-]+)\.[a-z]{2,}(/[A-Za-z_0-9.-]*)*\/?$"
    )

    def __init__(self, files) -> None:
        """
        @param files
        """
        self.graph = self.load_kg(files)
        self.model = SentenceTransformer(self.model_name)


    def load_kg(self, files) -> rdflib.Graph:
        """
        @param files
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
        @param file
        :param graph
        """
        def node_type(node: str):
            if self.uri_regex.match(node):
                return rdflib.URIRef(node)
            elif node != "":
                return rdflib.Literal(node)
            return rdflib.BNode()

        document = HDTDocument(file)
        triples, cardinality = document.search_triples("", "", "")
        for s, p, o in triples:
            s = node_type(s)
            graph.add((node_type(s), node_type(p), node_type(o)))

    def load_neighbors(self, iri: str, print_len: bool = False) -> set:
        """
        Loading all node neighbors
        :param print_len print the number of neighbors?
        """
        neighbors = set()
        for o in self.graph.objects(subject=rdflib.URIRef(iri)):
            # if isinstance(o, rdflib.Literal):
            neighbors.add(o)
        if print_len:
            print(f"{iri} has {len(list(neighbors))} neighbors")
        return neighbors

    def create_neighbor_embeddings(self, iri: str, level2: bool = False) -> list:
        """
        Return embeddings of node neighbors.
        """
        neighbors = self.load_neighbors(iri)
        neighbor_embeddings = []
        neighbor_iris = []

        for node in neighbors:
            if isinstance(node, rdflib.Literal):
                neighbor_embeddings.append(self.model.encode(node.value))

            elif level2 and isinstance(node, rdflib.URIRef):
                neighbor_iris.append(node.value)

        return neighbor_embeddings, neighbor_iris

    def create_embedding(self, iri: str, level2: bool = False) -> list:
        """
        Embed one node.
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

    def create_embeddings(self):
        """
        Embed all nodes.
        """
        embeddings = {}
        # print(f"Graph contains {len(list(self.graph.subjects(unique=True)))} unique triples.")
        for n, s in enumerate(self.graph.subjects(unique=True)):
            val = str(s) if isinstance(s, rdflib.URIRef) else ""
            embedding = self.create_embedding(val)
            if not isinstance(embedding, int):
                embeddings[val] = embedding
                print(f"{s}: {n + 1} / {len(list(self.graph.subjects(unique=True)))}")

        print(f"Embeddings contain {len(list(embeddings))} unique triple(s).")

        return embeddings

    def node_match_all(self):
        """
        match a node to node
        """
        node_matches = []
        es = self.create_embeddings()

        labels = list(es.keys())
        embeddings = list(es.values())

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                node_matches.append(
                    (
                        self.cosine_similarity(embeddings[i], embeddings[j]),
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
        """
        node_matches = []
        close_matches = []
        text_es = self.model.encode(text)
        es = self.create_embeddings()

        labels = list(es.keys())
        embeddings = list(es.values())

        for i in range(len(embeddings)):
            node_matches.append(
                (self.cosine_similarity(embeddings[i], text_es), labels[i], text)
            )

        node_matches = sorted(node_matches, key=lambda x: x[0], reverse=True)
        # print(f"Embeddings count: {len(es)} Score count: {len(matches)}")
        print("Similarity Score: Node 1,      Node 2")
        for match in node_matches:
            if match[0] > 0.4:
                print(f"{match[0]}: {match[1].split("/")[-1]}, {match[2]}")
                close_matches.append(match)
            else:
                break

        return close_matches

    @staticmethod
    def cosine_similarity(es1, es2):
        """
        :param es1, es2
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


parser = argparse.ArgumentParser()
parser.add_argument("--files", type=str, nargs="+", help="List of ttl files")
args = parser.parse_args()

if __name__ == "__main__":

    fileDir = "./files/"

    if not args.files:
        args.files = [fileDir + "climatepub4-kg.hdt"]
        # args.files = [f+"jim.ttl", f+"james.ttl", f+"rada.ttl"]
        # args.files = [f+"output.ttl"]

    myKGE = KGE(args.files)
    myKGE.create_embeddings()

    # myKGE.node_match_all()

    matches = myKGE.node_match_text("SPECTROHELIOGRAPHS")
    if len(matches) > 0:
        myKGE.view_graph(matches[0][1])

    # es = myKGE.es
    # pca = PCA(n_components=2)
    # reduced_embedding_pca = pca.fit_transform(list(es.values()))

    # plt.figure(figsize=(8, 6))
    # plt.scatter(reduced_embedding_pca[:, 0], reduced_embedding_pca[:, 1], c='blue', alpha=0.5)

    # for i, label in enumerate(es.keys()):
    #   plt.text(reduced_embedding_pca[i, 0], reduced_embedding_pca[i, 1], label, fontsize=9, ha='right')

    # plt.title("3D Visualization of 384-Dimensional Embedding (PCA)")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.ylabel("Principal Component 3")
    # plt.show()
