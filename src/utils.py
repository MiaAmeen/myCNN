"""
Utils file, converts csv to ttl
"""

from my_kge import KGE
import csv
import os
from rdflib import Graph, URIRef, Literal
from numpy import mean

file_dir = "./files/"

@staticmethod
def csv_ttl(file: str) -> None:
    """
    :param input
    """
    # Open the CSV file in read mode
    with open(file, "r") as ifile:
        reader = csv.reader(ifile)

        # Open the output file in write mode
        with open(f"{os.path.splitext(file)[0]}.ttl", "w") as outfile:

            # Loop through each row in the CSV
            for row in reader:

                # # Check if the third element (the object) is a literal (i.e., not a URI)
                # if not row[-1].startswith("<"):
                #     row[-1] = '"' + row[-1] + '"'  # Wrap literals in quotes

                # Construct the Turtle triple with a period at the end
                triple = f'<{row[0]}> <{row[1]}> "{row[2]}" .\n'

                # Write the triple to the output file
                outfile.write(triple)

@staticmethod
def orcid_csv(file: str) -> None:
    """
    mine orcid ids from input ttl file
    """
    infile = file_dir + file
    graph = Graph()
    graph.parse(infile, format="turtle")

    with open("output.ttl", "w") as ofile: 
        for _, p, o in graph:
            if str(p) == "https://dbpedia.org/ontology/orcidId":
                
                orcid_id = str(o).split("/")[-1]
                ofile.write(f"{orcid_id}\n")
                print(orcid_id)


@staticmethod
def author_pairs(sourceFile: str, targetFile: str, orcidFile: str) -> None:
    """
    Accepts three inputs:
    @param sourceFile: List of triples in ttl format from a source graph
    @param targetFile: List of triples in ttl format from a target graph
    @param orcidFile: List of orcid IDs of all the authors in the source graph, 
        some of which are in the target graph as well.
    """
    # orcid_ids = []
    file_paths = [f"{file_dir}{file_name}" for file_name in [sourceFile, targetFile, orcidFile]]

    with open(file_paths[2], encoding="utf-8") as orcids, \
        open("pairs_output.csv", mode="w", newline='', encoding="utf-8") as output_file:

        source_graph = Graph().parse(file_paths[0], format="turtle")
        target_graph = Graph().parse(file_paths[1], format="turtle")

        orcid_ids = (line.strip() for line in orcids)

        writer = csv.writer(output_file)
        
        for orcid_id in orcid_ids:
            
            # Find the subject in the target graph associated with this ORCID
            target_subjects = list(target_graph.subjects(
                predicate=URIRef("http://www.wikidata.org/prop/direct/P496"),
                object=Literal(f"{orcid_id}"), unique=True))
            
            
            # Find the subject in the source graph associated with this ORCID
            source_subjects = list(source_graph.subjects(
                predicate=URIRef("https://dbpedia.org/ontology/orcidId"), 
                object=Literal(f"https://orcid.org/{orcid_id}"), unique=True)) \
                if len(target_subjects) > 0 else None

            # Write matches to the CSV
            if target_subjects and source_subjects:
                print(f"{orcid_id}, {source_subjects[0]}, {target_subjects[0]}")
                writer.writerow([orcid_id, source_subjects[0], target_subjects[0]])

def check_neighbors() -> None:
    """
    check max/min number of neighbors of all training nodes
    """
    kge = KGE(["./files/semopenalex.ttl", "./files/wikidata.ttl"])
    with open("./files/pairs_output.csv", "r", encoding="utf-8") as pairs:
        reader = csv.reader(pairs)

        nbors = []

        for row in reader:
            source = row[1].strip()
            target = row[2].strip()

            nbor1 = kge.load_neighbors(source)
            len1 = len(nbor1)
            nbor2 = kge.load_neighbors(target)
            len2 = len(nbor2)
            if len1 < 15 and len2 < 15:
                print(f"{row[0]}, {source}, {target}")
                nbors.append(len1)
                nbors.append(len2)
        
        print(len(nbors))
        print(min(nbors))
        print(mean(nbors))
        print(max(nbors))


check_neighbors()            
# author_pairs("semopenalex.ttl", "wikidata.ttl", "output.csv")
# orcid_csv("semopenalex.ttl")
# csv_ttl("./files/semopenalex.csv")
# csv_ttl("./files/jim_openalex.csv")
