"""
Utils file, converts csv to ttl
"""

import csv
import os


@staticmethod
def csv_ttl(input: str) -> None:
    """
    :param input

    """

    # Open the CSV file in read mode
    with open(input, "r") as ifile:
        reader = csv.reader(ifile)

        # Open the output file in write mode
        with open(f"{os.path.splitext(input)[0]}.ttl", "w") as outfile:

            # Loop through each row in the CSV
            for row in reader:

                # Check if the third element (the object) is a literal (i.e., not a URI)
                if not row[-1].startswith("<"):
                    row[-1] = '"' + row[-1] + '"'  # Wrap literals in quotes

                # Construct the Turtle triple with a period at the end
                triple = ""
                for elem in row:
                    triple += elem + " "
                triple += ".\n"

                # Write the triple to the output file
                outfile.write(triple)


# csv_ttl("./files/jim_wiki.csv")
# csv_ttl("./files/jim_openalex.csv")
