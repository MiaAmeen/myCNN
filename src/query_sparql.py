"""
query sparql
"""
import subprocess


def run_query_engine(endpoint, query, output_file):
    """
    run sparql
    """
    try:
        subprocess.run(
            [
                "node", "./src/comunica/query_engine.js", 
                endpoint, 
                query, 
                output_file
            ], 
            check=True
        )
        print(f"Results written to {output_file}")
        return None
    
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    
    # soa_entry = "https://semopenalex.org/sparql"
    wd_entry = "https://query.wikidata.org/sparql"
    # query_file = "./src/comunica/query.sparql"
    output_file = "query_results.csv"
    file_dir = "./files/"
    openalex = file_dir + "semopenalex.ttl"  # Replace with your file path


    # run_query_engine(soa_entry, query_file, output_file)
