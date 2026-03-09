# CHANGE WHEN NEEDED TO LOAD PYTERRIER------------------------------------------
import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-25.0.2"
import pyterrier as pt
import json
#-------------------------------------------------------------------------------

import pandas as pd

query = "energy" # Change search queryfor testing (TODO: Kristen will add frontend/cli interaction)

df = pd.read_csv("songs_expanded.csv")
base = os.path.abspath("var_song_five_expanded/index") 
bm25 = pt.terrier.Retriever(base, wmodel="BM25")

def search_songs(query: str, top_k: int = 30) -> pd.DataFrame:
    results = bm25.search(query)
    return df.iloc[results.head(10).docid].song

search_songs(query)

while True:
    query = input("\nEnter search query (or 'q' to quit): ").strip()
    if query.lower() == "q":
        print("Goodbye!")
        break
    if not query:
        continue
    print("\n Personalized results: ")
    print(search_songs(query))