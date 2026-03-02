# THIS IS A DRAFT
# This was me just playing with the class code seeing if the data works. I will likely change this to be a more streamlined system and return individual songs rather 
# than the playlists



import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-25.0.2"
#os.environ["JVM_PATH"] = "C:\Program Files\Java\jdk-25.0.2\lib\jvm.lib"

import pyterrier as pt
#pt.java.init()

import os
import json
import pandas as pd
#import pyterrier as pt

# Load data
MPD_PATH = "./mpd_data"

def load_mpd_playlists(path):
    df = []

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
            data = json.load(file)

            for playlist in data["playlists"]:
                docno = str(playlist["pid"])
                title = playlist.get("name", "")

                tracks = playlist.get("tracks", "") # "" as a default value
                track_text = ""
                for track in tracks: 
                    track_name = track.get("track_name", "")
                    artist_name = track.get("artist_name", "")
                    album_name = track.get("album_name", "")
                    track_name = (f"{track_name} {artist_name} {album_name}")

                full_text = title + " " + " ".join(track_text)

                df.append({
                    "docno": docno,
                    "text": full_text, 
                    "songs": tracks
                })

    return pd.DataFrame(df)

df = load_mpd_playlists(MPD_PATH)
print(f"Loaded {len(df)} playlists")

base = os.path.abspath("var/index") # Append new folders onto absolute path
os.makedirs(base, exist_ok=True) 

indexer = pt.index.IterDictIndexer(base, overwrite=True)
index_ref = indexer.index(df.to_dict(orient="records"))

# Explore Index
index = pt.IndexFactory.of(index_ref)

print("Number of documents:", index.getCollectionStatistics().getNumberOfDocuments())
print("Number of unique terms:", index.getCollectionStatistics().getNumberOfUniqueTerms())
print("Number of tokens:", index.getCollectionStatistics().getNumberOfTokens())
print("Average doc length:", index.getCollectionStatistics().getAverageDocumentLength())

# Peek at first 20 terms in the index
it = index.getLexicon()
for i, (term, entry) in enumerate(it):
    if i >= 40:
        break
    print(term, "-> df:", entry.getDocumentFrequency(), "cf:", entry.getFrequency())

query = "drive"
bm25 = pt.terrier.Retriever(index, wmodel="BM25")
results = bm25.search(query)
print(results)