# CHANGE WHEN NEEDED TO LOAD PYTERRIER------------------------------------------
import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-25.0.2"
import pyterrier as pt
import json
#-------------------------------------------------------------------------------

import pandas as pd

query = "energy" # Change search queryfor testing (TODO: Kristen will add frontend/cli interaction)

df = pd.read_csv("songs.csv")
base = os.path.abspath("var_song_five_stem/index") 
bm25 = pt.terrier.Retriever(base, wmodel="BM25")
results = bm25.search(query)
results.sort_values

# CHOOSE TO PRINT A "RANKED" VERSION BY FOLLOWER COUNT OR JUST DO RELEVANCE

# Ranked with follower Account 
#sorted_df = df.sort_values(by='followers', ascending=True)
#print(sorted_df.iloc[results.head(30).docid].song)

# Purely by relevance score
print(df.iloc[results.head(30).docid].song)
