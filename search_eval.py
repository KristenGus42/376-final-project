# CHANGE WHEN NEEDED TO LOAD PYTERRIER------------------------------------------
import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-25.0.2"
import pyterrier as pt
import json
#-------------------------------------------------------------------------------

import pandas as pd

df = pd.read_csv("songs_expanded2.csv")
base = os.path.abspath("var_song_five_expanded2/index") 
bm25 = pt.terrier.Retriever(base, wmodel="BM25")

# test queries
queries = pd.DataFrame([
    {"qid": "1", "query": "sad breakup"},
    {"qid": "2", "query": "chill relaxing"},
    {"qid": "3", "query": "happy feel good vibes"},
    {"qid": "4", "query": "happy dance"},
    {"qid": "5", "query": "energetic gym"},
    {"qid": "6", "query": "depressed walk"},
    {"qid": "7", "query": "angry run"}, 
    {"qid": "8", "query": "heartbroken lonely"}

])

# print top 10 results per query for reference
for _, row in queries.iterrows():
    qid, query = row["qid"], row["query"]
    results = bm25.search(query)
    top_results = results.head(10)
    top_songs = df.iloc[top_results.docid]["song"].tolist()
    top_docnos = top_results["docno"].tolist()
    print(f"\nQuery: '{query}'")
    for i, (song, docno) in enumerate(zip(top_songs, top_docnos)):
        print(f"  {i:02d}. [{docno}] {song}")

# run bm25 on ALL queries at once for experiment
results = bm25.transform(queries)

qrels = pd.DataFrame([
    # qid 1: 'sad breakup'
    {"qid": "1", "docno": "1223_0", "label": 1},
    {"qid": "1", "docno": "1223_1", "label": 1},
    {"qid": "1", "docno": "1223_2", "label": 1},
    {"qid": "1", "docno": "1223_3", "label": 1},
    {"qid": "1", "docno": "1223_4", "label": 1},
    {"qid": "1", "docno": "1223_5", "label": 1},
    {"qid": "1", "docno": "1223_6", "label": 1},
    {"qid": "1", "docno": "1223_8", "label": 1},
    {"qid": "1", "docno": "1223_9", "label": 1},
    {"qid": "1", "docno": "1223_10", "label": 1},

    # qid 2: 'chill relaxing'
    {"qid": "2", "docno": "9630_0", "label": 0},   # Feel Good Inc by Gorillaz
    {"qid": "2", "docno": "9630_1", "label": 1},   # Clint Eastwood by Gorillaz
    {"qid": "2", "docno": "9630_2", "label": 1},   # Saturnz Barz by Gorillaz
    {"qid": "2", "docno": "9630_3", "label": 0},   # Icky Thump by The White Stripes
    {"qid": "2", "docno": "9630_4", "label": 0},   # Seven Nation Army by The White Stripes
    {"qid": "2", "docno": "9630_6", "label": 0},   # Paint It Black by The Rolling Stones
    {"qid": "2", "docno": "9630_7", "label": 0},   # Welcome To The Jungle by Guns N' Roses
    {"qid": "2", "docno": "9630_9", "label": 0},   # Sympathy For The Devil by The Rolling Stones
    {"qid": "2", "docno": "9630_10", "label": 0},  # Blue Orchid by The White Stripes
    {"qid": "2", "docno": "9630_12", "label": 0},  # American Idiot by Green Day

    # qid 3: 'happy feel good vibes'
    {"qid": "3", "docno": "7615_1", "label": 1},   # Sex With Me by Rihanna
    {"qid": "3", "docno": "7615_3", "label": 1},   # You Was Right by Lil Uzi Vert
    {"qid": "3", "docno": "7615_4", "label": 1},   # LUV by Tory Lanez
    {"qid": "3", "docno": "7615_5", "label": 1},   # Broccoli by DRAM
    {"qid": "3", "docno": "7615_7", "label": 1},   # The Way by Kehlani
    {"qid": "3", "docno": "7615_9", "label": 0},   # Uber Everywhere by MadeinTYO
    {"qid": "3", "docno": "7615_10", "label": 0},  # Drama by Roy Woods
    {"qid": "3", "docno": "7615_11", "label": 1},  # Weekend by Mac Miller
    {"qid": "3", "docno": "7615_12", "label": 1},  # Free Lunch by Isaiah Rashad
    {"qid": "3", "docno": "7615_13", "label": 0},  # R.I.P. Kevin Miller by Isaiah Rashad

    # qid 4: 'happy dance'
    {"qid": "4", "docno": "1234_0", "label": 1},   # Titanium by David Guetta
    {"qid": "4", "docno": "1234_1", "label": 1},   # Paris by David Guetta
    {"qid": "4", "docno": "1234_2", "label": 1},   # Give Me All Your Luvin' by Madonna
    {"qid": "4", "docno": "1234_3", "label": 1},   # Nah Neh Nah by Rico Bernasconi
    {"qid": "4", "docno": "1234_5", "label": 0},   # I Like That by Richard Vission
    {"qid": "4", "docno": "1234_6", "label": 0},   # Naked by DEV
    {"qid": "4", "docno": "1234_7", "label": 1},   # Respect by Melanie Amaro
    {"qid": "4", "docno": "1234_9", "label": 1},   # Never Forget You by Lupe Fiasco
    {"qid": "4", "docno": "1234_10", "label": 0},  # Changed The Way You Kiss Me
    {"qid": "4", "docno": "1234_11", "label": 1},  # Be Your Freak by Kenny Dope

    # qid 5: 'energetic gym'
    {"qid": "5", "docno": "85_0", "label": 1},     # Me, Myself & I by G-Eazy
    {"qid": "5", "docno": "85_1", "label": 1},     # Adventure Of A Lifetime by Coldplay
    {"qid": "5", "docno": "85_2", "label": 1},     # Hymn For The Weekend by Coldplay
    {"qid": "5", "docno": "85_3", "label": 1},     # Up&Up by Coldplay
    {"qid": "5", "docno": "85_4", "label": 1},     # A Head Full Of Dreams by Coldplay
    {"qid": "5", "docno": "85_5", "label": 1},     # A Sky Full of Stars by Coldplay
    {"qid": "5", "docno": "85_6", "label": 1},     # Paradise by Coldplay
    {"qid": "5", "docno": "85_7", "label": 1},     # Shot Me Down by David Guetta
    {"qid": "5", "docno": "85_8", "label": 0},     # Demons by Imagine Dragons
    {"qid": "5", "docno": "85_9", "label": 0},     # Radioactive by Imagine Dragons
])

qrels_pos = pd.DataFrame([
    # qid 2: 'chill relaxing'
    {"qid": "2", "docno": "9630_0", "label": 0},   # Feel Good Inc by Gorillaz
    {"qid": "2", "docno": "9630_1", "label": 1},   # Clint Eastwood by Gorillaz
    {"qid": "2", "docno": "9630_2", "label": 1},   # Saturnz Barz by Gorillaz
    {"qid": "2", "docno": "9630_3", "label": 0},   # Icky Thump by The White Stripes
    {"qid": "2", "docno": "9630_4", "label": 0},   # Seven Nation Army by The White Stripes
    {"qid": "2", "docno": "9630_6", "label": 0},   # Paint It Black by The Rolling Stones
    {"qid": "2", "docno": "9630_7", "label": 0},   # Welcome To The Jungle by Guns N' Roses
    {"qid": "2", "docno": "9630_9", "label": 0},   # Sympathy For The Devil by The Rolling Stones
    {"qid": "2", "docno": "9630_10", "label": 0},  # Blue Orchid by The White Stripes
    {"qid": "2", "docno": "9630_12", "label": 0},  # American Idiot by Green Day

    # qid 3: 'happy feel good vibes'
    {"qid": "3", "docno": "7615_1", "label": 1},   # Sex With Me by Rihanna
    {"qid": "3", "docno": "7615_3", "label": 1},   # You Was Right by Lil Uzi Vert
    {"qid": "3", "docno": "7615_4", "label": 1},   # LUV by Tory Lanez
    {"qid": "3", "docno": "7615_5", "label": 1},   # Broccoli by DRAM
    {"qid": "3", "docno": "7615_7", "label": 1},   # The Way by Kehlani
    {"qid": "3", "docno": "7615_9", "label": 0},   # Uber Everywhere by MadeinTYO
    {"qid": "3", "docno": "7615_10", "label": 0},  # Drama by Roy Woods
    {"qid": "3", "docno": "7615_11", "label": 1},  # Weekend by Mac Miller
    {"qid": "3", "docno": "7615_12", "label": 1},  # Free Lunch by Isaiah Rashad
    {"qid": "3", "docno": "7615_13", "label": 0},  # R.I.P. Kevin Miller by Isaiah Rashad

    # qid 4: 'happy dance'
    {"qid": "4", "docno": "1234_0", "label": 1},   # Titanium by David Guetta
    {"qid": "4", "docno": "1234_1", "label": 1},   # Paris by David Guetta
    {"qid": "4", "docno": "1234_2", "label": 1},   # Give Me All Your Luvin' by Madonna
    {"qid": "4", "docno": "1234_3", "label": 1},   # Nah Neh Nah by Rico Bernasconi
    {"qid": "4", "docno": "1234_5", "label": 0},   # I Like That by Richard Vission
    {"qid": "4", "docno": "1234_6", "label": 0},   # Naked by DEV
    {"qid": "4", "docno": "1234_7", "label": 1},   # Respect by Melanie Amaro
    {"qid": "4", "docno": "1234_9", "label": 1},   # Never Forget You by Lupe Fiasco
    {"qid": "4", "docno": "1234_10", "label": 0},  # Changed The Way You Kiss Me
    {"qid": "4", "docno": "1234_11", "label": 1},  # Be Your Freak by Kenny Dope

    # qid 5: 'energetic gym'
    {"qid": "5", "docno": "85_0", "label": 1},     # Me, Myself & I by G-Eazy
    {"qid": "5", "docno": "85_1", "label": 1},     # Adventure Of A Lifetime by Coldplay
    {"qid": "5", "docno": "85_2", "label": 1},     # Hymn For The Weekend by Coldplay
    {"qid": "5", "docno": "85_3", "label": 1},     # Up&Up by Coldplay
    {"qid": "5", "docno": "85_4", "label": 1},     # A Head Full Of Dreams by Coldplay
    {"qid": "5", "docno": "85_5", "label": 1},     # A Sky Full of Stars by Coldplay
    {"qid": "5", "docno": "85_6", "label": 1},     # Paradise by Coldplay
    {"qid": "5", "docno": "85_7", "label": 1},     # Shot Me Down by David Guetta
    {"qid": "5", "docno": "85_8", "label": 0},     # Demons by Imagine Dragons
    {"qid": "5", "docno": "85_9", "label": 0},     # Radioactive by Imagine Dragons
])


qrels_neg = pd.DataFrame([
    # qid 1: 'sad breakup'
    {"qid": "1", "docno": "1223_0", "label": 1},
    {"qid": "1", "docno": "1223_1", "label": 1},
    {"qid": "1", "docno": "1223_2", "label": 1},
    {"qid": "1", "docno": "1223_3", "label": 1},
    {"qid": "1", "docno": "1223_4", "label": 1},
    {"qid": "1", "docno": "1223_5", "label": 1},
    {"qid": "1", "docno": "1223_6", "label": 1},
    {"qid": "1", "docno": "1223_8", "label": 1},
    {"qid": "1", "docno": "1223_9", "label": 1},
    {"qid": "1", "docno": "1223_10", "label": 1},

    # Query: 'depressed walk' - qid: 6
    {"qid": "6", "docno": "2857_1",  "label": 1},  # Supermodel - SZA
    {"qid": "6", "docno": "2857_2",  "label": 1},  # Cool - Zack Villere
    {"qid": "6", "docno": "2857_3",  "label": 0},  # I. the worst guys - Childish Gambino
    {"qid": "6", "docno": "2857_4",  "label": 1},  # November - Tyler, The Creator
    {"qid": "6", "docno": "2857_5",  "label": 1},  # See You Again - Tyler, The Creator
    {"qid": "6", "docno": "2857_7",  "label": 0},  # IV. sweatpants - Childish Gambino
    {"qid": "6", "docno": "2857_8",  "label": 1},  # Redbone - Childish Gambino
    {"qid": "6", "docno": "2857_10", "label": 0},  # Killamonjaro - KILLY
    {"qid": "6", "docno": "2857_11", "label": 1},  # Dreaming Of You - Selena
    {"qid": "6", "docno": "2857_13", "label": 1},  # Breakin' My Heart (Pretty Brown Eyes) - Mint Condition

    # Query: 'angry run' - qid: 7
    {"qid": "7", "docno": "1652_0",  "label": 0},  # Best Friend's Brother - Victorious Cast
    {"qid": "7", "docno": "1652_1",  "label": 1},  # Lips Are Movin - Meghan Trainor
    {"qid": "7", "docno": "1652_2",  "label": 0},  # All About That Bass - Meghan Trainor
    {"qid": "7", "docno": "1652_3",  "label": 0},  # Carry on Wayward Son - Kansas
    {"qid": "7", "docno": "1652_5",  "label": 1},  # Eye of the Tiger - Survivor
    {"qid": "7", "docno": "1652_6",  "label": 0},  # Shut Up and Dance - WALK THE MOON
    {"qid": "7", "docno": "1652_7",  "label": 0},  # Wild Card - Hunter Hayes
    {"qid": "7", "docno": "1652_8",  "label": 0},  # Tattoo - Hunter Hayes
    {"qid": "7", "docno": "1652_9",  "label": 0},  # Heartbeat Song - Kelly Clarkson
    {"qid": "7", "docno": "1652_10", "label": 0},  # Dear Future Husband - Meghan Trainor


])

# evaluate
metrics = [pt.measures.P@10, pt.measures.R@10, pt.measures.nDCG@10]
exp = pt.Experiment([results], queries, qrels, metrics, names=["BM25"])
print(exp)

print("\n Negative")
exp_neg = pt.Experiment([results], queries, qrels_neg, metrics, names=["BM25"])
print(exp_neg)

print("\n Positive")
exp_pos = pt.Experiment([results], queries, qrels_pos, metrics, names=["BM25"])
print(exp_pos)

