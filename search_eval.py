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
    {"qid": "1", "query": "sad breakup", "class": "neg"}, # neg
    {"qid": "2", "query": "chill relaxing", "class": "pos"}, # pos
    {"qid": "3", "query": "happy feel good vibes", "class": "pos"},
    {"qid": "4", "query": "happy dance", "class": "pos"}, # pos
    {"qid": "5", "query": "energetic gym", "class": "pos"}, # pos
    {"qid": "6", "query": "depressed walk", "class": "neg"}, # neg
    {"qid": "7", "query": "angry run", "class": "neg"}, # neg
    {"qid": "8", "query": "heartbroken lonely", "class": "neg"}, # neg
    {"qid": "9", "query": "peaceful morning", "class": "pos"}, # pos
    {"qid": "10", "query": "stress study", "class": "neg"}, # neg
    {"qid": "11", "query": "anxiety", "class": "neg"}, # neg
    {"qid": "12", "query": "joyous celebration", "class": "pos"} # pos
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
    # qid 1: 'sad breakup' - neg
    {"qid": "1", "docno": "1223_0", "label": 1, "class": "neg"},   # Steal My Girl by One Direction
    {"qid": "1", "docno": "1223_1", "label": 1, "class": "neg"},   # Wait On Me by Rixton
    {"qid": "1", "docno": "1223_2", "label": 1, "class": "neg"},   # Habits (Stay High) by Tove Lo
    {"qid": "1", "docno": "1223_3", "label": 1, "class": "neg"},   # Don't by Ed Sheeran
    {"qid": "1", "docno": "1223_4", "label": 1, "class": "neg"},   # Miss Movin' On by Fifth Harmony
    {"qid": "1", "docno": "1223_5", "label": 1, "class": "neg"},   # When I Was Your Man by Bruno Mars
    {"qid": "1", "docno": "1223_6", "label": 1, "class": "neg"},   # Forget You by CeeLo Green
    {"qid": "1", "docno": "1223_8", "label": 1, "class": "neg"},   # We Are Never Ever Getting Back Together by Taylor Swift
    {"qid": "1", "docno": "1223_9", "label": 1, "class": "neg"},   # Since U Been Gone by Kelly Clarkson
    {"qid": "1", "docno": "1223_10", "label": 1, "class": "neg"},  # Stronger by Kelly Clarkson

    # qid 2: 'chill relaxing' - pos
    {"qid": "2", "docno": "9630_0", "label": 0, "class": "pos"},   # Feel Good Inc by Gorillaz
    {"qid": "2", "docno": "9630_1", "label": 1, "class": "pos"},   # Clint Eastwood by Gorillaz
    {"qid": "2", "docno": "9630_2", "label": 1, "class": "pos"},   # Saturnz Barz by Gorillaz
    {"qid": "2", "docno": "9630_3", "label": 0, "class": "pos"},   # Icky Thump by The White Stripes
    {"qid": "2", "docno": "9630_4", "label": 0, "class": "pos"},   # Seven Nation Army by The White Stripes
    {"qid": "2", "docno": "9630_6", "label": 0, "class": "pos"},   # Paint It Black by The Rolling Stones
    {"qid": "2", "docno": "9630_7", "label": 0, "class": "pos"},   # Welcome To The Jungle by Guns N' Roses
    {"qid": "2", "docno": "9630_9", "label": 0, "class": "pos"},   # Sympathy For The Devil by The Rolling Stones
    {"qid": "2", "docno": "9630_10", "label": 0, "class": "pos"},  # Blue Orchid by The White Stripes
    {"qid": "2", "docno": "9630_12", "label": 0, "class": "pos"},  # American Idiot by Green Day

    # qid 3: 'happy feel good vibes' - pos
    {"qid": "3", "docno": "7615_1", "label": 1, "class": "pos"},   # Sex With Me by Rihanna
    {"qid": "3", "docno": "7615_3", "label": 1, "class": "pos"},   # You Was Right by Lil Uzi Vert
    {"qid": "3", "docno": "7615_4", "label": 1, "class": "pos"},   # LUV by Tory Lanez
    {"qid": "3", "docno": "7615_5", "label": 1, "class": "pos"},   # Broccoli by DRAM
    {"qid": "3", "docno": "7615_7", "label": 1, "class": "pos"},   # The Way by Kehlani
    {"qid": "3", "docno": "7615_9", "label": 0, "class": "pos"},   # Uber Everywhere by MadeinTYO
    {"qid": "3", "docno": "7615_10", "label": 0, "class": "pos"},  # Drama by Roy Woods
    {"qid": "3", "docno": "7615_11", "label": 1, "class": "pos"},  # Weekend by Mac Miller
    {"qid": "3", "docno": "7615_12", "label": 1, "class": "pos"},  # Free Lunch by Isaiah Rashad
    {"qid": "3", "docno": "7615_13", "label": 0, "class": "pos"},  # R.I.P. Kevin Miller by Isaiah Rashad

    # qid 4: 'happy dance' - pos
    {"qid": "4", "docno": "1234_0", "label": 1, "class": "pos"},   # Titanium by David Guetta
    {"qid": "4", "docno": "1234_1", "label": 1, "class": "pos"},   # Paris by David Guetta
    {"qid": "4", "docno": "1234_2", "label": 1, "class": "pos"},   # Give Me All Your Luvin' by Madonna
    {"qid": "4", "docno": "1234_3", "label": 1, "class": "pos"},   # Nah Neh Nah by Rico Bernasconi
    {"qid": "4", "docno": "1234_5", "label": 0, "class": "pos"},   # I Like That by Richard Vission
    {"qid": "4", "docno": "1234_6", "label": 0, "class": "pos"},   # Naked by DEV
    {"qid": "4", "docno": "1234_7", "label": 1, "class": "pos"},   # Respect by Melanie Amaro
    {"qid": "4", "docno": "1234_9", "label": 1, "class": "pos"},   # Never Forget You by Lupe Fiasco
    {"qid": "4", "docno": "1234_10", "label": 0, "class": "pos"},  # Changed The Way You Kiss Me
    {"qid": "4", "docno": "1234_11", "label": 1, "class": "pos"},  # Be Your Freak by Kenny Dope

    # qid 5: 'energetic gym' - pos
    {"qid": "5", "docno": "85_0", "label": 1, "class": "pos"},     # Me, Myself & I by G-Eazy
    {"qid": "5", "docno": "85_1", "label": 1, "class": "pos"},     # Adventure Of A Lifetime by Coldplay
    {"qid": "5", "docno": "85_2", "label": 1, "class": "pos"},     # Hymn For The Weekend by Coldplay
    {"qid": "5", "docno": "85_3", "label": 1, "class": "pos"},     # Up&Up by Coldplay
    {"qid": "5", "docno": "85_4", "label": 1, "class": "pos"},     # A Head Full Of Dreams by Coldplay
    {"qid": "5", "docno": "85_5", "label": 1, "class": "pos"},     # A Sky Full of Stars by Coldplay
    {"qid": "5", "docno": "85_6", "label": 1, "class": "pos"},     # Paradise by Coldplay
    {"qid": "5", "docno": "85_7", "label": 1, "class": "pos"},     # Shot Me Down by David Guetta
    {"qid": "5", "docno": "85_8", "label": 0, "class": "pos"},     # Demons by Imagine Dragons
    {"qid": "5", "docno": "85_9", "label": 0, "class": "pos"},     # Radioactive by Imagine Dragons

    # qid 6: 'depressed walk' - neg
    {"qid": "6", "docno": "2857_1", "label": 1, "class": "neg"},    # Supermodel by SZA
    {"qid": "6", "docno": "2857_2", "label": 1, "class": "neg"},    # Cool by Zack Villere
    {"qid": "6", "docno": "2857_3", "label": 1, "class": "neg"},    # I. the worst guys by Childish Gambino
    {"qid": "6", "docno": "2857_4", "label": 1, "class": "neg"},    # November by Tyler, The Creator
    {"qid": "6", "docno": "2857_5", "label": 1, "class": "neg"},    # See You Again by Tyler, The Creator
    {"qid": "6", "docno": "2857_7", "label": 1, "class": "neg"},    # IV. sweatpants by Childish Gambino
    {"qid": "6", "docno": "2857_8", "label": 1, "class": "neg"},    # Redbone by Childish Gambino
    {"qid": "6", "docno": "2857_10", "label": 0, "class": "neg"},   # Killamonjaro by KILLY
    {"qid": "6", "docno": "2857_11", "label": 0, "class": "neg"},   # Dreaming Of You by Selena
    {"qid": "6", "docno": "2857_13", "label": 1, "class": "neg"},   # Breakin' My Heart by Mint Condition

    # qid 7: 'angry run' - neg
    {"qid": "7", "docno": "1652_0", "label": 1, "class": "neg"},    # Best Friend's Brother by Victorious Cast
    {"qid": "7", "docno": "1652_1", "label": 0, "class": "neg"},    # Lips Are Movin by Meghan Trainor
    {"qid": "7", "docno": "1652_2", "label": 0, "class": "neg"},    # All About That Bass by Meghan Trainor
    {"qid": "7", "docno": "1652_3", "label": 0, "class": "neg"},    # Carry on Wayward Son by Kansas
    {"qid": "7", "docno": "1652_5", "label": 1, "class": "neg"},    # Eye of the Tiger by Survivor
    {"qid": "7", "docno": "1652_6", "label": 0, "class": "neg"},    # Shut Up and Dance by WALK THE MOON
    {"qid": "7", "docno": "1652_7", "label": 0, "class": "neg"},    # Wild Card by Hunter Hayes
    {"qid": "7", "docno": "1652_8", "label": 0, "class": "neg"},    # Tattoo by Hunter Hayes
    {"qid": "7", "docno": "1652_9", "label": 0, "class": "neg"},    # Heartbeat Song by Kelly Clarkson
    {"qid": "7", "docno": "1652_10", "label": 0, "class": "neg"},   # Dear Future Husband by Meghan Trainor

    # qid 8: 'heartbroken lonely' - neg
    {"qid": "8", "docno": "10872_0", "label": 0, "class": "neg"},   # Night Fever by Bee Gees
    {"qid": "8", "docno": "10872_1", "label": 0, "class": "neg"},   # Living On Video by Trans-X
    {"qid": "8", "docno": "10872_3", "label": 0, "class": "neg"},   # Babe, We're Gonna Love Tonight by Lime
    {"qid": "8", "docno": "10872_4", "label": 1, "class": "neg"},   # Danger by The Flirts
    {"qid": "8", "docno": "10872_6", "label": 0, "class": "neg"},  # Tainted Love by Soft Cell
    {"qid": "8", "docno": "10872_7", "label": 1, "class": "neg"},   # Egypt Egypt by The Egyptian Lover
    {"qid": "8", "docno": "10872_9", "label": 1, "class": "neg"},   # Cool It Now by New Edition
    {"qid": "8", "docno": "10872_10", "label": 1, "class": "neg"},  # Mandolay by Gary's Gang
    {"qid": "8", "docno": "10872_11", "label": 1, "class": "neg"},  # Heaven Must Be Missing An Angel by Tavares
    {"qid": "8", "docno": "2516_0", "label": 0, "class": "neg"},   # I'm Coming Out by Diana Ross

    # qid 9: 'peaceful morning' - pos
    {"qid": "9", "docno": "33_0", "label": 1, "class": "pos"},      # Chan Chan by Buena Vista Social Club
    {"qid": "9", "docno": "33_1", "label": 1, "class": "pos"},      # De Camino a La Vereda by Buena Vista Social Club
    {"qid": "9", "docno": "33_2", "label": 1, "class": "pos"},     # Isn't She Lovely by Stevie Wonder
    {"qid": "9", "docno": "33_3", "label": 1, "class": "pos"},     # You Are The Sunshine Of My Life by Stevie Wonder
    {"qid": "9", "docno": "33_4", "label": 1, "class": "pos"},      # September by Earth, Wind & Fire
    {"qid": "9", "docno": "33_5", "label": 0, "class": "pos"},      # Love's Holiday by Earth, Wind & Fire
    {"qid": "9", "docno": "33_8", "label": 1, "class": "pos"},      # Momento by Bebel Gilberto
    {"qid": "9", "docno": "33_9", "label": 0, "class": "pos"},      # I Need a Dollar by Aloe Blacc
    {"qid": "9", "docno": "33_10", "label": 1, "class": "pos"},     # Green Lights by Aloe Blacc
    {"qid": "9", "docno": "33_11", "label": 0, "class": "pos"},     # You Make Me Smile by Aloe Blacc

    # qid 10: 'stress study' - neg
    {"qid": "10", "docno": "1079_1", "label": 1, "class": "neg"},  # Moonlight Sonata by Michael Silverman
    {"qid": "10", "docno": "1079_2", "label": 0, "class": "neg"},  # Sonata Pathetique by Michael Silverman
    {"qid": "10", "docno": "1079_3", "label": 0, "class": "neg"},  # Snow by Michael Silverman
    {"qid": "10", "docno": "1079_4", "label": 1, "class": "neg"},  # Pachelbel: Canon In D by Michael Silverman
    {"qid": "10", "docno": "1079_5", "label": 0, "class": "neg"},  # Simple Gifts by Michael Silverman
    {"qid": "10", "docno": "1079_6", "label": 1, "class": "neg"},  # Sentimental Summer by Michael Silverman
    {"qid": "10", "docno": "1079_7", "label": 1, "class": "neg"},  # Sentimental Piano Music by Michael Silverman
    {"qid": "10", "docno": "1079_8", "label": 1, "class": "neg"},  # Fall by Classical Study Music
    {"qid": "10", "docno": "1079_10", "label": 0, "class": "neg"}, # Johann's Dream by Classical Study Music
    {"qid": "10", "docno": "1079_11", "label": 1, "class": "neg"}, # Adagio by Michael Silverman

    # qid 11: 'anxiety' - neg
    {"qid": "11", "docno": "3288_0", "label": 1, "class": "neg"},  # Beach Baby by Bon Iver
    {"qid": "11", "docno": "3288_1", "label": 1, "class": "neg"},  # Perth by Bon Iver
    {"qid": "11", "docno": "3288_2", "label": 1, "class": "neg"},  # Holocene by Bon Iver
    {"qid": "11", "docno": "3288_3", "label": 1, "class": "neg"},  # Michicant by Bon Iver
    {"qid": "11", "docno": "3288_4", "label": 1, "class": "neg"},  # Wash. by Bon Iver
    {"qid": "11", "docno": "3288_5", "label": 1, "class": "neg"},  # Blindsided by Bon Iver
    {"qid": "11", "docno": "3288_6", "label": 1, "class": "neg"},  # Sunshine on My Back by The National
    {"qid": "11", "docno": "3288_7", "label": 0, "class": "neg"},  # Hard To Find by The National
    {"qid": "11", "docno": "3288_8", "label": 0, "class": "neg"},  # I Should Live in Salt by The National
    {"qid": "11", "docno": "3288_9", "label": 1, "class": "neg"},  # Demons by The National

    # qid 12: 'joy celebration' - pos
    {"qid": "12", "docno": "2805_0", "label": 0, "class": "pos"},   # Takes My Body Higher by Shoffy
    {"qid": "12", "docno": "2805_2", "label": 1, "class": "pos"},   # Never Be Like You by Flume
    {"qid": "12", "docno": "2805_3", "label": 0, "class": "pos"},   # Say It by Flume
    {"qid": "12", "docno": "2805_4", "label": 1, "class": "pos"},   # Let Me Hold You by Dante Klein
    {"qid": "12", "docno": "2805_5", "label": 1, "class": "pos"},   # The Ocean by Mike Perry
    {"qid": "12", "docno": "2805_6", "label": 1, "class": "pos"},   # Gold by Kiiara
    {"qid": "12", "docno": "2805_7", "label": 0, "class": "pos"},   # Equation by Camille
    {"qid": "12", "docno": "2805_8", "label": 1, "class": "pos"},   # Final Song by MØ
    {"qid": "12", "docno": "2805_9", "label": 1, "class": "pos"},   # Blue Sky by CAZZETTE
    {"qid": "12", "docno": "2805_10", "label": 1, "class": "pos"},  # Everyday by A$AP Rocky
])

qrels_neg = pd.DataFrame([
    # qid 1: 'sad breakup' - neg
    {"qid": "1", "docno": "1223_0", "label": 1, "class": "neg"},   # Steal My Girl by One Direction
    {"qid": "1", "docno": "1223_1", "label": 1, "class": "neg"},   # Wait On Me by Rixton
    {"qid": "1", "docno": "1223_2", "label": 1, "class": "neg"},   # Habits (Stay High) by Tove Lo
    {"qid": "1", "docno": "1223_3", "label": 1, "class": "neg"},   # Don't by Ed Sheeran
    {"qid": "1", "docno": "1223_4", "label": 1, "class": "neg"},   # Miss Movin' On by Fifth Harmony
    {"qid": "1", "docno": "1223_5", "label": 1, "class": "neg"},   # When I Was Your Man by Bruno Mars
    {"qid": "1", "docno": "1223_6", "label": 1, "class": "neg"},   # Forget You by CeeLo Green
    {"qid": "1", "docno": "1223_8", "label": 1, "class": "neg"},   # We Are Never Ever Getting Back Together by Taylor Swift
    {"qid": "1", "docno": "1223_9", "label": 1, "class": "neg"},   # Since U Been Gone by Kelly Clarkson
    {"qid": "1", "docno": "1223_10", "label": 1, "class": "neg"},  # Stronger by Kelly Clarkson


    # qid 6: 'depressed walk' - neg
    {"qid": "6", "docno": "2857_1", "label": 1, "class": "neg"},    # Supermodel by SZA
    {"qid": "6", "docno": "2857_2", "label": 1, "class": "neg"},    # Cool by Zack Villere
    {"qid": "6", "docno": "2857_3", "label": 1, "class": "neg"},    # I. the worst guys by Childish Gambino
    {"qid": "6", "docno": "2857_4", "label": 1, "class": "neg"},    # November by Tyler, The Creator
    {"qid": "6", "docno": "2857_5", "label": 1, "class": "neg"},    # See You Again by Tyler, The Creator
    {"qid": "6", "docno": "2857_7", "label": 1, "class": "neg"},    # IV. sweatpants by Childish Gambino
    {"qid": "6", "docno": "2857_8", "label": 1, "class": "neg"},    # Redbone by Childish Gambino
    {"qid": "6", "docno": "2857_10", "label": 0, "class": "neg"},   # Killamonjaro by KILLY
    {"qid": "6", "docno": "2857_11", "label": 0, "class": "neg"},   # Dreaming Of You by Selena
    {"qid": "6", "docno": "2857_13", "label": 1, "class": "neg"},   # Breakin' My Heart by Mint Condition

    # qid 7: 'angry run' - neg
    {"qid": "7", "docno": "1652_0", "label": 1, "class": "neg"},    # Best Friend's Brother by Victorious Cast
    {"qid": "7", "docno": "1652_1", "label": 0, "class": "neg"},    # Lips Are Movin by Meghan Trainor
    {"qid": "7", "docno": "1652_2", "label": 0, "class": "neg"},    # All About That Bass by Meghan Trainor
    {"qid": "7", "docno": "1652_3", "label": 0, "class": "neg"},    # Carry on Wayward Son by Kansas
    {"qid": "7", "docno": "1652_5", "label": 1, "class": "neg"},    # Eye of the Tiger by Survivor
    {"qid": "7", "docno": "1652_6", "label": 0, "class": "neg"},    # Shut Up and Dance by WALK THE MOON
    {"qid": "7", "docno": "1652_7", "label": 0, "class": "neg"},    # Wild Card by Hunter Hayes
    {"qid": "7", "docno": "1652_8", "label": 0, "class": "neg"},    # Tattoo by Hunter Hayes
    {"qid": "7", "docno": "1652_9", "label": 0, "class": "neg"},    # Heartbeat Song by Kelly Clarkson
    {"qid": "7", "docno": "1652_10", "label": 0, "class": "neg"},   # Dear Future Husband by Meghan Trainor

    # qid 8: 'heartbroken lonely' - neg
    {"qid": "8", "docno": "10872_0", "label": 0, "class": "neg"},   # Night Fever by Bee Gees
    {"qid": "8", "docno": "10872_1", "label": 0, "class": "neg"},   # Living On Video by Trans-X
    {"qid": "8", "docno": "10872_3", "label": 0, "class": "neg"},   # Babe, We're Gonna Love Tonight by Lime
    {"qid": "8", "docno": "10872_4", "label": 1, "class": "neg"},   # Danger by The Flirts
    {"qid": "8", "docno": "10872_6", "label": 0, "class": "neg"},  # Tainted Love by Soft Cell
    {"qid": "8", "docno": "10872_7", "label": 1, "class": "neg"},   # Egypt Egypt by The Egyptian Lover
    {"qid": "8", "docno": "10872_9", "label": 1, "class": "neg"},   # Cool It Now by New Edition
    {"qid": "8", "docno": "10872_10", "label": 1, "class": "neg"},  # Mandolay by Gary's Gang
    {"qid": "8", "docno": "10872_11", "label": 1, "class": "neg"},  # Heaven Must Be Missing An Angel by Tavares
    {"qid": "8", "docno": "2516_0", "label": 0, "class": "neg"},   # I'm Coming Out by Diana Ross

    # qid 10: 'stress study' - neg
    {"qid": "10", "docno": "1079_1", "label": 1, "class": "neg"},  # Moonlight Sonata by Michael Silverman
    {"qid": "10", "docno": "1079_2", "label": 0, "class": "neg"},  # Sonata Pathetique by Michael Silverman
    {"qid": "10", "docno": "1079_3", "label": 0, "class": "neg"},  # Snow by Michael Silverman
    {"qid": "10", "docno": "1079_4", "label": 1, "class": "neg"},  # Pachelbel: Canon In D by Michael Silverman
    {"qid": "10", "docno": "1079_5", "label": 0, "class": "neg"},  # Simple Gifts by Michael Silverman
    {"qid": "10", "docno": "1079_6", "label": 1, "class": "neg"},  # Sentimental Summer by Michael Silverman
    {"qid": "10", "docno": "1079_7", "label": 1, "class": "neg"},  # Sentimental Piano Music by Michael Silverman
    {"qid": "10", "docno": "1079_8", "label": 1, "class": "neg"},  # Fall by Classical Study Music
    {"qid": "10", "docno": "1079_10", "label": 0, "class": "neg"}, # Johann's Dream by Classical Study Music
    {"qid": "10", "docno": "1079_11", "label": 1, "class": "neg"}, # Adagio by Michael Silverman

    # qid 11: 'anxiety' - neg
    {"qid": "11", "docno": "3288_0", "label": 1, "class": "neg"},  # Beach Baby by Bon Iver
    {"qid": "11", "docno": "3288_1", "label": 1, "class": "neg"},  # Perth by Bon Iver
    {"qid": "11", "docno": "3288_2", "label": 1, "class": "neg"},  # Holocene by Bon Iver
    {"qid": "11", "docno": "3288_3", "label": 1, "class": "neg"},  # Michicant by Bon Iver
    {"qid": "11", "docno": "3288_4", "label": 1, "class": "neg"},  # Wash. by Bon Iver
    {"qid": "11", "docno": "3288_5", "label": 1, "class": "neg"},  # Blindsided by Bon Iver
    {"qid": "11", "docno": "3288_6", "label": 1, "class": "neg"},  # Sunshine on My Back by The National
    {"qid": "11", "docno": "3288_7", "label": 0, "class": "neg"},  # Hard To Find by The National
    {"qid": "11", "docno": "3288_8", "label": 0, "class": "neg"},  # I Should Live in Salt by The National
    {"qid": "11", "docno": "3288_9", "label": 1, "class": "neg"},  # Demons by The National
])

qrels_pos = pd.DataFrame([
    # qid 2: 'chill relaxing' - pos
    {"qid": "2", "docno": "9630_0", "label": 0, "class": "pos"},   # Feel Good Inc by Gorillaz
    {"qid": "2", "docno": "9630_1", "label": 1, "class": "pos"},   # Clint Eastwood by Gorillaz
    {"qid": "2", "docno": "9630_2", "label": 0, "class": "pos"},   # Saturnz Barz by Gorillaz
    {"qid": "2", "docno": "9630_3", "label": 0, "class": "pos"},   # Icky Thump by The White Stripes
    {"qid": "2", "docno": "9630_4", "label": 0, "class": "pos"},   # Seven Nation Army by The White Stripes
    {"qid": "2", "docno": "9630_6", "label": 0, "class": "pos"},   # Paint It Black by The Rolling Stones
    {"qid": "2", "docno": "9630_7", "label": 0, "class": "pos"},   # Welcome To The Jungle by Guns N' Roses
    {"qid": "2", "docno": "9630_9", "label": 0, "class": "pos"},   # Sympathy For The Devil by The Rolling Stones
    {"qid": "2", "docno": "9630_10", "label": 0, "class": "pos"},  # Blue Orchid by The White Stripes
    {"qid": "2", "docno": "9630_12", "label": 0, "class": "pos"},  # American Idiot by Green Day

    # qid 4: 'happy dance' - pos
    {"qid": "4", "docno": "1234_0", "label": 1, "class": "pos"},   # Titanium by David Guetta
    {"qid": "4", "docno": "1234_1", "label": 1, "class": "pos"},   # Paris by David Guetta
    {"qid": "4", "docno": "1234_2", "label": 1, "class": "pos"},   # Give Me All Your Luvin' by Madonna
    {"qid": "4", "docno": "1234_3", "label": 1, "class": "pos"},   # Nah Neh Nah by Rico Bernasconi
    {"qid": "4", "docno": "1234_5", "label": 0, "class": "pos"},   # I Like That by Richard Vission
    {"qid": "4", "docno": "1234_6", "label": 0, "class": "pos"},   # Naked by DEV
    {"qid": "4", "docno": "1234_7", "label": 1, "class": "pos"},   # Respect by Melanie Amaro
    {"qid": "4", "docno": "1234_9", "label": 1, "class": "pos"},   # Never Forget You by Lupe Fiasco
    {"qid": "4", "docno": "1234_10", "label": 0, "class": "pos"},  # Changed The Way You Kiss Me
    {"qid": "4", "docno": "1234_11", "label": 1, "class": "pos"},  # Be Your Freak by Kenny Dope

    # qid 3: 'happy feel good vibes' - pos
    {"qid": "3", "docno": "7615_1", "label": 1, "class": "pos"},   # Sex With Me by Rihanna
    {"qid": "3", "docno": "7615_3", "label": 1, "class": "pos"},   # You Was Right by Lil Uzi Vert
    {"qid": "3", "docno": "7615_4", "label": 1, "class": "pos"},   # LUV by Tory Lanez
    {"qid": "3", "docno": "7615_5", "label": 1, "class": "pos"},   # Broccoli by DRAM
    {"qid": "3", "docno": "7615_7", "label": 1, "class": "pos"},   # The Way by Kehlani
    {"qid": "3", "docno": "7615_9", "label": 0, "class": "pos"},   # Uber Everywhere by MadeinTYO
    {"qid": "3", "docno": "7615_10", "label": 0, "class": "pos"},  # Drama by Roy Woods
    {"qid": "3", "docno": "7615_11", "label": 1, "class": "pos"},  # Weekend by Mac Miller
    {"qid": "3", "docno": "7615_12", "label": 1, "class": "pos"},  # Free Lunch by Isaiah Rashad
    {"qid": "3", "docno": "7615_13", "label": 0, "class": "pos"},  # R.I.P. Kevin Miller by Isaiah Rashad


    # qid 5: 'energetic gym' - pos
    {"qid": "5", "docno": "85_0", "label": 1, "class": "pos"},     # Me, Myself & I by G-Eazy
    {"qid": "5", "docno": "85_1", "label": 1, "class": "pos"},     # Adventure Of A Lifetime by Coldplay
    {"qid": "5", "docno": "85_2", "label": 1, "class": "pos"},     # Hymn For The Weekend by Coldplay
    {"qid": "5", "docno": "85_3", "label": 1, "class": "pos"},     # Up&Up by Coldplay
    {"qid": "5", "docno": "85_4", "label": 1, "class": "pos"},     # A Head Full Of Dreams by Coldplay
    {"qid": "5", "docno": "85_5", "label": 1, "class": "pos"},     # A Sky Full of Stars by Coldplay
    {"qid": "5", "docno": "85_6", "label": 1, "class": "pos"},     # Paradise by Coldplay
    {"qid": "5", "docno": "85_7", "label": 1, "class": "pos"},     # Shot Me Down by David Guetta
    {"qid": "5", "docno": "85_8", "label": 0, "class": "pos"},     # Demons by Imagine Dragons
    {"qid": "5", "docno": "85_9", "label": 0, "class": "pos"},     # Radioactive by Imagine Dragons

    # qid 9: 'peaceful morning' - pos
    {"qid": "9", "docno": "33_0", "label": 1, "class": "pos"},      # Chan Chan by Buena Vista Social Club
    {"qid": "9", "docno": "33_1", "label": 1, "class": "pos"},      # De Camino a La Vereda by Buena Vista Social Club
    {"qid": "9", "docno": "33_2", "label": 1, "class": "pos"},     # Isn't She Lovely by Stevie Wonder
    {"qid": "9", "docno": "33_3", "label": 1, "class": "pos"},     # You Are The Sunshine Of My Life by Stevie Wonder
    {"qid": "9", "docno": "33_4", "label": 1, "class": "pos"},      # September by Earth, Wind & Fire
    {"qid": "9", "docno": "33_5", "label": 0, "class": "pos"},      # Love's Holiday by Earth, Wind & Fire
    {"qid": "9", "docno": "33_8", "label": 1, "class": "pos"},      # Momento by Bebel Gilberto
    {"qid": "9", "docno": "33_9", "label": 0, "class": "pos"},      # I Need a Dollar by Aloe Blacc
    {"qid": "9", "docno": "33_10", "label": 1, "class": "pos"},     # Green Lights by Aloe Blacc
    {"qid": "9", "docno": "33_11", "label": 0, "class": "pos"},     # You Make Me Smile by Aloe Blacc

    # qid 12: 'joy celebration' - pos
    {"qid": "12", "docno": "2805_0", "label": 0, "class": "pos"},   # Takes My Body Higher by Shoffy
    {"qid": "12", "docno": "2805_2", "label": 1, "class": "pos"},   # Never Be Like You by Flume
    {"qid": "12", "docno": "2805_3", "label": 0, "class": "pos"},   # Say It by Flume
    {"qid": "12", "docno": "2805_4", "label": 1, "class": "pos"},   # Let Me Hold You by Dante Klein
    {"qid": "12", "docno": "2805_5", "label": 1, "class": "pos"},   # The Ocean by Mike Perry
    {"qid": "12", "docno": "2805_6", "label": 1, "class": "pos"},   # Gold by Kiiara
    {"qid": "12", "docno": "2805_7", "label": 0, "class": "pos"},   # Equation by Camille
    {"qid": "12", "docno": "2805_8", "label": 1, "class": "pos"},   # Final Song by MØ
    {"qid": "12", "docno": "2805_9", "label": 1, "class": "pos"},   # Blue Sky by CAZZETTE
    {"qid": "12", "docno": "2805_10", "label": 1, "class": "pos"},  # Everyday by A$AP Rocky
])

# evaluate
metrics = [pt.measures.P@10, pt.measures.R@10, pt.measures.nDCG@10]
exp = pt.Experiment([results], queries, qrels, metrics, names=["BM25"])
print(exp)

print("\n Negative")
exp_neg = pt.Experiment([results], queries, qrels[qrels["class"] == "neg"], metrics, names=["BM25"])
print(exp_neg)

print("\n Positive")
exp_pos = pt.Experiment([results], queries, qrels[qrels["class"] == "pos"], metrics, names=["BM25"])
print(exp_pos)

