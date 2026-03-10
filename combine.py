import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import pandas as pd



#df_path = "mpd_data/mpd.slice.0-999.json"
df_path = "mpd_data/"

random_seed = 7


def _clean(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s&']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


KEYWORD_THEMES: Dict[str, List[str]] = {
    "workout": ["workout", "gym", "run", "running", "cardio", "lift", "lifting", "training"],
    "chill": ["chill", "relax", "calm", "ambient", "study", "focus", "sleep"],
    "party": ["party", "dance", "club", "pregame", "turn up", "banger"],
    "hiphop": ["hip hop", "hip-hop", "rap", "trap"],
    "rock": ["rock", "metal", "punk", "grunge", "emo"],
    "pop": ["pop", "top hits", "hits"],
    "country": ["country"],
    "edm": ["edm", "electro", "house", "techno", "dubstep", "trance"],
    "rnb": ["r&b", "rnb", "soul"],
    "indie": ["indie", "alt", "alternative"],
    "disney": ["disney", "pixar"],
    "latin": ["latin", "reggaeton", "salsa", "bachata"],
    "throwback": ["throwback", "90s", "00s", "2000s", "80s", "70s", "oldies"],
    "sad": ["sad", "cry", "heartbreak", "breakup"],
    "happy": ["happy", "feel good", "good vibes"],
}


def infer_theme(playlist_name: str) -> str:
    n = _clean(playlist_name)
    for theme, keywords in KEYWORD_THEMES.items():
        if any(kw in n for kw in keywords):
            return theme
    return "misc"


@dataclass
class MockUser:
    user_id: str
    display_name: str
    preferred_themes: List[str]
    discovery_level: str
    seed_playlists: List[int]
    history_track_uris: List[str]


class Combine:
    def __init__(self, mpd_json_path: str = df_path, seed: int = random_seed):
        self.mpd_json_path = mpd_json_path
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.playlist_names: Dict[int, str] = {}
        self.playlist_tracks: Dict[int, List[str]] = {}
        self.track_meta: Dict[str, Dict[str, str]] = {}

        self.theme_to_pids: Dict[str, List[int]] = defaultdict(list)
        self.users: List[MockUser] = []
        self.track_to_id: Dict[str, int] = {}
        self.id_to_track: Dict[int, str] = {}
        self.R: Optional[csr_matrix] = None
        self.nn: Optional[NearestNeighbors] = None
        self.item_neighbors: Optional[List[List[Tuple[int, float]]]] = None

    def load_mpd(self, search_df: pd.DataFrame = None) -> None:
        import ast

        playlists = []
        for filename in os.listdir(self.mpd_json_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.mpd_json_path, filename)
                with open(filepath, "r") as f:
                    mpd = json.load(f)
                playlists.extend(mpd.get("playlists", []))

        if search_df is not None:
            seed_uris = set()
            for val in search_df["object"]:
                track = val if isinstance(val, dict) else ast.literal_eval(val)
                seed_uris.add(track.get("track_uri"))

            playlists = [
                pl for pl in playlists
                if any(t.get("track_uri") in seed_uris for t in pl.get("tracks", []))
            ]

        for pl in playlists:
            pid = int(pl["pid"])
            name = pl.get("name", "") or ""
            self.playlist_names[pid] = name

            seen = set()
            uris: List[str] = []
            for tr in pl.get("tracks", []):
                uri = tr.get("track_uri")
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                uris.append(uri)
                if uri not in self.track_meta:
                    self.track_meta[uri] = {
                        "track_uri": uri,
                        "track_name": tr.get("track_name", "") or "",
                        "artist_name": tr.get("artist_name", "") or "",
                        "album_name": tr.get("album_name", "") or "",
                    }
            self.playlist_tracks[pid] = uris

        self.theme_to_pids.clear()
        for pid, name in self.playlist_names.items():
            th = infer_theme(name)
            self.theme_to_pids[th].append(pid)

    def create_mock_users(
        self,
        n_users: int = 30,
        themes_per_user: Tuple[int, int] = (2, 4),
        playlists_per_user: Tuple[int, int] = (3, 8),
        max_history_tracks: int = 200,
        exclude_theme: str = "misc",
    ) -> List[MockUser]:
        """
        Create mock user profiles by sampling playlists from their preferred themes
        and using those playlists' tracks as the user's implicit history.
        """
        if not self.playlist_tracks:
            self.load_mpd()

        #theme_pool = [t for t in self.theme_to_pids.keys() if t != exclude_theme]
        #if not theme_pool:
        #    theme_pool = list(self.theme_to_pids.keys())
        theme_pool = [t for t in KEYWORD_THEMES.keys() if t != exclude_theme]

        users: List[MockUser] = []
        for i in range(n_users):
            user_id = f"u{i:03d}"
            preferred_n = random.randint(themes_per_user[0], themes_per_user[1])
            preferred_themes = random.sample(theme_pool, k=min(preferred_n, len(theme_pool)))

            candidate_pids: List[int] = []
            for th in preferred_themes:
                candidate_pids.extend(self.theme_to_pids.get(th, []))

            if not candidate_pids:
                candidate_pids = self.theme_to_pids.get("misc", [])[:]

            n_pl = random.randint(playlists_per_user[0], playlists_per_user[1])
            chosen = random.sample(candidate_pids, k=min(n_pl, len(candidate_pids)))

            history: List[str] = []
            for pid in chosen:
                history.extend(self.playlist_tracks.get(pid, []))

            history = list(dict.fromkeys(history))
            random.shuffle(history)
            history = history[:max_history_tracks]

            users.append(
                MockUser(
                    user_id=user_id,
                    display_name=f"User {i+1}",
                    preferred_themes=preferred_themes,
                    discovery_level=random.choice(["low", "medium", "high"]),
                    seed_playlists=chosen,
                    history_track_uris=history,
                )
            )

        self.users = users
        return users

    def build_user_item_matrix(self) -> csr_matrix:
        """
        Build sparse user-item matrix R (users x tracks).
        R[u, i] = 1 if user u has track i in history.
        """
        if not self.users:
            raise ValueError("No users yet. Call create_mock_users() first.")

        all_tracks = sorted({uri for u in self.users for uri in u.history_track_uris})
        self.track_to_id = {uri: idx for idx, uri in enumerate(all_tracks)}
        self.id_to_track = {idx: uri for uri, idx in self.track_to_id.items()}

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for u_idx, u in enumerate(self.users):
            for uri in u.history_track_uris:
                tid = self.track_to_id.get(uri)
                if tid is None:
                    continue
                rows.append(u_idx)
                cols.append(tid)
                data.append(1.0)

        R = csr_matrix((data, (rows, cols)), shape=(len(self.users), len(all_tracks)), dtype=np.float32)
        self.R = R
        return R

    def train_item_item_cf(self, k_neighbors: int = 50) -> None:
        """
        Train item-item cosine similarity model using kNN.
        """
        if self.R is None:
            raise ValueError("Matrix not built. Call build_user_item_matrix() first.")

        X_items = self.R.T.toarray()
        X_items = normalize(X_items, norm="l2", axis=1)

        n_items = X_items.shape[0]
        n_neighbors = min(k_neighbors + 1, n_items)

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(X_items)

        distances, indices = nn.kneighbors(X_items, return_distance=True)
        sims = 1.0 - distances

        item_neighbors: List[List[Tuple[int, float]]] = []
        for i in range(n_items):
            neigh: List[Tuple[int, float]] = []
            for j, s in zip(indices[i], sims[i]):
                if int(j) == i:
                    continue
                neigh.append((int(j), float(s)))
            item_neighbors.append(neigh)

        self.nn = nn
        self.item_neighbors = item_neighbors

    def recommend_for_user(self, user_id: str, k: int = 20) -> List[Dict]:
        """
        Recommend tracks for a user using item-item CF.
        """
        if self.item_neighbors is None:
            raise ValueError("Model not trained. Call train_item_item_cf() first.")

        u_idx = next((i for i, u in enumerate(self.users) if u.user_id == user_id), None)
        if u_idx is None:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_track_ids = {
            self.track_to_id[uri]
            for uri in self.users[u_idx].history_track_uris
            if uri in self.track_to_id
        }

        scores: Dict[int, float] = defaultdict(float)
        for tid in user_track_ids:
            for neigh_id, sim in self.item_neighbors[tid]:
                if neigh_id in user_track_ids:
                    continue
                scores[neigh_id] += sim

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        recs: List[Dict] = []

        for item_id, score in top:
            uri = self.id_to_track[item_id]
            meta = self.track_meta.get(
                uri,
                {
                    "track_uri": uri,
                    "track_name": "",
                    "artist_name": "",
                    "album_name": "",
                },
            )
            recs.append({**meta, "score": round(float(score), 5)})

        return recs

    def build(self, search_df: pd.DataFrame = None, n_users: int = 30, 
          k_neighbors: int = 50, max_history_tracks: int = 200) -> None:
        self.load_mpd(search_df=search_df)  # <-- pass search_df here
        self.create_mock_users(n_users=n_users, max_history_tracks=max_history_tracks)
        self.build_user_item_matrix()
        self.train_item_item_cf(k_neighbors=k_neighbors)

def evaluate(rec, k=10):
    hits, total = 0, 0

    for user in rec.users:
        tracks = user.history_track_uris
        if len(tracks) < 5:
            continue

        held_out = set(tracks[-2:])
        user.history_track_uris = tracks[:-2]  # temporarily hide last 2

        recs = {r["track_uri"] for r in rec.recommend_for_user(user.user_id, k=k)}
        hits += len(recs & held_out)
        total += len(held_out)

        user.history_track_uris = tracks  # restore

    print(f"Precision: {hits / (total / 2 * k):.4f}")
    print(f"Recall: {hits / total:.4f}")

if __name__ == "__main__":
    import os
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-25.0.2"
    import pyterrier as pt

    # Load search index once
    songs_df = pd.read_csv("songs_expanded2.zip")
    base = os.path.abspath("var_song_five_expanded2/index")
    bm25 = pt.terrier.Retriever(base, wmodel="BM25")

    while True:
        query = input("\nEnter a semantic query and find your songs (or 'q' to quit): ").strip()
        if query.lower() == "q":
            print("Thanks for searching!")
            break
        if not query:
            continue

        # Search
        results = bm25.search(query)
        search_df = songs_df.iloc[results.head(50).docid]
        print(songs_df.iloc[results.head(50)])
        print(f"\n--- Search Results for '{query}' ---")

        # Build recommender fresh each query using search results as seed
        rec = Combine(df_path)
        rec.build(search_df=search_df, n_users=30, k_neighbors=50)
        evaluate(rec, k=10)

        # Build User
        UID = 21 # Change user id
        #uid_rand = random.Random()
        #UID = uid_rand.randint(1, 29) # For demo purposes 
        demo_user = rec.users[UID]

        print(f"\nDemo user: {demo_user.user_id}")
        print(f"Preferred themes: {demo_user.preferred_themes}")

        # Get reccomendations for user based off serach
        recommendations = rec.recommend_for_user(demo_user.user_id, k=10)
        print("\nTop 10 recommendations:")
        for i, r in enumerate(recommendations, 1):
            print(f"{i:02d}. {r['track_name']} — {r['artist_name']}  (score={r['score']})")