"""Transactional namespace store. Activation is atomic; incomplete indexes never rank.

SQLite keeps compact float32 vectors and durable jobs/reviews in a separate database.
The existing SigLIP2 LanceDB tables remain the explicit baseline only.
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import numpy as np

class ReindexRequired(ValueError):
    pass

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def vectors(values, dimension):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dimension or not np.isfinite(arr).all():
        raise ValueError("Invalid embedding shape or non-finite values")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("Zero embedding")
    return arr / norm

class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.sqlite3"
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS normality_profiles(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS normality_banks(profile_id TEXT, version TEXT, data TEXT NOT NULL, PRIMARY KEY(profile_id,version));
            CREATE TABLE IF NOT EXISTS videos(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS indexes(namespace TEXT PRIMARY KEY, video_id TEXT, config TEXT, expected INTEGER, status TEXT);
            CREATE TABLE IF NOT EXISTS active(video_id TEXT PRIMARY KEY, namespace TEXT);
            CREATE TABLE IF NOT EXISTS windows(namespace TEXT, id INTEGER, data TEXT, PRIMARY KEY(namespace,id));
            CREATE TABLE IF NOT EXISTS embeddings(namespace TEXT, window_id INTEGER, modality TEXT, slot INTEGER, dimension INTEGER, vector BLOB, PRIMARY KEY(namespace,window_id,modality,slot));
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, video_id TEXT, data TEXT);
            CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, decision TEXT, note TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS vector_cache(namespace TEXT, key TEXT, dimension INTEGER, vector BLOB, PRIMARY KEY(namespace,key));
            ''')

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def put(self, table, key, value):
        if table not in ("videos", "jobs"):
            raise ValueError("Unsupported table")
        with self.db() as db:
            db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def get(self, table, key):
        if table not in ("videos", "jobs", "events"):
            raise ValueError("Unsupported table")
        with self.db() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            return json.loads(row[0])

    def list(self, table, video_id=None):
        if table not in ("videos", "jobs", "events"):
            raise ValueError("Unsupported table")
        with self.db() as db:
            sql = f"SELECT data FROM {table}"
            rows = db.execute(sql + " WHERE video_id=?", (video_id,)) if table == "events" and video_id else db.execute(sql)
            return [json.loads(row[0]) for row in rows]

    def begin(self, video, namespace, config, expected):
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO indexes VALUES (?,?,?,?,?)", (namespace, video, json.dumps(config, sort_keys=True), expected, "building"))

    def windows(self, namespace):
        with self.db() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT data FROM windows WHERE namespace=? ORDER BY id", (namespace,))]

    def save_window(self, namespace, window, channels, dimension):
        # Validate all channels before beginning the write: OOM/failure cannot leave half a window.
        checked = {kind: vectors(value, dimension) for kind, value in channels.items()}
        with self.db() as db:
            for kind, array in checked.items():
                db.execute("DELETE FROM embeddings WHERE namespace=? AND window_id=? AND modality=?", (namespace, window["id"], kind))
                db.executemany("INSERT INTO embeddings VALUES (?,?,?,?,?,?)",
                    [(namespace, window["id"], kind, slot, dimension, row.tobytes()) for slot, row in enumerate(array)])
            db.execute("INSERT OR REPLACE INTO windows VALUES (?,?,?)", (namespace, window["id"], json.dumps(window, ensure_ascii=False)))

    def save_joint(self, namespace, window_id, value, dimension):
        vector = vectors(value, dimension)
        if len(vector) != 1:
            raise ValueError("Joint embedding must represent one window")
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", (namespace, window_id, "joint", 0, dimension, vector[0].tobytes()))

    def cached_vector(self, namespace, key, dimension):
        with self.db() as db:
            row = db.execute("SELECT dimension,vector FROM vector_cache WHERE namespace=? AND key=?", (namespace, key)).fetchone()
            if row is None:
                return None
            if row[0] != dimension:
                raise ReindexRequired("Cache dimension mismatch")
            return vectors([np.frombuffer(row[1], dtype=np.float32)], dimension)[0]

    def cache_vector(self, namespace, key, value, dimension):
        checked = vectors([value], dimension)[0]
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO vector_cache VALUES (?,?,?,?)", (namespace, key, dimension, checked.tobytes()))

    def activate(self, video, namespace):
        with self.db() as db:
            row = db.execute("SELECT expected,video_id FROM indexes WHERE namespace=?", (namespace,)).fetchone()
            count = db.execute("SELECT COUNT(*) FROM windows WHERE namespace=?", (namespace,)).fetchone()[0]
            if row is None or row[1] != video or row[0] != count:
                raise ReindexRequired("Incomplete index")
            db.execute("UPDATE indexes SET status='ready' WHERE namespace=?", (namespace,))
            db.execute("INSERT OR REPLACE INTO active VALUES (?,?)", (video, namespace))

    def require(self, video, namespace):
        with self.db() as db:
            row = db.execute("SELECT i.namespace,i.status FROM active a JOIN indexes i ON a.namespace=i.namespace WHERE a.video_id=?", (video,)).fetchone()
            if row is None or row[0] != namespace or row[1] != "ready":
                raise ReindexRequired("reindex_required")
        return self.windows(namespace)

    def embeddings(self, namespace, dimension):
        result = {}
        with self.db() as db:
            for row in db.execute("SELECT window_id,modality,dimension,vector FROM embeddings WHERE namespace=? ORDER BY window_id,slot", (namespace,)):
                if row[2] != dimension:
                    raise ReindexRequired("Vector dimension mismatch")
                result.setdefault(row[1], []).append((row[0], np.frombuffer(row[3], dtype=np.float32).copy()))
        return {kind: (np.array([r[0] for r in rows]), vectors([r[1] for r in rows], dimension)) for kind, rows in result.items()}

    def save_event(self, event):
        with self.db() as db:
            # Model updates must not erase an existing human review.
            old = db.execute("SELECT data FROM events WHERE id=?", (event["id"],)).fetchone()
            if old:
                previous = json.loads(old[0])
                event["review"] = previous.get("review", "pending")
                event["review_note"] = previous.get("review_note", "")
            db.execute("INSERT OR REPLACE INTO events VALUES (?,?,?)", (event["id"], event["video_id"], json.dumps(event, ensure_ascii=False)))

    def review(self, event_id, decision, note):
        if decision not in ("confirmed", "rejected", "pending"):
            raise ValueError("Invalid review decision")
        with self.db() as db:
            row = db.execute("SELECT data FROM events WHERE id=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(event_id)
            event = json.loads(row[0])
            event.update(review=decision, review_note=note)
            db.execute("UPDATE events SET data=? WHERE id=?", (json.dumps(event, ensure_ascii=False), event_id))
            db.execute("INSERT INTO reviews(event_id,decision,note) VALUES (?,?,?)", (event_id, decision, note))
            return event
