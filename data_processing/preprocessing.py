import sqlite3
import numpy as np
from collections import defaultdict, deque
import sys
sys.path.append('..')
from config.settings import BUCKET_SIZE, SESSION_GAP

def bucket_id(timestamp):
    """Convert timestamp to bucket ID"""
    return timestamp // BUCKET_SIZE


def aggregate_to_buckets(conn, output_conn=None):
    """Aggregate events into 30-min buckets"""
    if output_conn is None:
        output_conn = conn
    
    cur = conn.cursor()
    cur.execute("SELECT timestamp, app_id FROM app_events ORDER BY timestamp")
    events = cur.fetchall()
    
    buckets = defaultdict(lambda: defaultdict(int))
    for ts, app in events:
        b = bucket_id(ts)
        buckets[b][app] += 1
    
    # Insert into app_buckets
    out_cur = output_conn.cursor()
    for b_id, apps in buckets.items():
        for app, count in apps.items():
            out_cur.execute("""
                INSERT OR REPLACE INTO app_buckets (bucket_id, app_id, usage_count)
                VALUES (?, ?, ?)
            """, (b_id, app, count))
    
    output_conn.commit()
    
    print(f"✓ Aggregated {len(events)} events into {len(buckets)} buckets")
    return buckets


def extract_sessions(conn, output_conn=None):
    """Build sessions with gap threshold"""
    if output_conn is None:
        output_conn = conn
    
    cur = conn.cursor()
    cur.execute("SELECT timestamp, app_id FROM app_events ORDER BY timestamp")
    events = cur.fetchall()
    
    sessions = []
    current_session = []
    last_ts = None
    start_ts = None
    
    for ts, app in events:
        if last_ts is None or ts - last_ts > SESSION_GAP:
            if current_session:
                sessions.append((start_ts, last_ts, current_session))
            current_session = [app]
            start_ts = ts
        else:
            current_session.append(app)
        last_ts = ts
    
    if current_session:
        sessions.append((start_ts, last_ts, current_session))
    
    # Insert into app_sessions
    out_cur = output_conn.cursor()
    out_cur.execute("DELETE FROM app_sessions")
    for start, end, apps in sessions:
        out_cur.execute("""
            INSERT INTO app_sessions (start_time, end_time, apps)
            VALUES (?, ?, ?)
        """, (start, end, ','.join(apps)))
    
    output_conn.commit()
    
    print(f"✓ Extracted {len(sessions)} sessions")
    return sessions


def build_vocab(conn, min_count=10):
    """Build app vocabulary from buckets"""
    cur = conn.cursor()
    cur.execute("""
        SELECT app_id, SUM(usage_count) as total
        FROM app_buckets
        GROUP BY app_id
        HAVING total >= ?
        ORDER BY app_id
    """, (min_count,))
    
    vocab = {row[0]: idx for idx, row in enumerate(cur.fetchall())}
    
    print(f"✓ Vocabulary size: {len(vocab)} apps (min_count={min_count})")
    return vocab


if __name__ == "__main__":
    # Test on synthetic data
    conn = sqlite3.connect("../usage_synthetic.db")
    
    print("=== Preprocessing Test ===")
    aggregate_to_buckets(conn)
    extract_sessions(conn)
    vocab = build_vocab(conn)
    
    print("\nVocabulary:", list(vocab.keys()))
    
    conn.close()