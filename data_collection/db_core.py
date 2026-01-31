import sqlite3
import time
import threading

DB_PATH = "usage.db"


def init_db(db_path=DB_PATH):
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    # Raw app events
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        app_id TEXT NOT NULL,
        event_type TEXT CHECK(event_type IN ('open','close','focus')) NOT NULL
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_events_time
    ON app_events(timestamp)
    """)

    # Aggregated buckets (for training)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_buckets (
        bucket_id INTEGER NOT NULL,
        app_id TEXT NOT NULL,
        usage_count INTEGER NOT NULL,
        PRIMARY KEY (bucket_id, app_id)
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_bucket_time
    ON app_buckets(bucket_id)
    """)

    # Sessions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time INTEGER NOT NULL,
        end_time INTEGER NOT NULL,
        apps TEXT NOT NULL
    )
    """)

    # Model predictions (for evaluation)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        bucket_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        predicted_apps TEXT NOT NULL,
        scores TEXT NOT NULL,
        actual_app TEXT
    )
    """)

    conn.commit()
    return conn


class EventLogger:
    """Thread-safe event logger"""
    
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
    
    def log_event(self, app_id, event_type='focus'):
        """Log app usage event with timestamp"""
        ts = int(time.time())
        with self.lock:
            self.conn.execute(
                "INSERT INTO app_events (timestamp, app_id, event_type) VALUES (?, ?, ?)",
                (ts, app_id, event_type)
            )
            self.conn.commit()
    
    def get_recent_events(self, limit=10):
        """Get recent events for debugging"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT timestamp, app_id, event_type FROM app_events ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return cur.fetchall()


if __name__ == "__main__":
    # Test database creation
    conn = init_db()
    logger = EventLogger(conn)
    
    # Test logging
    logger.log_event("test_app", "focus")
    print("Recent events:", logger.get_recent_events())
    
    conn.close()
    print("✓ Database initialized successfully")