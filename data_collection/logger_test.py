import sqlite3
import random
import time
from datetime import datetime, timedelta
from db_core import init_db

# Developer usage patterns
PATTERNS = {
    'morning': {  # 8-12
        'apps': ['chrome', 'slack', 'terminal', 'vscode'],
        'weights': [0.3, 0.3, 0.2, 0.2]
    },
    'afternoon': {  # 12-17
        'apps': ['vscode', 'terminal', 'chrome', 'slack', 'git'],
        'weights': [0.4, 0.25, 0.2, 0.1, 0.05]
    },
    'evening': {  # 17-22
        'apps': ['chrome', 'spotify', 'discord', 'vscode'],
        'weights': [0.4, 0.3, 0.2, 0.1]
    }
}

TRANSITIONS = {
    'vscode': ['terminal', 'chrome', 'slack'],
    'terminal': ['vscode', 'chrome'],
    'chrome': ['vscode', 'slack', 'terminal'],
    'slack': ['chrome', 'vscode'],
}


def get_pattern_for_hour(hour):
    if 8 <= hour < 12:
        return PATTERNS['morning']
    elif 12 <= hour < 17:
        return PATTERNS['afternoon']
    elif 17 <= hour < 22:
        return PATTERNS['evening']
    return None


def generate_day(conn, date, noise=0.15):
    """Generate one day of realistic events"""
    current_time = int(date.timestamp())
    end_time = current_time + 86400  # 24 hours
    
    last_app = None
    events = []
    
    while current_time < end_time:
        hour = datetime.fromtimestamp(current_time).hour
        
        # Skip night hours (22-8)
        if hour >= 22 or hour < 8:
            current_time += 3600  # Skip hour
            continue
        
        pattern = get_pattern_for_hour(hour)
        if not pattern:
            current_time += 3600
            continue
        
        # Choose app: 85% from pattern, 15% from transitions or random
        if random.random() < noise or last_app is None:
            app = random.choices(pattern['apps'], pattern['weights'])[0]
        else:
            # Follow transitions
            if last_app in TRANSITIONS:
                app = random.choice(TRANSITIONS[last_app])
            else:
                app = random.choices(pattern['apps'], pattern['weights'])[0]
        
        events.append((current_time, app, 'focus'))
        last_app = app
        
        # Random session length: 2-15 minutes
        current_time += random.randint(120, 900)
    
    # Insert into database
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO app_events (timestamp, app_id, event_type) VALUES (?, ?, ?)",
        events
    )
    conn.commit()
    
    return len(events)


def generate_dataset(days=21):
    """Generate multi-week synthetic dataset"""
    conn = init_db("usage_synthetic.db")
    
    start_date = datetime.now() - timedelta(days=days)
    total_events = 0
    
    print(f"Generating {days} days of synthetic data...")
    
    for i in range(days):
        date = start_date + timedelta(days=i)
        count = generate_day(conn, date)
        total_events += count
        print(f"Day {i+1}/{days}: {count} events ({date.strftime('%Y-%m-%d')})")
    
    print(f"\n✓ Generated {total_events} events")
    print(f"Database: usage_synthetic.db")
    
    conn.close()


if __name__ == "__main__":
    generate_dataset(21)