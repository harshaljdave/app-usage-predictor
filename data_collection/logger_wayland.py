import time
import sqlite3
import subprocess
from pathlib import Path
from db_core import EventLogger, init_db

POLL_INTERVAL = 5
IDLE_THRESHOLD = 300
KACTIVITY_DB = Path.home() / ".local/share/kactivitymanagerd/resources/database"


def is_locked():
    """Check if screen is locked"""
    try:
        out = subprocess.check_output(
            ["loginctl", "show-session", "-p", "LockedHint", "self"],
            stderr=subprocess.DEVNULL
        ).decode()
        return "LockedHint=yes" in out
    except:
        return False


def get_idle_time():
    """Get idle time in seconds"""
    try:
        idle_ms = int(subprocess.check_output(
            ["xprintidle"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return idle_ms / 1000
    except:
        return 0


def get_active_app_from_kactivity():
    """Query KActivityManager database for recent app usage"""
    try:
        conn = sqlite3.connect(f"file:{KACTIVITY_DB}?mode=ro", uri=True)
        cur = conn.cursor()
        
        # Get most recent application focus
        cur.execute("""
            SELECT targettedResource 
            FROM ResourceEvent 
            WHERE targettedResource LIKE 'applications:%'
            ORDER BY start DESC 
            LIMIT 1
        """)
        
        result = cur.fetchone()
        conn.close()
        
        if result and result[0]:
            # Extract: applications:google-chrome.desktop -> chrome
            resource = result[0]
            app = resource.replace('applications:', '').replace('.desktop', '')
            
            # Handle org.kde.* format
            if '.' in app:
                app = app.split('.')[-1]
            
            return normalize_app_name(app)
        
        return None
    except Exception as e:
        print(f"Debug error: {e}")
        return None


def normalize_app_name(raw_name):
    """Normalize app names"""
    if not raw_name:
        return None
    
    normalizations = {
        'google-chrome': 'chrome',
        'firefox-esr': 'firefox',
        'code': 'vscode',
        'konsole': 'terminal',
        'gnome-terminal': 'terminal',
    }
    
    name = raw_name.lower().strip()
    return normalizations.get(name, name)


def run_logger():
    """Main logging loop"""
    if not KACTIVITY_DB.exists():
        print(f"❌ KActivityManager database not found at {KACTIVITY_DB}")
        print("   Make sure KDE Activities are enabled")
        return
    
    conn = init_db()
    logger = EventLogger(conn)
    last_app = None
    
    print("🚀 Wayland logger started (KActivityManager)")
    print(f"📊 Polling every {POLL_INTERVAL}s")
    print(f"💤 Idle threshold: {IDLE_THRESHOLD}s")
    print(f"📁 Database: usage.db")
    print("-" * 50)
    
    while True:
        try:
            if is_locked():
                if last_app is not None:
                    print("🔒 Screen locked")
                last_app = None
                time.sleep(POLL_INTERVAL)
                continue
            
            idle_time = get_idle_time()
            if idle_time > IDLE_THRESHOLD:
                if last_app is not None:
                    print(f"💤 Idle for {idle_time:.0f}s")
                last_app = None
                time.sleep(POLL_INTERVAL)
                continue
            
            app = get_active_app_from_kactivity()
            
            if app and app != last_app:
                logger.log_event(app, 'focus')
                print(f"✓ {time.strftime('%H:%M:%S')} | {app}")
                last_app = app
            
        except KeyboardInterrupt:
            print("\n👋 Stopping logger...")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
        
        time.sleep(POLL_INTERVAL)
    
    recent = logger.get_recent_events(5)
    print("\nLast 5 events:")
    for ts, app, event in recent:
        print(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))} | {app}")
    
    conn.close()

if __name__ == "__main__":
    run_logger()