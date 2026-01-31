import time
import subprocess
from db_core import EventLogger, init_db

POLL_INTERVAL = 3
IDLE_THRESHOLD = 300


def get_active_app():
    """Get active app via xdotool + /proc"""
    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", win_id],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except:
        return None


def normalize_app_name(raw_name):
    """Normalize app names"""
    if not raw_name:
        return None
    
    normalizations = {
        'google-chrome': 'chrome',
        'chrome-browser': 'chrome',
        'firefox-esr': 'firefox',
        'code': 'vscode',
        'konsole': 'terminal',
        'gnome-terminal': 'terminal',
    }
    
    name = raw_name.lower().strip()
    return normalizations.get(name, name)


def get_idle_time():
    """Get idle time in seconds"""
    try:
        idle_ms = int(subprocess.check_output(
            ["xprintidle"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return idle_ms / 1000
    except:
        return 0


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


def run_logger():
    """Main logging loop"""
    conn = init_db()
    logger = EventLogger(conn)
    last_app = None
    
    print("🚀 X11 logger started")
    print(f"📊 Polling every {POLL_INTERVAL}s")
    print(f"💤 Idle threshold: {IDLE_THRESHOLD}s")
    print("-" * 50)
    
    while True:
        try:
            if is_locked():
                last_app = None
                time.sleep(POLL_INTERVAL)
                continue
            
            if get_idle_time() > IDLE_THRESHOLD:
                last_app = None
                time.sleep(POLL_INTERVAL)
                continue
            
            raw_app = get_active_app()
            app = normalize_app_name(raw_app)
            
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
    
    # Summary
    recent = logger.get_recent_events(10)
    print(f"\nLogged {len(recent)} recent events")
    
    conn.close()


if __name__ == "__main__":
    run_logger()