import time
import subprocess
from db_core import EventLogger, init_db

POLL_INTERVAL = 3
IDLE_THRESHOLD = 300


def get_window_info():
    """Get active window process AND title"""
    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get PID for process name
        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", win_id],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        with open(f"/proc/{pid}/comm") as f:
            process = f.read().strip()
        
        # Get window title
        title = subprocess.check_output(
            ["xdotool", "getwindowname", win_id],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return process, title
    except:
        return None, None


def parse_chrome_title(title):
    """Extract profile and domain from Chrome window title
    
    Examples:
    "GitHub - Google Chrome" -> chrome:github.com
    "Gmail - Work - Google Chrome" -> chrome:work:gmail
    "YouTube - Profile 1 - Google Chrome" -> chrome:profile_1:youtube
    """
    # Remove " - Google Chrome" suffix
    title = title.replace(" - Google Chrome", "")
    title = title.replace(" - Chromium", "")
    
    parts = [p.strip() for p in title.split(" - ")]
    
    # Check if profile name is present (usually second-to-last)
    if len(parts) >= 2:
        profile = parts[-1].lower().replace(" ", "_")
        page = parts[0].lower()
        
        # Extract domain-like info from page title
        # "GitHub" -> "github", "Gmail" -> "gmail"
        domain = page.split()[0] if page else "unknown"
        
        if profile in ['default', 'person 1', 'profile 1']:
            return f"chrome:{domain}"
        else:
            return f"chrome:{profile}:{domain}"
    
    # Fallback
    page = parts[0].lower().split()[0] if parts else "unknown"
    return f"chrome:{page}"


def parse_terminal_title(title):
    """Extract context from terminal title
    
    Examples:
    "~/projects/app-predictor — Konsole" -> terminal:app-predictor
    "git push — Konsole" -> terminal:git
    "user@host: ~/Documents" -> terminal:documents
    """
    # Remove terminal app name
    title = title.replace(" — Konsole", "")
    title = title.replace(" - Terminal", "")
    
    # Extract directory or command
    if "~/" in title or "/" in title:
        # Directory shown
        path = title.split(":")[-1].strip() if ":" in title else title
        dir_name = path.strip("~/").split("/")[-1] or "home"
        return f"terminal:{dir_name}"
    else:
        # Command shown
        cmd = title.split()[0].lower()
        return f"terminal:{cmd}"


def normalize_app_name(process, title):
    """Enhanced normalization with title parsing"""
    
    # Basic normalizations
    base_normalizations = {
        'google-chrome': 'chrome',
        'chrome-browser': 'chrome',
        'firefox-esr': 'firefox',
        'code': 'vscode',
        'konsole': 'terminal',
        'gnome-terminal': 'terminal',
    }
    
    process = base_normalizations.get(process.lower(), process.lower())
    
    # Enhanced parsing for Chrome and Terminal
    if process == 'chrome' and title:
        return parse_chrome_title(title)
    elif process == 'terminal' and title:
        return parse_terminal_title(title)
    
    return process


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
    """Main logging loop with title parsing"""
    conn = init_db()
    logger = EventLogger(conn)
    last_app = None
    
    print("🚀 X11 logger started (with title parsing)")
    print(f"📊 Polling every {POLL_INTERVAL}s")
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
            
            process, title = get_window_info()
            app = normalize_app_name(process, title) if process else None
            
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
    
    conn.close()


if __name__ == "__main__":
    run_logger()