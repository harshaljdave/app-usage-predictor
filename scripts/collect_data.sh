#!/bin/bash
# scripts/collect_data.sh
# Start/stop/status the X11 usage logger
#
# Usage:
#   bash scripts/collect_data.sh start
#   bash scripts/collect_data.sh stop
#   bash scripts/collect_data.sh status

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/outputs/logs/logger.log"
PID_FILE="$PROJECT_DIR/outputs/logs/logger.pid"

mkdir -p "$PROJECT_DIR/outputs/logs"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "⚠️  Logger already running (PID: $(cat $PID_FILE))"
            exit 1
        fi

        echo "🚀 Starting logger..."
        cd "$PROJECT_DIR"
        nohup python -u data_collection/logger_x11.py >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "✓ Logger started (PID: $(cat $PID_FILE))"
        echo "📝 Log: $LOG_FILE"
        ;;

    stop)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            kill $(cat "$PID_FILE")
            rm "$PID_FILE"
            echo "👋 Logger stopped"
        else
            echo "⚠️  Logger not running"
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "✅ Logger is running (PID: $(cat $PID_FILE))"
            echo "📝 Recent logs:"
            tail -5 "$LOG_FILE"

            # Event count
            DB="$PROJECT_DIR/usage.db"
            if [ -f "$DB" ]; then
                COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM app_events" 2>/dev/null || echo "0")
                APPS=$(sqlite3 "$DB" "SELECT COUNT(DISTINCT app_id) FROM app_events" 2>/dev/null || echo "0")
                echo "📊 Events: $COUNT | Unique apps: $APPS"
            fi
        else
            echo "❌ Logger is not running"
        fi
        ;;

    *)
        echo "Usage: bash scripts/collect_data.sh {start|stop|status}"
        exit 1
        ;;
esac