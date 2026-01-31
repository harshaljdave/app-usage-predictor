import sqlite3
import numpy as np
from collections import defaultdict
import pickle


class FrequencyBaseline:
    """Simple time-of-day frequency baseline"""
    
    def __init__(self):
        self.hour_freq = defaultdict(lambda: defaultdict(int))  # {hour: {app: count}}
        self.day_freq = defaultdict(lambda: defaultdict(int))   # {day: {app: count}}
        self.transition_freq = defaultdict(lambda: defaultdict(int))  # {app: {next_app: count}}
        self.total_freq = defaultdict(int)  # {app: count}
    
    def train(self, conn):
        """Train on app_events"""
        import time
        
        cur = conn.cursor()
        cur.execute("SELECT timestamp, app_id FROM app_events ORDER BY timestamp")
        events = cur.fetchall()
        
        last_app = None
        
        for ts, app in events:
            t = time.localtime(ts)
            hour = t.tm_hour
            day = t.tm_wday
            
            self.hour_freq[hour][app] += 1
            self.day_freq[day][app] += 1
            self.total_freq[app] += 1
            
            if last_app:
                self.transition_freq[last_app][app] += 1
            
            last_app = app
        
        print(f"✓ Baseline trained on {len(events)} events")
    
    def predict(self, context, top_k=5):
        """Predict top-k apps
        
        context = {
            'hour': int,
            'day': int,
            'last_app': str or None
        }
        """
        scores = defaultdict(float)
        
        hour = context.get('hour', 12)
        day = context.get('day', 0)
        last_app = context.get('last_app')
        
        # Hour-of-day frequency
        hour_total = sum(self.hour_freq[hour].values()) or 1
        for app, count in self.hour_freq[hour].items():
            scores[app] += 0.4 * (count / hour_total)
        
        # Day-of-week frequency
        day_total = sum(self.day_freq[day].values()) or 1
        for app, count in self.day_freq[day].items():
            scores[app] += 0.3 * (count / day_total)
        
        # Transition probability
        if last_app and last_app in self.transition_freq:
            trans_total = sum(self.transition_freq[last_app].values()) or 1
            for app, count in self.transition_freq[last_app].items():
                scores[app] += 0.3 * (count / trans_total)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'hour_freq': dict(self.hour_freq),
                'day_freq': dict(self.day_freq),
                'transition_freq': dict(self.transition_freq),
                'total_freq': dict(self.total_freq)
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.hour_freq = defaultdict(lambda: defaultdict(int), data['hour_freq'])
            self.day_freq = defaultdict(lambda: defaultdict(int), data['day_freq'])
            self.transition_freq = defaultdict(lambda: defaultdict(int), data['transition_freq'])
            self.total_freq = defaultdict(int, data['total_freq'])


if __name__ == "__main__":
    import time
    
    conn = sqlite3.connect("usage_synthetic.db")
    
    model = FrequencyBaseline()
    model.train(conn)
    
    # Test prediction
    now = time.localtime()
    context = {
        'hour': now.tm_hour,
        'day': now.tm_wday,
        'last_app': 'vscode'
    }
    
    predictions = model.predict(context, top_k=5)
    print(f"\nPredictions for hour={context['hour']}, last_app={context['last_app']}:")
    for app, score in predictions:
        print(f"  {app}: {score:.3f}")
    
    # Save model
    model.save("baseline_model.pkl")
    print("\n✓ Model saved to baseline_model.pkl")
    
    conn.close()