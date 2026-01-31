import numpy as np
import pickle
from collections import defaultdict
from itertools import combinations


def mine_frequent_itemsets(sessions, min_support=5, max_size=3):
    """Mine frequent itemsets using simple counting"""
    counts = defaultdict(int)
    
    for session in sessions:
        unique = set(session)
        for size in range(1, min(max_size + 1, len(unique) + 1)):
            for combo in combinations(sorted(unique), size):
                counts[combo] += 1
    
    # Filter by min_support
    frequent = {
        itemset: cnt 
        for itemset, cnt in counts.items() 
        if cnt >= min_support
    }
    
    return frequent


def generate_rules(itemsets, min_confidence=0.6):
    """Generate association rules from frequent itemsets"""
    rules = []
    
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        
        # Generate all possible rules
        for i in range(len(itemset)):
            lhs = tuple(x for j, x in enumerate(itemset) if j != i)
            rhs = itemset[i]
            
            if lhs in itemsets:
                confidence = support / itemsets[lhs]
                if confidence >= min_confidence:
                    rules.append({
                        'lhs': lhs,
                        'rhs': rhs,
                        'confidence': confidence,
                        'support': support
                    })
    
    return rules


class AssociationRuleMiner:
    """Mine and apply association rules"""
    
    def __init__(self, min_support=5, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = []
    
    def train(self, sessions):
        """Mine rules from sessions"""
        # Mine itemsets
        itemsets = mine_frequent_itemsets(
            sessions, 
            min_support=self.min_support, 
            max_size=3
        )
        
        # Generate rules
        self.rules = generate_rules(itemsets, self.min_confidence)
        
        print(f"✓ Mined {len(itemsets)} itemsets, {len(self.rules)} rules")
    
    def get_rule_score(self, recent_apps, target_app):
        """Get confidence score for target_app given recent_apps"""
        if not recent_apps:
            return 0.0
        
        recent_set = set(recent_apps)
        max_conf = 0.0
        
        for rule in self.rules:
            if rule['rhs'] == target_app:
                lhs_set = set(rule['lhs'])
                if lhs_set.issubset(recent_set):
                    max_conf = max(max_conf, rule['confidence'])
        
        return max_conf
    
    def get_top_consequents(self, recent_apps, top_k=5):
        """Get top-k predicted apps given recent_apps"""
        scores = defaultdict(float)
        
        for rule in self.rules:
            lhs_set = set(rule['lhs'])
            if lhs_set.issubset(set(recent_apps)):
                scores[rule['rhs']] = max(scores[rule['rhs']], rule['confidence'])
        
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'min_support': self.min_support,
                'min_confidence': self.min_confidence
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.rules = data['rules']
            self.min_support = data['min_support']
            self.min_confidence = data['min_confidence']


if __name__ == "__main__":
    import sqlite3
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=== Association Rules Test ===")
    
    conn = sqlite3.connect("usage_synthetic.db")
    
    # Get sessions
    cur = conn.cursor()
    cur.execute("SELECT apps FROM app_sessions")
    sessions = [row[0].split(',') for row in cur.fetchall()]
    
    print(f"Mining from {len(sessions)} sessions")
    
    # Mine rules
    miner = AssociationRuleMiner(min_support=5, min_confidence=0.6)
    miner.train(sessions)
    
    # Show top rules
    print("\nTop 10 rules:")
    sorted_rules = sorted(miner.rules, key=lambda x: -x['confidence'])[:10]
    for rule in sorted_rules:
        lhs = ', '.join(rule['lhs'])
        print(f"  {{{lhs}}} → {rule['rhs']} (conf={rule['confidence']:.2f})")
    
    # Test prediction
    test_recent = ['vscode', 'terminal']
    predictions = miner.get_top_consequents(test_recent, top_k=5)
    print(f"\nPredictions given {test_recent}:")
    for app, score in predictions:
        print(f"  {app}: {score:.2f}")
    
    # Save
    miner.save("association_rules.pkl")
    print("\n✓ Rules saved to association_rules.pkl")
    
    conn.close()