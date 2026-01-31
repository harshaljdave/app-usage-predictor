import numpy as np
import pickle


class AppEmbeddings:
    """Skip-gram style embeddings for app co-occurrence"""
    
    def __init__(self, vocab, dim=16, lr=0.01):
        self.vocab = vocab
        self.inv_vocab = {i: app for app, i in vocab.items()}
        self.dim = dim
        self.lr = lr
        
        # Initialize embeddings
        n = len(vocab)
        self.W_in = np.random.randn(n, dim) * 0.01
        self.W_out = np.random.randn(n, dim) * 0.01
    
    def train_session(self, session, window=2):
        """Online update from one session"""
        indices = [self.vocab[app] for app in session if app in self.vocab]
        
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i == j:
                    continue
                context = indices[j]
                self._update(center, context)
    
    def _update(self, center_idx, context_idx):
        """Single SGD update"""
        v = self.W_in[center_idx]
        u = self.W_out[context_idx]
        
        # Sigmoid
        score = 1.0 / (1.0 + np.exp(-np.dot(v, u)))
        
        # Gradient
        grad = self.lr * (1.0 - score)
        
        # Update
        self.W_in[center_idx] += grad * u
        self.W_out[context_idx] += grad * v
    
    def similarity(self, app1, app2):
        """Cosine similarity between apps"""
        if app1 not in self.vocab or app2 not in self.vocab:
            return 0.0
        
        v1 = self.W_in[self.vocab[app1]]
        v2 = self.W_in[self.vocab[app2]]
        
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    
    def similarity_to_set(self, app, app_list):
        """Average similarity to a set of apps"""
        if not app_list or app not in self.vocab:
            return 0.0
        
        sims = [self.similarity(app, other) for other in app_list if other in self.vocab]
        return np.mean(sims) if sims else 0.0
    
    def most_similar(self, app, top_k=5):
        """Find k most similar apps"""
        if app not in self.vocab:
            return []
        
        v = self.W_in[self.vocab[app]]
        sims = []
        
        for other_app, idx in self.vocab.items():
            if other_app == app:
                continue
            
            v_other = self.W_in[idx]
            sim = np.dot(v, v_other) / (np.linalg.norm(v) * np.linalg.norm(v_other) + 1e-8)
            sims.append((other_app, sim))
        
        return sorted(sims, key=lambda x: -x[1])[:top_k]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'dim': self.dim,
                'W_in': self.W_in,
                'W_out': self.W_out
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inv_vocab = {i: app for app, i in self.vocab.items()}
            self.dim = data['dim']
            self.W_in = data['W_in']
            self.W_out = data['W_out']


if __name__ == "__main__":
    import sqlite3
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_processing.preprocessing import extract_sessions, build_vocab
    
    print("=== Embeddings Test ===")
    
    conn = sqlite3.connect("usage_synthetic.db")
    
    # Extract and store sessions
    extract_sessions(conn)
    vocab = build_vocab(conn, min_count=10)
    
    # Read sessions from database
    cur = conn.cursor()
    cur.execute("SELECT apps FROM app_sessions")
    sessions = [row[0].split(',') for row in cur.fetchall()]
    
    print(f"Training on {len(sessions)} sessions")
    print(f"Vocabulary: {list(vocab.keys())}")
    
    # Initialize embeddings
    emb = AppEmbeddings(vocab, dim=16, lr=0.01)
    
    # Train (multiple passes)
    for epoch in range(5):
        for session in sessions:
            emb.train_session(session, window=2)
        print(f"Epoch {epoch+1}/5 complete")
    
    # Test similarities
    print("\nApp similarities:")
    for app in list(vocab.keys())[:3]:
        similar = emb.most_similar(app, top_k=3)
        print(f"\n{app}:")
        for sim_app, score in similar:
            print(f"  {sim_app}: {score:.3f}")
    
    # Save
    emb.save("app_embeddings.pkl")
    print("\n✓ Embeddings saved to app_embeddings.pkl")
    
    conn.close()