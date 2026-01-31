import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with causal convolutions"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        # Causal convolution (trim future)
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # Trim to input length
        out = self.norm(out)
        out = F.relu(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    """Temporal Convolutional Network for app prediction"""
    
    def __init__(self, num_apps, hidden_dim=32):
        super().__init__()
        self.num_apps = num_apps
        
        # TCN blocks with increasing dilation
        self.block1 = TCNBlock(num_apps, hidden_dim, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.block3 = TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_apps)
    
    def forward(self, x):
        # x: (batch, time, apps)
        x = x.transpose(1, 2)  # -> (batch, apps, time)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Take last timestep
        x = x[:, :, -1]  # -> (batch, hidden_dim)
        
        # Predict next apps
        out = self.fc(x)
        return torch.sigmoid(out)


def train_tcn(model, X_train, Y_train, X_val, Y_val, epochs=20, lr=1e-3, batch_size=32):
    """Train TCN model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    Y_val = torch.FloatTensor(Y_val)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_Y = Y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, Y_val).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model


def save_tcn(model, vocab, path):
    """Save model and vocabulary"""
    torch.save({
        'model_state': model.state_dict(),
        'num_apps': model.num_apps,
        'vocab': vocab
    }, path)


def load_tcn(path):
    """Load model and vocabulary"""
    checkpoint = torch.load(path)
    
    model = TCN(num_apps=checkpoint['num_apps'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint['vocab']


if __name__ == "__main__":
    import sqlite3
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_processing.preprocessing import aggregate_to_buckets, build_vocab
    from data_processing.feature_engineering import build_tcn_dataset
    
    print("=== TCN Model Test ===")
    
    conn = sqlite3.connect("usage_synthetic.db")
    
    # Prepare data
    buckets = aggregate_to_buckets(conn)
    vocab = build_vocab(conn, min_count=10)
    X, Y = build_tcn_dataset(buckets, vocab, window=24)
    
    print(f"Dataset: {len(X)} samples, {len(vocab)} apps")
    
    # Train/val split (80/20)
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Initialize model
    model = TCN(num_apps=len(vocab))
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    print("\nTraining...")
    model = train_tcn(model, X_train, Y_train, X_val, Y_val, epochs=10)
    
    # Save
    save_tcn(model, vocab, "tcn_model.pt")
    print("\n✓ Model saved to tcn_model.pt")
    
    # Test prediction
    test_input = X_val[0:1]
    with torch.no_grad():
        pred = model(torch.FloatTensor(test_input))
        scores = pred[0].numpy()
    
    inv_vocab = {i: app for app, i in vocab.items()}
    top_apps = sorted(enumerate(scores), key=lambda x: -x[1])[:5]
    
    print("\nSample prediction:")
    for idx, score in top_apps:
        print(f"  {inv_vocab[idx]}: {score:.3f}")
    
    conn.close()