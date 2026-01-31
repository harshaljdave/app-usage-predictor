# Time windows
BUCKET_SIZE = 30 * 60           # 30 minutes
SESSION_GAP = 15 * 60           # 15 minutes

# Training
MIN_APP_COUNT = 10
TCN_EPOCHS = 20
EMBEDDING_DIM = 16
WINDOW_SIZE = 24                # 12 hours lookback

# Memory
MAX_ROLLING_BUCKETS = 672       # 14 days