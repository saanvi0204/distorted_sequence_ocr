import os

class CFG:
    # Image 
    IMG_HEIGHT = 100        # Native image height; avoids distortion from resizing
    IMG_WIDTH  = 200        # Native image width;  preserves horizontal character spacing

    # Model 
    HIDDEN_SIZE     = 256   # BiLSTM hidden size per direction (total = 512)
    NUM_LSTM_LAYERS = 2     # Stacked BiLSTM layers for richer sequential representations
    DROPOUT         = 0.3   # LSTM inter-layer dropout
    ATTN_HEADS      = 8     # Multi-head attention heads (embed_dim 512 / 8 = 64 per head)

    # Training 
    BATCH_SIZE    = 64      # Fits comfortably in GPU memory
    EPOCHS        = 80      # Upper bound; early stopping typically halts before this
    LEARNING_RATE = 3e-4    # AdamW default; pairs well with cosine annealing
    WEIGHT_DECAY  = 1e-4    # L2 regularisation strength
    VAL_SPLIT     = 0.10    # 10 % of training data held out for validation
    LR_MIN        = 1e-6    # Cosine annealing floor — prevents LR collapsing to zero
    NUM_WORKERS   = 4       # DataLoader worker processes

    # Paths 
    # Override via environment variables or pass directly to the CLI scripts.
    TRAIN_IMG_DIR  = os.getenv("TRAIN_IMG_DIR",  "data/train_images")
    TEST_IMG_DIR   = os.getenv("TEST_IMG_DIR",   "data/test_images")
    TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH", "data/train-labels.csv")
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
    SUBMISSION_DIR = os.getenv("SUBMISSION_DIR", "outputs")

    # Reproducibility
    SEED = 42
