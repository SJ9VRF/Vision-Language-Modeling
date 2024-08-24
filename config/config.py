# config.py

# Paths
DATA_DIR = '/path/to/data'
TRAIN_DATA_FILE = 'train.csv'
VALIDATION_DATA_FILE = 'validation.csv'
TEST_DATA_FILE = 'test.csv'

# Model parameters
IMAGE_DIM = 768  # Image embedding dimensions
TEXT_DIM = 768   # Text embedding dimensions
NUM_LABELS = 13  # Number of output labels

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
