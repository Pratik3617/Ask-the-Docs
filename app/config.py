# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"


# chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4

# Generation
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# Paths
STORAGE_DIR = "app/storage"
INDEX_PATH = "app/storage/index/faiss.index"
METADATA_PATH = "app/storage/metadata.json"