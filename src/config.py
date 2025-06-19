from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class DCNConfig:
    """
    Model configuration
    """
    # Directory paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    model_save_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    
    # Embedding configuration
    glove_path: str = "glove_embeddings/glove.840B.300d.txt"
    glove_dim: int = 300
    # glove_path: str = "glove_embeddings/glove.6B.300d.txt"
    # glove_dim: int = 300
    
    # Input dimensions
    context_len: int = 600
    question_len: int = 30
    
    # Model architecture
    hidden_dim: int = 200
    embedding_dim: int = 300
    decoding_steps: int = 4
    max_dec_steps: int = 4
    maxout_pool_size: int = 16
    
    # Training hyperparameters
    lr: float = 0.001
    dropout_ratio: float = 0.15
    max_grad_norm: float = 5.0
    batch_size: int = 32
    num_epochs: int = 1
    reg_lambda: float = 0.00007
    skip_batches = True  # or False to disable
    skip_ratio = 0.8     # Skip 80% of batches
    
    # Logging and checkpointing
    print_frequency: int = 100 # Print every 100 iterations
    eval_frequency: int = 1 # Evaluate every epoch
    # save_every: int = 50000000
    
    # Optional configurations
    num_workers: int = 4
    use_cuda: bool = True
