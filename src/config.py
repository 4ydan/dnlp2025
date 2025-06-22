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
    model_save_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "models")
    
    # Embedding configuration
    glove_path: str = "glove_embeddings/glove.840B.300d.txt"
    glove_dim: int = 300
    # glove_path: str = "glove_embeddings/glove.6B.300d.txt"
    # glove_dim: int = 300
    
    # Input dimensions
    context_len: int = 600
    question_len: int = 50
    
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
    num_epochs: int = 2
    reg_lambda: float = 0.00007
    
    # Logging and checkpointing
    print_frequency: int = 100 # Print every 100 iterations
    eval_frequency: int = 1 # Evaluate every epoch
    skip_frequency = 1  # Train every 10th batch (skip 90%)
    # config.skip_frequency = 1  # Train on all batches (no skipping)
    # config.skip_frequency = 2  # Train every 2nd batch (skip 50%)
    
    # Optional configurations
    num_workers: int = 4
    use_cuda: bool = True
