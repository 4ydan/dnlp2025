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
    # log_root: Path = field(default_factory=lambda: Path.home() / 'co-attention' / 'log')
    
    # Embedding configuration
    # glove_path: str = "glove_embeddings/glove.840B.300d.txt"
    # glove_dim: int = 300
    glove_path: str = "glove_embeddings/glove.6B.300d.txt"
    glove_dim: int = 300
    
    # Input dimensions
    context_len: int = 600
    question_len: int = 30
    
    # Model architecture
    hidden_dim: int = 200
    embedding_dim: int = 300
    decoding_steps: int = 4
    max_dec_steps: int = 4
    maxout_pool_size: int = 16
    model_type: Literal['co-attention', 'baseline'] = 'co-attention'
    
    # Training hyperparameters
    lr: float = 0.001
    dropout_ratio: float = 0.15
    max_grad_norm: float = 5.0
    batch_size: int = 32
    num_epochs: int = 50
    reg_lambda: float = 0.00007
    
    # Logging and checkpointing
    print_frequency: int = 100 # Print every 100 iterations
    eval_frequency: int = 1 # Evaluate every epoch
    # save_every: int = 50000000
    
    # Optional configurations
    seed: int = 42
    num_workers: int = 4
    use_cuda: bool = True
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    
    # Computed properties
    @property
    def embedding_path(self) -> Path:
        """Full path to embedding file."""
        return self.data_dir / self.embedding_filename
    
    @property
    def train_path(self) -> Path:
        """Path to training data."""
        return self.data_dir / 'train-v1.1.json'
    
    @property
    def dev_path(self) -> Path:
        """Path to development data."""
        return self.data_dir / 'dev-v1.1.json'
    
    @property
    def model_dir(self) -> Path:
        """Directory for saving model checkpoints."""
        return self.log_root / 'models'
    
    # def __post_init__(self):
    #     """Create directories if they don't exist."""
    #     self.data_dir.mkdir(parents=True, exist_ok=True)
    #     self.log_root.mkdir(parents=True, exist_ok=True)
    #     self.model_dir.mkdir(parents=True, exist_ok=True)
