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
    embedding_filename: str = 'glove.6B.50d.txt'
    embedding_size: int = 100
    
    # Input dimensions
    context_len: int = 600
    question_len: int = 30

    epochs: int = 30
    
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
    print_frequency: int = 100
    print_every: int = 100
    save_every: int = 50000000
    eval_every: int = 1000
    
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
    
    @property
    def tensorboard_dir(self) -> Path:
        """Directory for tensorboard logs."""
        return self.log_root / 'tensorboard'
    
    # def __post_init__(self):
    #     """Create directories if they don't exist."""
    #     self.data_dir.mkdir(parents=True, exist_ok=True)
    #     self.log_root.mkdir(parents=True, exist_ok=True)
    #     self.model_dir.mkdir(parents=True, exist_ok=True)
    #     self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            'data_dir': str(self.data_dir),
            'log_root': str(self.log_root),
            'embedding_filename': self.embedding_filename,
            'embedding_size': self.embedding_size,
            'context_len': self.context_len,
            'question_len': self.question_len,
            'hidden_dim': self.hidden_dim,
            'embedding_dim': self.embedding_dim,
            'decoding_steps': self.decoding_steps,
            'max_dec_steps': self.max_dec_steps,
            'maxout_pool_size': self.maxout_pool_size,
            'model_type': self.model_type,
            'lr': self.lr,
            'dropout_ratio': self.dropout_ratio,
            'max_grad_norm': self.max_grad_norm,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'reg_lambda': self.reg_lambda,
            'print_frequency': self.print_frequency,
            'print_every': self.print_every,
            'save_every': self.save_every,
            'eval_every': self.eval_every,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'use_cuda': self.use_cuda,
            'fp16': self.fp16,
            'gradient_accumulation_steps': self.gradient_accumulation_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        # Convert string paths back to Path objects
        if 'data_dir' in config_dict:
            config_dict['data_dir'] = Path(config_dict['data_dir'])
        if 'log_root' in config_dict:
            config_dict['log_root'] = Path(config_dict['log_root'])
        
        # Filter out computed properties
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.embedding_size > 0, "Embedding size must be positive"
        assert self.embedding_dim > 0, "Embedding dimension must be positive"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.context_len > 0, "Context length must be positive"
        assert self.question_len > 0, "Question length must be positive"
        assert 0 <= self.dropout_ratio < 1, "Dropout ratio must be in [0, 1)"
        assert self.lr > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.decoding_steps > 0, "Decoding steps must be positive"
        assert self.max_dec_steps > 0, "Max decoding steps must be positive"
        assert self.maxout_pool_size > 0, "Maxout pool size must be positive"
        
    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["Model Configuration:"]
        lines.append("=" * 50)
        
        # Group related parameters
        groups = {
            "Paths": ["data_dir", "log_root", "embedding_path"],
            "Model Architecture": ["embedding_size", "embedding_dim", "hidden_dim", 
                                  "decoding_steps", "max_dec_steps", "maxout_pool_size", "model_type"],
            "Input Dimensions": ["context_len", "question_len"],
            "Training": ["lr", "dropout_ratio", "max_grad_norm", "batch_size", 
                        "num_epochs", "reg_lambda"],
            "Logging": ["print_frequency", "print_every", "save_every", "eval_every"],
            "System": ["seed", "num_workers", "use_cuda", "fp16"]
        }
        
        for group_name, params in groups.items():
            lines.append(f"\n{group_name}:")
            lines.append("-" * 30)
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    if isinstance(value, Path):
                        value = str(value)
                    lines.append(f"  {param}: {value}")
        
        return "\n".join(lines)