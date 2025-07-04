import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
import logging

class GloVeEmbeddings:
    """
    GloVe embeddings loader and manager for SQuAD dataset and coattention models.
    
    Features:
    - Loads pretrained GloVe embeddings
    - Handles vocabulary mapping
    - Provides embedding matrices for neural networks
    - Supports out-of-vocabulary (OOV) tokens
    - Optimized for SQuAD dataset preprocessing
    """
    
    def __init__(self, embedding_dim: int = 50):
        """
        Initialize GloVe embeddings loader.
        
        Args:
            embedding_dim: Dimension of GloVe embeddings (50, 100, 200, 300)
            cache_dir: Directory to cache processed embeddings
        """
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_glove_embeddings(self, glove_file_path: str, vocab: Optional[set] = None) -> None:
        """
        Load GloVe embeddings from file.
        
        Args:
            glove_file_path: Path to GloVe embeddings file (e.g., glove.6B.300d.txt)
            vocab: Optional vocabulary set to filter embeddings (improves memory usage)
        """
        
        self.logger.info(f"Loading GloVe embeddings from {glove_file_path}")
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Read GloVe file
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % 100000 == 0:
                    self.logger.info(f"Processed {line_idx} lines")
                
                values = line.split(' ')
                word = values[0]
                vector_values = values[1:]
                
                # Skip malformed lines
                if len(vector_values) != self.embedding_dim:
                    print(f"Skipping line {line_idx} due to unexpected vector length: word: {word}, values length: {len(vector_values)}")
                    continue

                # Filter by vocabulary if provided
                if vocab is not None and word not in vocab:
                    continue
                
                try:
                    vector = np.array(vector_values, dtype=np.float32)
                    self._add_word(word, vector)
                except ValueError:
                    print(f"ValueError occured when loading embeddings: word: {word}, len(vector_values): {len(vector_values)}")
        
        self._finalize_embeddings()
    
        
        self.logger.info(f"Loaded {self.vocab_size} words with {self.embedding_dim}d embeddings")
    
    def _init_special_tokens(self) -> None:
        """Initialize special tokens with random embeddings."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        
        for token in special_tokens:
            # PAD token gets zero embedding
            if token == self.PAD_TOKEN:
                vector = np.zeros(self.embedding_dim, dtype=np.float32)
            elif token == self.UNK_TOKEN:
                vector = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                # Other special tokens get random embeddings
                vector = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            
            self._add_word(token, vector)
    
    def _add_word(self, word: str, vector: np.ndarray) -> None:
        """Add word and its embedding vector to the vocabulary."""
        if word == self.UNK_TOKEN:
            idx = 1
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        if word == self.PAD_TOKEN:
            idx = 0
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def _finalize_embeddings(self) -> None:
        """Convert embeddings to numpy array and finalize vocabulary."""
        self.vocab_size = len(self.word_to_idx)
        self.embeddings = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)
        
        # Re-read to populate embeddings array (for filtered vocab case)
        for word, idx in self.word_to_idx.items():
            if word in [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                if word == self.PAD_TOKEN:
                    self.embeddings[idx] = np.zeros(self.embedding_dim)
                else:
                    self.embeddings[idx] = np.random.normal(0, 0.1, self.embedding_dim)
    
    def get_embedding_matrix(self) -> np.ndarray:
        """
        Get the full embedding matrix for use in neural network embedding layers.
        
        Returns:
            numpy array of shape (vocab_size, embedding_dim)
        """
        return self.embeddings
    
    def word_to_index(self, word: str) -> int:
        """
        Convert word to index, return UNK index if not found.
        
        Args:
            word: Input word
            
        Returns:
            Index of the word in vocabulary
        """
        return self.word_to_idx.get(word, self.word_to_idx[self.UNK_TOKEN])
    
    def index_to_word(self, idx: int) -> str:
        """
        Convert index to word.
        
        Args:
            idx: Index in vocabulary
            
        Returns:
            Word corresponding to the index
        """
        return self.idx_to_word.get(idx, self.UNK_TOKEN)
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to sequence of indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (pad or truncate)
            
        Returns:
            List of word indices
        """
        words = text.lower().split()
        indices = [self.word_to_index(word) for word in words]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                pad_idx = self.word_to_idx[self.PAD_TOKEN]
                indices.extend([pad_idx] * (max_length - len(indices)))
        
        return indices
    
    def decode_indices(self, indices: List[int]) -> str:
        """
        Decode sequence of indices back to text.
        
        Args:
            indices: List of word indices
            
        Returns:
            Decoded text string
        """
        words = [self.index_to_word(idx) for idx in indices 
                if idx != self.word_to_idx[self.PAD_TOKEN]]
        return ' '.join(words)
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding vector for a specific word.
        
        Args:
            word: Input word
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        idx = self.word_to_index(word)
        return self.embeddings[idx]
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'pad_idx': self.word_to_idx[self.PAD_TOKEN],
            'unk_idx': self.word_to_idx[self.UNK_TOKEN],
            'start_idx': self.word_to_idx[self.START_TOKEN],
            'end_idx': self.word_to_idx[self.END_TOKEN]
        }
