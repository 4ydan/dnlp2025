import os
import torch
from torch.utils.data import Dataset
from src.config import DCNConfig

config = DCNConfig()

class SquadDataset(Dataset):
    """
    Squad dataset
    """

    def __init__(self, word2idx, split="train"):
        """
        constructor function
        """
        super(SquadDataset, self).__init__()

        self.context_data = []
        self.question_data = []
        self.answer_data = []
        self.answer_span_data = []
        
        # Use the global word2idx dictionary from your preprocessing
        self.word2idx = word2idx
        
        # Determine file paths based on split
        prefix = "train" if split == "train" else "eval"
        context_path = os.path.join(config.data_dir, f"{prefix}.context")
        question_path = os.path.join(config.data_dir, f"{prefix}.question")
        answer_path = os.path.join(config.data_dir, f"{prefix}.answer")
        span_path = os.path.join(config.data_dir, f"{prefix}.span")
        
        # Read each file line-by-line
        with open(context_path, 'r', encoding='utf-8') as f:
            self.context_data.extend([line.strip() for line in f if line.strip()])

        with open(question_path, 'r', encoding='utf-8') as f:
            self.question_data.extend([line.strip() for line in f if line.strip()])

        with open(answer_path, 'r', encoding='utf-8') as f:
            self.answer_data.extend([line.strip() for line in f if line.strip()])

        with open(span_path, 'r', encoding='utf-8') as f:
            self.answer_span_data.extend([
                [int(x) for x in line.strip().split()] for line in f if line.strip()
            ])

    def __len__(self):
        return len(self.answer_span_data)

    def _padding(self, token_ids, max_len):
        sent_len = len(token_ids)
        if sent_len > max_len:
            return token_ids[:max_len]

        token_ids = token_ids + (max_len - sent_len) * [0]  # Use 0 for <PAD>
        return token_ids

    def tokens_to_ids(self, tokens, max_len):
        """
        Convert tokens to token ids using the word2idx dictionary
        """
        token_ids = []
        for token in tokens:
            if token.lower() in self.word2idx:
                token_ids.append(self.word2idx[token.lower()])
            else:
                token_ids.append(self.word2idx["<UNK>"])  # Use 1 for <UNK>
        
        padded_token_ids = self._padding(token_ids, max_len=max_len)
        return padded_token_ids, len(token_ids)

    def __getitem__(self, index):
        """
        Get item at index
        """
        # Get tokenized data (already lists of tokens)
        context_tokens = self.context_data[index]
        question_tokens = self.question_data[index]
        answer_tokens = self.answer_data[index]
        answer_span = self.answer_span_data[index]
        
        # Convert tokens to IDs
        context_ids, context_len = self.tokens_to_ids(
            context_tokens, max_len=config.context_len)
        question_ids, question_len = self.tokens_to_ids(
            question_tokens, max_len=config.question_len)
        
        # Convert span to start and end indices
        start_idx, end_idx = answer_span
        
        return (
            torch.LongTensor(context_ids), 
            torch.LongTensor([context_len]), 
            torch.LongTensor(question_ids), 
            torch.LongTensor([question_len]), 
            torch.LongTensor([start_idx, end_idx])
        )