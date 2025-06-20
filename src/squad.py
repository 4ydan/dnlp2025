import os
import torch
from torch.utils.data import Dataset
from src.config import DCNConfig
from datasets import load_dataset
import re


class SquadDataset(Dataset):
    """
    Constructor function that loads the specified SQuAD dataset split.
    
    Args:
        word2idx (dict): A dictionary mapping words to indices.
        split (str): The dataset split to load: "train" or "validation".
    """

    def __init__(self, word2idx, split="train"):
        """
        constructor function
        """
        super(SquadDataset, self).__init__()
        
        self.config = DCNConfig()
        self.context_data = []
        self.question_data = []
        self.answer_data = []
        self.answer_span_data = []
        
        # Use the global word2idx dictionary from your preprocessing
        self.word2idx = word2idx

        # load squad dataset from huggingface
        if split not in ["train", "validation"]:
            raise ValueError(f"Invalid split: {split}. Choose 'train' or 'validation'.")

        # Load the specified split of the SQuAD dataset
        squad_dataset = load_dataset("squad")[split]

        # Filter out contexts longer than config.context_len tokens for training split
        if split == "train":
            squad_dataset = squad_dataset.filter(lambda x: len(self.simple_tokenize(x["context"])) <= self.config.context_len)

        # store raw contexts, questions, and answer spans as strings
        self.context_data = [example["context"] for example in squad_dataset]
        self.question_data = [example["question"] for example in squad_dataset]

        # Process answer spans to convert from character indices to token indices
        self.answer_span_data = [self._get_token_span(example) for example in squad_dataset]

    def __len__(self):
        return len(self.answer_span_data)

    def _padding(self, token_ids, max_len):
        sent_len = len(token_ids)
        if sent_len > max_len:
            return token_ids[:max_len]

        token_ids = token_ids + (max_len - sent_len) * [0]  # Use 0 for <PAD>
        return token_ids
    
    def simple_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    
    def _get_token_span(self, example):
        """
        Convert character-based answer spans to token indices for SQuAD dataset examples.
        
        Args:
            example: A SQuAD dataset example containing 'context', 'answers', and tokenized data
            
        Returns:
            list: start and end token index
        """
        context = example['context']
        answers = example['answers']
        
        # Handle cases where there are no answers (unanswerable questions)
        if not answers['text'] or len(answers['text']) == 0:
            raise Exception("Not answerable")
        
        # Get the first answer (SQuAD can have multiple answer annotations)
        answer_text = answers['text'][0]
        answer_start_char = answers['answer_start'][0]
        answer_end_char = answer_start_char + len(answer_text)
        
        # Simple whitespace tokenization fallback
        tokens = context.split()
        
        # Create character to token mapping
        char_to_token = {}
        char_pos = 0
        
        for token_idx, token in enumerate(tokens):
            # Find token position in context
            token_start = context.find(token, char_pos)
            if token_start != -1:
                token_end = token_start + len(token)
                for char_idx in range(token_start, token_end):
                    if char_idx < len(context):
                        char_to_token[char_idx] = token_idx
                char_pos = token_end
        
        # Find start and end token indices
        start_token = char_to_token.get(answer_start_char, 0)
        end_token = char_to_token.get(answer_end_char - 1, len(tokens) - 1)
        
        return start_token, end_token

    def sentence_tokenids(self, sentence, max_len):
        """
        Convert a sentence to token IDs and pad to specified length.
        
        Args:
            sentence (str): Input sentence to tokenize
            max_len (int): Maximum sequence length for padding/truncation
            
        Returns:
            tuple: (padded_token_ids, actual_length)
                - padded_token_ids (list): Token IDs padded/truncated to max_len
                - actual_length (int): Original sequence length (capped at max_len)
        """
        tokens = self.simple_tokenize(sentence)
        token_ids = [self.word2idx.get(word, 0) for word in tokens]
        padded = self._padding(token_ids, max_len)
        return padded, min(len(token_ids), max_len)

    def __getitem__(self, index):
        context = self.context_data[index]
        context_ids, context_len = self.sentence_tokenids(
            context, max_len=self.config.context_len)
        question = self.question_data[index]
        question_ids, question_len = self.sentence_tokenids(
            question, max_len=self.config.question_len)
        
        start_idx, end_idx = self.answer_span_data[index]
        answer_span = torch.LongTensor([start_idx, end_idx])

        return (
                torch.LongTensor(context_ids),
                torch.LongTensor([context_len]),
                torch.LongTensor(question_ids),
                torch.LongTensor([question_len]),
                answer_span
        )