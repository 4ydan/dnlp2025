import os
import torch
from torch.utils.data import Dataset
from src.config import DCNConfig
from datasets import load_dataset
import re


class SquadDataset(Dataset):
    """
    Squad dataset
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
        squad_dataset = load_dataset("squad")["train"]

        # store raw contexts, questions, and answer spans as strings
        self.context_data = [example["context"] for example in squad_dataset]
        self.quesiton_data = [example["question"] for example in squad_dataset]

        # for answer span, store first answer_start as int, convert to string for LT in __getitem__
        self.answer_span_data = [
            (example["answers"]["answer_start"][0], example["answers"]["answer_start"][0] + len(example["answers"]["text"][0]))
            if example["answers"]["answer_start"] and example["answers"]["text"] else (0, 1)
            for example in squad_dataset
        ]

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

    def sentence_tokenids(self, sentence, max_len):
        tokens = self.simple_tokenize(sentence)
        token_ids = [self.word2idx.get(word, 0) for word in tokens]
        padded = self._padding(token_ids, max_len)
        return padded, min(len(token_ids), max_len)

    def __getitem__(self, index):
            """
            """
            context = self.context_data[index]
            context_ids, context_len = self.sentence_tokenids(
                context, max_len=self.config.context_len)
            question = self.quesiton_data[index]
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