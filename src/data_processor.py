import json
from typing import List, Dict
from datasets import Dataset
from transformers import PreTrainedTokenizer

class ConversationDataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation into a single string with special tokens."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>{content}</s>"
            elif role == "user":
                formatted += f"<|user|>{content}</s>"
            elif role == "assistant":
                formatted += f"<|assistant|>{content}</s>"
        return formatted

    def load_conversations(self, file_path: str) -> Dataset:
        """Load and preprocess conversation data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for conv in raw_data:
            formatted_conv = self.format_conversation(conv['messages'])
            processed_data.append({
                'text': formatted_conv,
                'conversation_id': conv.get('conversation_id', ''),
                'metadata': conv.get('metadata', {})
            })
            
        return Dataset.from_list(processed_data)

    def tokenize_conversations(self, dataset: Dataset) -> Dataset:
        """Tokenize the conversations dataset."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized
