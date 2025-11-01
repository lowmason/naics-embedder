import torch
from torch.utils.data import Dataset

class NaicsDataset(Dataset):
    """
    A PyTorch Dataset for NAICS classification.
    It takes raw text and labels and returns tokenized data.
    """
    def __init__(self, texts, labels, tokenizer, max_seq_length=512):
        """
        Args:
            texts (list[str]): A list of raw text strings.
            labels (list[int]): A list of corresponding integer labels.
            tokenizer: A Hugging Face style tokenizer.
            max_seq_length (int): The maximum sequence length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Fetches one sample from the dataset at the specified index.
        
        Args:
            idx (int): The index of the sample to fetch.
            
        Returns:
            dict: A dictionary containing:
                  - 'input_ids': Token IDs
                  - 'attention_mask': Attention mask
                  - 'labels': The integer label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_seq_length,
            padding='max_length',     # Pad to max_length
            truncation=True,          # Truncate to max_length
            return_attention_mask=True,
            return_tensors='pt',      # Return PyTorch tensors
        )

        # Squeeze tensors to remove the batch dimension (as this is for one item)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
