"""
PyTorch Dataset class for Arabic Sentiment Analysis.

Handles:
- Loading processed CSV data
- Splitting into train/val/test
- Tokenization using AraBERT tokenizer
- Preparing tensors for model input
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# AraBERT v0.2 base model
MODEL_NAME = "aubmindlab/bert-base-arabertv02"

class ArabicSentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for Arabic Sentiment Analysis using AraBERT.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        """
        Args:
            texts: List of review texts.
            labels: List of sentiment labels (0, 1, 2).
            tokenizer: Pre-trained AraBERT tokenizer.
            max_len: Maximum token sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def create_data_loaders(
    data_path: str,
    batch_size: int = 16,
    max_len: int = 128,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AutoTokenizer]:
    """
    Load data, split into train/val/test, and return DataLoaders.

    Args:
        data_path: Path to processed CSV file.
        batch_size: Batch size for loaders.
        max_len: Max sequence length for tokenizer.
        test_size: Proportion of dataset to include in test split.
        val_size: Proportion of training set to include in validation split.
        random_state: Random seed for splitting.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    df = pd.read_csv(data_path)
    
    # Handle NaN values explicitly if any slipped through
    df = df.dropna(subset=['text', 'label'])
    
    # Stratified split to maintain class balance
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Split train further into train/val
    train_df, val_df = train_test_split(
        train_df, test_size=val_size / (1 - test_size), random_state=random_state, stratify=train_df['label']
    )

    print(f"Data Split Summary:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create Datasets
    train_dataset = ArabicSentimentDataset(
        texts=train_df['text'].to_numpy(),
        labels=train_df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_dataset = ArabicSentimentDataset(
        texts=val_df['text'].to_numpy(),
        labels=val_df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    test_dataset = ArabicSentimentDataset(
        texts=test_df['text'].to_numpy(),
        labels=test_df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, tokenizer
