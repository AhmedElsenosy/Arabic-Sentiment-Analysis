"""
Model definition for Arabic Sentiment Analysis.

Wraps AraBERT with a custom classification head.
Freezes lower BERT layers to save GPU memory (optimized for 4GB VRAM).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

# AraBERT v0.2 base model
MODEL_NAME = "aubmindlab/bert-base-arabertv02"


class ArabicSentimentModel(nn.Module):
    """
    AraBERT-based model for 3-class sentiment analysis.
    
    Freezes all BERT layers EXCEPT the last 2 encoder layers and the pooler.
    This dramatically reduces memory usage while keeping good performance.
    """
    def __init__(self, n_classes: int = 3, dropout_rate: float = 0.3):
        super(ArabicSentimentModel, self).__init__()

        # Load pretrained AraBERT
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # === Freeze all BERT parameters first ===
        for param in self.bert.parameters():
            param.requires_grad = False

        # === Unfreeze ONLY the last 2 encoder layers + pooler ===
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Classification head (always trainable)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        # Print trainable vs frozen params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Total params: {total:,}")
        print(f"🔥 Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"🧊 Frozen params: {total - trainable:,}")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)
