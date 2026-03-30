"""
Training script for Arabic Sentiment Analysis (AraBERT).

Optimized for GTX 1650 Ti (4GB VRAM) with:
- Frozen BERT layers (only last 2 layers trainable)
- Mixed precision (fp16) training
- Gradient accumulation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from model.dataset import create_data_loaders
from model.model import ArabicSentimentModel
from model.utils import set_seed, get_device, setup_logging

# --- Configuration ---
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2        # Effective batch size = 8 * 2 = 16
MAX_LEN = 64
EPOCHS = 4
LR = 2e-5
DATA_PATH = 'model/data/processed/reviews_cleaned.csv'
MODEL_SAVE_PATH = 'sentiment_model/best_model.pt'

logger = setup_logging()


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, scaler):
    model.train()
    losses = []
    correct_predictions = 0

    optimizer.zero_grad()

    for batch_idx, d in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)
            loss = loss / GRAD_ACCUM_STEPS

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item() * GRAD_ACCUM_STEPS)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def main():
    set_seed(42)
    device = get_device()
    torch.cuda.empty_cache()

    logger.info("🚀 Starting training preparation...")

    # 1. Load Data
    train_data_loader, val_data_loader, test_data_loader, tokenizer = create_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )
    logger.info(f"Initialized DataLoaders. Train size: {len(train_data_loader.dataset)}")

    # 2. Init Model
    model = ArabicSentimentModel(n_classes=3)
    model = model.to(device)

    # 3. Only optimize TRAINABLE parameters (unfrozen layers + head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR)
    total_steps = (len(train_data_loader) // GRAD_ACCUM_STEPS) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler('cuda')

    # 4. Training Loop
    best_accuracy = 0

    logger.info(f"🔥 Starting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler,
            len(train_data_loader.dataset), scaler
        )
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device,
            len(val_data_loader.dataset)
        )
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

        if val_acc > best_accuracy:
            Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            tokenizer.save_pretrained("sentiment_model/")
            best_accuracy = val_acc
            logger.info(f"✅ New best model saved with accuracy: {best_accuracy:.4f}")

    total_time = time.time() - start_time
    logger.info(f"🏁 Training complete in {total_time:.1f}s. Best Val Accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
