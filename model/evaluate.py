"""
Evaluation script for Arabic Sentiment Analysis.

Runs the best saved model on the test set and reports:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model.dataset import create_data_loaders
from model.model import ArabicSentimentModel
from model.utils import get_device, set_seed
from model.preprocessing import LABEL_NAMES

# --- Configuration ---
BATCH_SIZE = 8
MAX_LEN = 64
DATA_PATH = 'model/data/processed/reviews_cleaned.csv'
MODEL_PATH = 'sentiment_model/best_model.pt'


def get_predictions(model, data_loader, device):
    """Run model on all batches and collect predictions + true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    set_seed(42)
    device = get_device()

    # 1. Load Data (we only need the test loader)
    _, _, test_data_loader, _ = create_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )
    print(f"\nTest set size: {len(test_data_loader.dataset)}")

    # 2. Load Best Model
    model = ArabicSentimentModel(n_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print("✅ Best model loaded successfully\n")

    # 3. Get Predictions
    preds, labels = get_predictions(model, test_data_loader, device)

    # 4. Overall Accuracy
    acc = accuracy_score(labels, preds)
    print(f"{'='*50}")
    print(f"  Overall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}\n")

    # 5. Per-Class Report (Precision, Recall, F1)
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    report = classification_report(labels, preds, target_names=target_names)
    print("📊 Classification Report:")
    print(report)

    # 6. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("📋 Confusion Matrix:")
    print(f"{'':>12} {'Pred Neg':>10} {'Pred Neu':>10} {'Pred Pos':>10}")
    for i, row in enumerate(cm):
        print(f"{'True '+target_names[i]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    print(f"\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
