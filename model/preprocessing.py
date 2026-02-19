"""
Arabic text preprocessing module for sentiment analysis.

Handles:
- Arabic text normalization & cleaning
- Rating to sentiment label mapping (3-class)
- Duplicate & short review removal
- Full pipeline from raw data to processed/cleaned CSV
"""

import re
import pandas as pd
from pathlib import Path

# ============================================================
# Arabic Text Cleaning Functions
# ============================================================

def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel): fatHa, damma, kasra, shadda, sukun, etc."""
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670]')
    return diacritics.sub('', text)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic characters to a canonical form."""
    # Normalize alef variants → ا
    text = re.sub(r'[إأآٱ]', 'ا', text)
    # Normalize taa marbouta → ه
    text = re.sub(r'ة', 'ه', text)
    # Normalize alef maqsura → ي
    text = re.sub(r'ى', 'ي', text)
    # Normalize waw with hamza → و
    text = re.sub(r'ؤ', 'و', text)
    # Normalize yaa with hamza → ي
    text = re.sub(r'ئ', 'ي', text)
    return text


def remove_elongation(text: str) -> str:
    """Remove Arabic elongation/stretching (e.g., رااااائع → رائع)."""
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r'\S+@\S+\.\S+', '', text)


def remove_mentions_hashtags(text: str) -> str:
    """Remove @mentions and #hashtags."""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_emojis(text: str) -> str:
    """Remove emojis and special unicode symbols."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # misc
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub('', text)


def remove_special_characters(text: str) -> str:
    """Remove non-Arabic, non-space characters (keep Arabic letters, digits, spaces)."""
    # Keep Arabic letters, Arabic-Indic digits, Western digits, and spaces
    return re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF0-9\s]', ' ', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# Combined Cleaning Pipeline
# ============================================================

def clean_text(text: str) -> str:
    """
    Apply the full Arabic text cleaning pipeline.
    
    Pipeline order:
    1. Remove URLs, emails, mentions, hashtags
    2. Remove emojis
    3. Remove diacritics (tashkeel)
    4. Normalize Arabic letters
    5. Remove elongation
    6. Remove special characters
    7. Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_mentions_hashtags(text)
    text = remove_emojis(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_elongation(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    
    return text


# ============================================================
# Label Mapping
# ============================================================

SENTIMENT_MAP = {
    1: 0,   # Negative
    2: 0,   # Negative
    3: 1,   # Neutral
    4: 2,   # Positive
    5: 2,   # Positive
}

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def map_rating_to_sentiment(rating: int) -> int:
    """Map 1-5 rating to 3-class sentiment (0=Negative, 1=Neutral, 2=Positive)."""
    return SENTIMENT_MAP[rating]


# ============================================================
# Full Preprocessing Pipeline
# ============================================================

def preprocess_dataset(
    input_path: str,
    output_path: str,
    min_words: int = 3,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load raw data → clean → filter → save.

    Args:
        input_path: Path to raw TSV file.
        output_path: Path to save processed CSV.
        min_words: Minimum number of words to keep a review.

    Returns:
        Cleaned DataFrame.
    """
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(
        input_path,
        sep='\t',
        header=None,
        names=['rating', 'review_id', 'user_id', 'book_id', 'review_text'],
    )
    initial_count = len(df)
    print(f"  → Loaded {initial_count} rows")

    # --- Step 1: Remove duplicates ---
    df = df.drop_duplicates(subset=['review_text'], keep='first')
    print(f"  → After removing duplicate texts: {len(df)} rows (removed {initial_count - len(df)})")

    # --- Step 2: Clean text ---
    print("  → Cleaning Arabic text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)

    # --- Step 3: Remove short reviews ---
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    before_filter = len(df)
    df = df[df['word_count'] >= min_words].copy()
    print(f"  → After removing short reviews (< {min_words} words): {len(df)} rows (removed {before_filter - len(df)})")

    # --- Step 4: Remove empty cleaned texts ---
    df = df[df['cleaned_text'].str.strip().astype(bool)].copy()
    print(f"  → After removing empty texts: {len(df)} rows")

    # --- Step 5: Map ratings to sentiment labels ---
    df['label'] = df['rating'].map(SENTIMENT_MAP)
    print(f"  → Label distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = (df['label'] == label_id).sum()
        pct = count / len(df) * 100
        print(f"      {name} ({label_id}): {count} ({pct:.1f}%)")

    # --- Step 6: Keep only needed columns ---
    df_final = df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'})
    df_final = df_final.reset_index(drop=True)

    # --- Step 7: Save ---
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output, index=False)
    print(f"\n✅ Saved {len(df_final)} cleaned reviews to: {output_path}")

    return df_final


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    RAW_PATH = BASE_DIR / "data" / "raw" / "reviews.tsv.txt"
    PROCESSED_PATH = BASE_DIR / "data" / "processed" / "reviews_cleaned.csv"

    df = preprocess_dataset(
        input_path=str(RAW_PATH),
        output_path=str(PROCESSED_PATH),
    )
    print(f"\nFinal dataset shape: {df.shape}")
    print(df.head())
