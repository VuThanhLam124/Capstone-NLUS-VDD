"""
Data Split Script for Research Pipeline
Splits finetune_data.csv into train/dev/test sets with 80/10/10 ratio.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "research_pipeline" / "finetune_data.csv"
OUTPUT_DIR = REPO_ROOT / "research_pipeline" / "data"

# Config
SEED = 42
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
    print(f"Total samples: {len(df)}")
    
    # First split: train + (dev+test)
    train_df, temp_df = train_test_split(
        df, test_size=(DEV_RATIO + TEST_RATIO), random_state=SEED, shuffle=True
    )
    
    # Second split: dev + test
    dev_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (DEV_RATIO + TEST_RATIO), random_state=SEED
    )
    
    # Save splits
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    dev_df.to_csv(OUTPUT_DIR / "dev.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    print(f"Train: {len(train_df)} samples -> {OUTPUT_DIR / 'train.csv'}")
    print(f"Dev: {len(dev_df)} samples -> {OUTPUT_DIR / 'dev.csv'}")
    print(f"Test: {len(test_df)} samples -> {OUTPUT_DIR / 'test.csv'}")

if __name__ == "__main__":
    main()
