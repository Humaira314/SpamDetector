from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
RAW_FILENAME = "SMSSpamCollection"


def download_dataset(dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "smsspamcollection.zip"

    if zip_path.exists() and not force:
        return zip_path

    urlretrieve(DATASET_URL, zip_path)
    return zip_path


def extract_dataset(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    return dest_dir / RAW_FILENAME


def prepare_csv(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path, sep="\t", names=["label", "text"], encoding="latin-1")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def maybe_download_nltk(download_nltk: bool) -> None:
    if not download_nltk:
        return
    try:
        import nltk

        nltk.download("stopwords")
    except Exception:
        print("NLTK download failed. You can run: python -c \"import nltk; nltk.download('stopwords')\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the SMS Spam dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset zip")
    parser.add_argument("--prepare", action="store_true", help="Prepare CSV dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", default="data/processed/sms.csv", help="Output CSV path")
    parser.add_argument("--download-nltk", action="store_true", help="Download NLTK stopwords")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_path = raw_dir / RAW_FILENAME
    output_path = Path(args.output)

    if args.download:
        zip_path = download_dataset(raw_dir, force=args.force)
        raw_path = extract_dataset(zip_path, raw_dir)
    elif not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run with --download or place the file manually."
        )

    if args.prepare:
        prepare_csv(raw_path, output_path)
        print(f"Wrote {output_path}")

    maybe_download_nltk(args.download_nltk)


if __name__ == "__main__":
    main()
