#!/usr/bin/env python3
"""Download the ISOT Fake News dataset (True.csv / Fake.csv) into ./data."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Official ISOT research lab mirror of the Fake/True news CSVs
# (same files commonly redistributed on Kaggle as clmentbisaillon/fake-and-real-news-dataset).
DEFAULT_URL = (
    "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/"
    "sites/7295/2023/03/News-_dataset.zip"
)


def download(url: str, dest_dir: Path, force: bool = False) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    true_path = dest_dir / "True.csv"
    fake_path = dest_dir / "Fake.csv"

    if true_path.exists() and fake_path.exists() and not force:
        print(f"Dataset already present in {dest_dir.resolve()}")
        return

    zip_path = dest_dir / "News-_dataset.zip"
    print(f"Downloading ISOT Fake News dataset from:\n  {url}")
    urlretrieve(url, zip_path)

    print(f"Extracting into {dest_dir.resolve()}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    zip_path.unlink(missing_ok=True)

    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Expected True.csv and Fake.csv in {dest_dir}, extraction may have failed."
        )

    print("Done.")
    print(f"  {true_path} ({true_path.stat().st_size:,} bytes)")
    print(f"  {fake_path} ({fake_path.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory to place True.csv and Fake.csv (default: ./data)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Zip URL for the ISOT News dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if CSV files already exist",
    )
    args = parser.parse_args()

    try:
        download(args.url, args.dest, force=args.force)
    except Exception as exc:  # noqa: BLE001 - surface clear CLI error
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "\nFallback: download manually from Kaggle "
            "(clmentbisaillon/fake-and-real-news-dataset) or the ISOT lab page, "
            "then place True.csv and Fake.csv under ./data/",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
