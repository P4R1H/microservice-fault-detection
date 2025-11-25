#!/usr/bin/env python3
"""
Extract RCAEval Dataset from Pre-downloaded Zip Files

This script extracts the RCAEval dataset from manually downloaded zip files
located in data/zip_cache/ and organizes them into the proper structure.

Usage:
    python scripts/download_dataset.py --all
    python scripts/download_dataset.py --systems TrainTicket
    python scripts/download_dataset.py --reversions RE2
"""

import sys
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


# System name mapping
SYSTEM_MAPPING = {
    "TT": "TrainTicket",
    "SS": "SockShop",
    "OB": "OnlineBoutique"
}

REVERSE_MAPPING = {
    "TrainTicket": "TT",
    "SockShop": "SS",
    "OnlineBoutique": "OB"
}

SYSTEM_INFO = {
    "TrainTicket": {
        "code": "TT",
        "description": "41-service train ticket booking system",
        "services": 41
    },
    "SockShop": {
        "code": "SS",
        "description": "13-service e-commerce demo",
        "services": 13
    },
    "OnlineBoutique": {
        "code": "OB",
        "description": "12-service Google demo application",
        "services": 12
    }
}

RE_VERSIONS = ["RE1", "RE2", "RE3"]


def extract_zip(zip_path: Path, extract_base_dir: Path, system: str, re_version: str):
    """
    Extract zip file to proper directory structure

    Args:
        zip_path: Path to zip file
        extract_base_dir: Base directory for extraction (data/RCAEval)
        system: System name (TrainTicket, SockShop, OnlineBoutique)
        re_version: RE version (RE1, RE2, RE3)

    Returns:
        Path to extracted directory or None if failed
    """
    extract_dir = extract_base_dir / system / re_version
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"   📂 Extracting to {extract_dir}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total uncompressed size
            total_size = sum(info.file_size for info in zip_ref.infolist())

            # Extract with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="   Extracting") as pbar:
                for member in zip_ref.infolist():
                    try:
                        zip_ref.extract(member, extract_dir)
                        pbar.update(member.file_size)
                    except Exception as e:
                        print(f"      ⚠️  Warning: Could not extract {member.filename}: {e}")

        # Verify extraction
        extracted_files = list(extract_dir.rglob('*'))
        file_count = sum(1 for f in extracted_files if f.is_file())
        total_size_mb = sum(f.stat().st_size for f in extracted_files if f.is_file()) / (1024**2)

        if file_count > 0:
            print(f"   ✅ Extracted successfully")
            print(f"      Files: {file_count}, Size: {total_size_mb:.1f} MB")
            return extract_dir
        else:
            print(f"   ❌ No files extracted")
            return None

    except zipfile.BadZipFile:
        print(f"   ❌ Bad zip file: {zip_path}")
        return None
    except Exception as e:
        print(f"   ❌ Extraction error: {e}")
        return None


def extract_rcaeval_dataset(
    systems: list,
    re_versions: list,
    zip_cache_dir: Path,
    data_dir: Path,
    force: bool = False
):
    """
    Extract RCAEval dataset from pre-downloaded zip files

    Args:
        systems: List of system names ('TrainTicket', 'SockShop', 'OnlineBoutique')
        re_versions: List of RE versions to extract ('RE1', 'RE2', 'RE3')
        zip_cache_dir: Directory containing downloaded zip files
        data_dir: Base directory for extracted data (data/RCAEval)
        force: Re-extract even if already extracted
    """
    print("=" * 80)
    print("RCAEval Dataset Extraction")
    print("=" * 80)
    print(f"Zip cache: {zip_cache_dir}")
    print(f"Destination: {data_dir}")
    print(f"Systems: {', '.join(systems)}")
    print(f"RE Versions: {', '.join(re_versions)}")
    print("=" * 80)

    # Verify zip cache exists
    if not zip_cache_dir.exists():
        print(f"\n❌ Error: Zip cache directory not found: {zip_cache_dir}")
        print(f"   Please ensure zip files are in: {zip_cache_dir.absolute()}")
        return False

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    total_requested = len(systems) * len(re_versions)

    # Extract each requested combination
    for system in systems:
        if system not in SYSTEM_INFO:
            print(f"\n⚠️  Unknown system: {system}")
            failed_count += 1
            continue

        system_code = SYSTEM_INFO[system]['code']
        system_desc = SYSTEM_INFO[system]['description']

        for re_version in re_versions:
            # Construct expected filename: RE1-TT.zip, RE2-SS.zip, etc.
            zip_filename = f"{re_version}-{system_code}.zip"
            zip_path = zip_cache_dir / zip_filename

            print(f"\n{'='*80}")
            print(f"📦 {system} - {re_version}")
            print(f"   {system_desc}")
            print(f"   File: {zip_filename}")
            print(f"{'='*80}")

            # Check if zip file exists
            if not zip_path.exists():
                print(f"   ❌ Zip file not found: {zip_path}")
                print(f"   Please download from Zenodo: https://zenodo.org/record/14590730")
                failed_count += 1
                continue

            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"   📦 Found: {zip_filename} ({zip_size_mb:.1f} MB)")

            # Check if already extracted
            extract_target = data_dir / system / re_version
            if not force and extract_target.exists() and any(extract_target.iterdir()):
                extracted_files = list(extract_target.rglob('*'))
                file_count = sum(1 for f in extracted_files if f.is_file())
                total_size_mb = sum(f.stat().st_size for f in extracted_files if f.is_file()) / (1024**2)

                print(f"   ✅ Already extracted to {extract_target}")
                print(f"      Files: {file_count}, Size: {total_size_mb:.1f} MB")
                print(f"   ⏭️  Skipping (use --force to re-extract)")
                skipped_count += 1
                continue

            # Extract
            extract_dir = extract_zip(zip_path, data_dir, system, re_version)
            if extract_dir:
                extracted_count += 1
            else:
                failed_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("✅ Extraction Complete!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   Total requested: {total_requested}")
    print(f"   Extracted: {extracted_count}")
    print(f"   Skipped (already exists): {skipped_count}")
    print(f"   Failed: {failed_count}")
    print(f"\n📁 Dataset location: {data_dir}")
    print("=" * 80)

    # Show directory structure
    print(f"\n📂 Directory Structure:")
    for system in systems:
        system_dir = data_dir / system
        if system_dir.exists():
            print(f"\n  {system}/")
            for re_version in re_versions:
                re_dir = system_dir / re_version
                if re_dir.exists():
                    files = list(re_dir.rglob('*'))
                    file_count = sum(1 for f in files if f.is_file())
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    print(f"    ├── {re_version}/ ({file_count} files, {total_size / (1024**2):.1f} MB)")

    # Show what's in zip cache
    print(f"\n📦 Zip Cache ({zip_cache_dir}):")
    if zip_cache_dir.exists():
        zip_files = sorted(zip_cache_dir.glob('*.zip'))
        if zip_files:
            for zf in zip_files:
                size_mb = zf.stat().st_size / (1024 * 1024)
                print(f"   • {zf.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️  No zip files found in {zip_cache_dir}")

    success = (extracted_count + skipped_count) > 0 and failed_count == 0
    return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract RCAEval Dataset from Pre-downloaded Zip Files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all systems, all RE versions
  python scripts/download_dataset.py --all

  # Extract only RE2 versions
  python scripts/download_dataset.py --all --reversions RE2

  # Extract specific system and version
  python scripts/download_dataset.py --systems TrainTicket --reversions RE2

  # Force re-extraction even if already extracted
  python scripts/download_dataset.py --all --force

File naming convention:
  - RE1-TT.zip = Round 1, TrainTicket
  - RE2-SS.zip = Round 2, SockShop
  - RE3-OB.zip = Round 3, OnlineBoutique

Expected zip files location:
  data/zip_cache/
    ├── RE1-TT.zip
    ├── RE1-SS.zip
    ├── RE1-OB.zip
    ├── RE2-TT.zip
    ├── RE2-SS.zip
    ├── RE2-OB.zip
    ├── RE3-TT.zip
    ├── RE3-SS.zip
    └── RE3-OB.zip

Extracted structure:
  data/RCAEval/
    ├── TrainTicket/
    │   ├── RE1/
    │   ├── RE2/
    │   └── RE3/
    ├── SockShop/
    │   ├── RE1/
    │   ├── RE2/
    │   └── RE3/
    └── OnlineBoutique/
        ├── RE1/
        ├── RE2/
        └── RE3/
        """
    )

    parser.add_argument(
        '--systems',
        nargs='+',
        choices=['TrainTicket', 'SockShop', 'OnlineBoutique'],
        help='Systems to extract (default: all if --all specified)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Extract all three systems'
    )

    parser.add_argument(
        '--reversions',
        nargs='+',
        choices=['RE1', 'RE2', 'RE3'],
        default=['RE1', 'RE2', 'RE3'],
        help='RE versions to extract (default: all three)'
    )

    parser.add_argument(
        '--zip-cache',
        type=Path,
        default=Path('data/zip_cache'),
        help='Directory containing downloaded zip files (default: data/zip_cache)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/RCAEval'),
        help='Extraction directory (default: data/RCAEval)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-extract even if already extracted'
    )

    args = parser.parse_args()

    # Determine which systems to extract
    if args.all:
        systems = list(SYSTEM_INFO.keys())
    elif args.systems:
        systems = args.systems
    else:
        print("No systems specified. Use --systems or --all")
        print("Example: python scripts/download_dataset.py --all")
        return 1

    # Extract dataset
    success = extract_rcaeval_dataset(
        systems=systems,
        re_versions=args.reversions,
        zip_cache_dir=args.zip_cache,
        data_dir=args.data_dir,
        force=args.force
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
