# Quick Start Guide - RCAEval Dataset Download

## Fixed Download Script ✅

The dataset downloader has been updated to correctly match Zenodo's actual file structure.

---

## Usage Examples

### 1. Download All Systems, All Versions (Recommended for full project)
```bash
cd /path/to/fault-detection-microservices/project
python scripts/download_dataset.py --all
```

**This downloads:**
- 9 zip files total (3 systems × 3 RE versions)
- ~810 failure cases (90 per system per version)
- Total size: ~4-5 GB

**Files downloaded:**
- RE1-TT.zip, RE1-SS.zip, RE1-OB.zip
- RE2-TT.zip, RE2-SS.zip, RE2-OB.zip
- RE3-TT.zip, RE3-SS.zip, RE3-OB.zip

---

### 2. Download Only RE2 (Main Benchmark - 270 cases)
```bash
python scripts/download_dataset.py --all --reversions RE2
```

**This downloads:**
- RE2-TT.zip (TrainTicket - 90 cases)
- RE2-SS.zip (SockShop - 90 cases)
- RE2-OB.zip (OnlineBoutique - 90 cases)
- Total: 270 cases (sufficient for main experiments)

---

### 3. Download Only TrainTicket (All Versions)
```bash
python scripts/download_dataset.py --systems TrainTicket
```

**This downloads:**
- RE1-TT.zip (90 cases)
- RE2-TT.zip (90 cases)
- RE3-TT.zip (90 cases)
- Total: 270 cases for one system

---

### 4. Download Specific System + Version
```bash
python scripts/download_dataset.py --systems TrainTicket --reversions RE2
```

**This downloads:**
- RE2-TT.zip only (90 cases)
- Minimal download for quick testing

---

### 5. Force Re-download (if corrupted)
```bash
python scripts/download_dataset.py --all --force
```

**This:**
- Re-downloads all files even if they exist
- Useful if checksums failed or files corrupted

---

## Directory Structure After Download

```
project/
└── data/
    └── RCAEval/
        ├── downloads/           # Original zip files
        │   ├── RE1-TT.zip
        │   ├── RE1-SS.zip
        │   ├── RE1-OB.zip
        │   ├── RE2-TT.zip
        │   ├── RE2-SS.zip
        │   ├── RE2-OB.zip
        │   ├── RE3-TT.zip
        │   ├── RE3-SS.zip
        │   └── RE3-OB.zip
        │
        ├── TrainTicket/         # Extracted data
        │   ├── RE1/
        │   │   ├── metrics/
        │   │   ├── logs/
        │   │   ├── traces/
        │   │   └── ground_truth.csv
        │   ├── RE2/
        │   └── RE3/
        │
        ├── SockShop/
        │   ├── RE1/
        │   ├── RE2/
        │   └── RE3/
        │
        └── OnlineBoutique/
            ├── RE1/
            ├── RE2/
            └── RE3/
```

---

## What Each Flag Does

| Flag | Description | Example |
|------|-------------|---------|
| `--all` | Download all three systems | `--all` |
| `--systems` | Specific systems only | `--systems TrainTicket SockShop` |
| `--reversions` | Specific RE versions | `--reversions RE2` or `--reversions RE1 RE2` |
| `--no-extract` | Download only, don't extract | `--all --no-extract` |
| `--force` | Re-download even if exists | `--all --force` |
| `--data-dir` | Custom download location | `--data-dir /custom/path` |

---

## Expected Output

When you run the download, you'll see:

```
================================================================================
RCAEval Dataset Downloader
================================================================================
DOI: 10.5281/zenodo.14590730
Destination: data/RCAEval
Systems: TrainTicket, SockShop, OnlineBoutique
RE Versions: RE1, RE2, RE3
================================================================================
📡 Fetching dataset information from Zenodo...
✅ Found 9 files in Zenodo record

================================================================================
📦 TrainTicket - RE1
   41-service train ticket booking system
   File: RE1-TT.zip
================================================================================
   📥 Downloading RE1-TT.zip (456.3 MB)
   URL: https://zenodo.org/api/records/14590730/files/RE1-TT.zip/content...
   RE1-TT: 100%|████████████████████| 456.3M/456.3M [01:23<00:00, 5.47MiB/s]
   Verifying checksum...
   ✅ Checksum verified: a3f7d9e2b4c8f1a5...
   📂 Extracting to data/RCAEval/TrainTicket/RE1...
   Extracting: 100%|████████████████| 523.1M/523.1M [00:45<00:00, 11.6MB/s]
   ✅ Extracted successfully

[... continues for all 9 files ...]

================================================================================
✅ Download Complete!
================================================================================
📊 Summary:
   Downloaded: 9 files
   Skipped (already verified): 0 files
   Extracted: 9 archives
   Total files processed: 9

📁 Dataset location: data/RCAEval
================================================================================

📂 Directory Structure:

  TrainTicket/
    ├── RE1/ (1234 files, 523.1 MB)
    ├── RE2/ (1234 files, 523.1 MB)
    └── RE3/ (1234 files, 523.1 MB)

  SockShop/
    ├── RE1/ (890 files, 412.3 MB)
    ├── RE2/ (890 files, 412.3 MB)
    └── RE3/ (890 files, 412.3 MB)

  OnlineBoutique/
    ├── RE1/ (945 files, 387.9 MB)
    ├── RE2/ (945 files, 387.9 MB)
    └── RE3/ (945 files, 387.9 MB)
```

---

## Troubleshooting

### If download fails with "File not found in Zenodo"

Check the actual files available:
```python
import requests
response = requests.get("https://zenodo.org/api/records/14590730")
files = response.json()['files']
for f in files:
    print(f['key'], f['size'] / (1024**2), 'MB')
```

### If extraction fails

Try downloading without extraction first:
```bash
python scripts/download_dataset.py --all --no-extract
```

Then extract manually:
```bash
cd data/RCAEval/downloads
unzip RE2-TT.zip -d ../TrainTicket/RE2/
```

### If checksums fail repeatedly

Re-download with force flag:
```bash
python scripts/download_dataset.py --all --force
```

---

## After Download - Verify Data

```python
from src.utils.data_loader import RCAEvalDataLoader

# Initialize loader
loader = RCAEvalDataLoader('data/RCAEval')

# This will fail with proper error if dataset not downloaded
try:
    train, val, test = loader.load_splits()
    print(f"✅ Dataset loaded: {len(train)} train, {len(val)} val, {len(test)} test")
except FileNotFoundError as e:
    print(f"❌ Dataset not found: {e}")
```

---

## Recommended Download Strategy

For **full project with all ablations**:
```bash
python scripts/download_dataset.py --all
```

For **quick testing and development**:
```bash
python scripts/download_dataset.py --systems TrainTicket --reversions RE2
```

For **main benchmark experiments** (matches literature):
```bash
python scripts/download_dataset.py --all --reversions RE2
```

---

## Next Steps

Once download completes:
1. ✅ Verify data loaded correctly
2. ✅ Run EDA notebooks to explore data
3. ✅ Begin baseline implementations
4. ✅ Start model training

Let me know when download completes successfully! 🚀
