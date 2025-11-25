#!/bin/bash
#
# Master script to generate ALL visualizations and tables
#
# This script runs all generation scripts in sequence and produces
# a complete set of publication-ready figures, tables, and diagrams.
#
# Usage:
#   bash generate_everything.sh
#
# Output:
#   - figures/: 10 result visualizations (PNG + PDF)
#   - diagrams/: 4 architecture diagrams (PNG + PDF)
#   - tables/: 9 result tables (CSV + MD + TEX)
#

echo "================================================================================"
echo "GENERATING ALL VISUALIZATIONS FOR PROJECT"
echo "================================================================================"
echo ""
echo "This will generate:"
echo "  - 10 result figures (performance, ablations, comparisons)"
echo "  - 4 architecture diagrams (system, pipeline, fusion, training)"
echo "  - 9 result tables (baselines, ablations, statistics)"
echo ""
echo "Total output: 23 visualizations + 27 files (PNG/PDF/CSV/MD/TEX formats)"
echo ""
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found. Please install Python 3."
    exit 1
fi

# Check required packages
echo "Checking Python dependencies..."
python3 -c "import matplotlib, seaborn, numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Some Python packages missing. Installing..."
    pip install matplotlib seaborn numpy pandas tabulate || {
        echo "❌ Failed to install packages. Please run:"
        echo "   pip install matplotlib seaborn numpy pandas tabulate"
        exit 1
    }
fi

echo "✓ All dependencies satisfied"
echo ""

# Step 1: Generate Result Figures
echo "================================================================================"
echo "STEP 1/3: Generating Result Figures (10 figures)"
echo "================================================================================"
python3 generate_all_figures.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to generate figures"
    exit 1
fi
echo ""

# Step 2: Generate Architecture Diagrams
echo "================================================================================"
echo "STEP 2/3: Generating Architecture Diagrams (4 diagrams)"
echo "================================================================================"
python3 generate_architecture_diagrams.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to generate diagrams"
    exit 1
fi
echo ""

# Step 3: Generate Result Tables
echo "================================================================================"
echo "STEP 3/3: Generating Result Tables (9 tables × 3 formats)"
echo "================================================================================"
python3 generate_all_tables.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to generate tables"
    exit 1
fi
echo ""

# Summary
echo "================================================================================"
echo "✅ GENERATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📊 Figures:  $(ls figures/*.png 2>/dev/null | wc -l) PNG + $(ls figures/*.pdf 2>/dev/null | wc -l) PDF"
echo "  📐 Diagrams: $(ls diagrams/*.png 2>/dev/null | wc -l) PNG + $(ls diagrams/*.pdf 2>/dev/null | wc -l) PDF"
echo "  📋 Tables:   $(ls tables/*.csv 2>/dev/null | wc -l) CSV + $(ls tables/*.md 2>/dev/null | wc -l) MD + $(ls tables/*.tex 2>/dev/null | wc -l) TEX"
echo ""
echo "Output directories:"
echo "  - figures/  (result visualizations)"
echo "  - diagrams/ (architecture diagrams)"
echo "  - tables/   (result tables)"
echo ""
echo "================================================================================"
echo "NEXT STEPS:"
echo "================================================================================"
echo ""
echo "1. Review generated files in figures/, diagrams/, tables/"
echo "2. Insert into your report (see INTEGRATION_NOTES.md)"
echo "3. When you have REAL experimental results:"
echo "   - Update JSON files in raw_results/"
echo "   - Re-run: bash generate_everything.sh"
echo "   - All visualizations update automatically!"
echo ""
echo "================================================================================"
echo "You're ready for A+ submission! 🎉"
echo "================================================================================"
