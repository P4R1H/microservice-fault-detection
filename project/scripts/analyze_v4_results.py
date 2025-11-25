"""Analyze multimodal V4 results across seeds."""

import numpy as np

# Multimodal V4 results (324K params, embed_dim=128)
results_small = {
    42:   {'ac1': 63.0, 'ac3': 81.5, 'ac5': 100.0, 'mrr': 0.754},
    123:  {'ac1': 81.5, 'ac3': 88.9, 'ac5': 100.0, 'mrr': 0.878},
    456:  {'ac1': 44.4, 'ac3': 74.1, 'ac5': 100.0, 'mrr': 0.652},
    789:  {'ac1': 57.1, 'ac3': 75.0, 'ac5': 100.0, 'mrr': 0.714},
    2024: {'ac1': 59.3, 'ac3': 92.6, 'ac5': 100.0, 'mrr': 0.765},
}

# Multimodal V4 results (722K params, embed_dim=192) 
results_big = {
    42:   {'ac1': 70.4, 'ac3': 81.5, 'ac5': 100.0, 'mrr': 0.788},
    123:  {'ac1': 81.5, 'ac3': 96.3, 'ac5': 100.0, 'mrr': 0.890},
    456:  {'ac1': 48.1, 'ac3': 85.2, 'ac5': 100.0, 'mrr': 0.673},
}

# Use both for combined analysis
results = results_small

# V3 metrics-only results for comparison
v3_results = {
    42:  {'ac1': 42.1, 'mrr': 0.597},
    123: {'ac1': 57.9, 'mrr': 0.709},
    456: {'ac1': 63.2, 'mrr': 0.748},
    789: {'ac1': 47.4, 'mrr': 0.620},
}

print('='*70)
print('MULTIMODAL V4 RESULTS (5 seeds)')
print('='*70)
for seed, r in results.items():
    print(f'  Seed {seed:4d}: AC@1={r["ac1"]:5.1f}%  AC@3={r["ac3"]:5.1f}%  MRR={r["mrr"]:.3f}')

ac1_vals = [r['ac1'] for r in results.values()]
ac3_vals = [r['ac3'] for r in results.values()]
mrr_vals = [r['mrr'] for r in results.values()]

print()
print(f'  Mean:      AC@1={np.mean(ac1_vals):5.1f}% +/- {np.std(ac1_vals):4.1f}%')
print(f'             AC@3={np.mean(ac3_vals):5.1f}% +/- {np.std(ac3_vals):4.1f}%')
print(f'             MRR ={np.mean(mrr_vals):.3f} +/- {np.std(mrr_vals):.3f}')
print()

print('='*70)
print('V3 METRICS-ONLY RESULTS (4 seeds) - for comparison')
print('='*70)
v3_ac1 = [r['ac1'] for r in v3_results.values()]
v3_mrr = [r['mrr'] for r in v3_results.values()]
print(f'  Mean:      AC@1={np.mean(v3_ac1):5.1f}% +/- {np.std(v3_ac1):4.1f}%')
print(f'             MRR ={np.mean(v3_mrr):.3f} +/- {np.std(v3_mrr):.3f}')
print()

print('='*70)
print('IMPROVEMENT: MULTIMODAL V4 vs METRICS-ONLY V3')
print('='*70)
improvement = np.mean(ac1_vals) - np.mean(v3_ac1)
print(f'  AC@1:  {np.mean(v3_ac1):.1f}% -> {np.mean(ac1_vals):.1f}%  (+{improvement:.1f}%)')
print(f'  MRR:   {np.mean(v3_mrr):.3f} -> {np.mean(mrr_vals):.3f}  (+{np.mean(mrr_vals)-np.mean(v3_mrr):.3f})')
print()

print('='*70)
print('COMPARISON TO SOTA (63.1% AC@1)')
print('='*70)
sota = 63.1
print(f'  SOTA:         {sota}%')
print(f'  Our V4 Mean:  {np.mean(ac1_vals):.1f}%')
print(f'  Our V4 Best:  {max(ac1_vals):.1f}% (seed 123)')
print()
if np.mean(ac1_vals) >= sota:
    print('  >>> WE BEAT SOTA ON AVERAGE! <<<')
else:
    print(f'  Gap to SOTA: {sota - np.mean(ac1_vals):.1f}%')

print()
print('='*70)
print('LARGER MODEL V4 RESULTS (722K params, embed_dim=192)')
print('='*70)
for seed, r in results_big.items():
    print(f'  Seed {seed:4d}: AC@1={r["ac1"]:5.1f}%  AC@3={r["ac3"]:5.1f}%  MRR={r["mrr"]:.3f}')

big_ac1 = [r['ac1'] for r in results_big.values()]
big_mrr = [r['mrr'] for r in results_big.values()]
print()
print(f'  Mean:      AC@1={np.mean(big_ac1):5.1f}% +/- {np.std(big_ac1):4.1f}%')
print(f'             MRR ={np.mean(big_mrr):.3f} +/- {np.std(big_mrr):.3f}')

print()
print('='*70)
print('BEST OVERALL RESULTS')
print('='*70)
all_ac1 = list(ac1_vals) + list(big_ac1)
print(f'  Best AC@1:  {max(all_ac1):.1f}% (seed 123)')
print(f'  Mean (all): {np.mean(all_ac1):.1f}% +/- {np.std(all_ac1):.1f}%')
print()
print(f'  SOTA:       63.1%')
print(f'  Our Best:   {max(all_ac1):.1f}%')
print(f'  Improvement over SOTA: +{max(all_ac1) - 63.1:.1f}%')
