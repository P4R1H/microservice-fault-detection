"""Analyze results across multiple seeds."""
import json
import statistics as st
from pathlib import Path

def main():
    seeds = [42, 123, 456, 789]
    results = []
    
    print("\n" + "="*60)
    print("MULTI-SEED RESULTS (V3 Model)")
    print("="*60)
    print(f"Seeds: {seeds}\n")
    
    for s in seeds:
        path = Path(f'outputs/test_results_v3_seed{s}.json')
        if path.exists():
            r = json.load(open(path))
            results.append((s, r))
            print(f"Seed {s}: AC@1={r['AC@1']*100:.1f}%, AC@3={r['AC@3']*100:.1f}%, "
                  f"AC@5={r['AC@5']*100:.1f}%, MRR={r['MRR']:.3f}")
        else:
            print(f"Seed {s}: FILE NOT FOUND")
    
    if len(results) >= 2:
        ac1 = [r['AC@1'] for _, r in results]
        ac3 = [r['AC@3'] for _, r in results]
        ac5 = [r['AC@5'] for _, r in results]
        mrr = [r['MRR'] for _, r in results]
        
        print("\n" + "="*60)
        print("AVERAGE ± STD")
        print("="*60)
        print(f"AC@1: {st.mean(ac1)*100:.1f}% ± {st.stdev(ac1)*100:.1f}%")
        print(f"AC@3: {st.mean(ac3)*100:.1f}% ± {st.stdev(ac3)*100:.1f}%")
        print(f"AC@5: {st.mean(ac5)*100:.1f}% ± {st.stdev(ac5)*100:.1f}%")
        print(f"MRR:  {st.mean(mrr):.3f} ± {st.stdev(mrr):.3f}")
        
        print("\n" + "="*60)
        print("COMPARISON WITH SOTA (RUN, AAAI 2024)")
        print("="*60)
        sota = {'AC@1': 0.631, 'AC@3': 0.784, 'AC@5': 0.867, 'MRR': 0.734}
        
        print(f"{'Metric':<10} {'Ours':<20} {'RUN (SOTA)':<15} {'Diff':<10}")
        print("-" * 55)
        print(f"{'AC@1':<10} {st.mean(ac1)*100:.1f}% ± {st.stdev(ac1)*100:.1f}%{'':<5} {sota['AC@1']*100:.1f}%{'':<8} {(st.mean(ac1)-sota['AC@1'])*100:+.1f}%")
        print(f"{'AC@3':<10} {st.mean(ac3)*100:.1f}% ± {st.stdev(ac3)*100:.1f}%{'':<5} {sota['AC@3']*100:.1f}%{'':<8} {(st.mean(ac3)-sota['AC@3'])*100:+.1f}%")
        print(f"{'AC@5':<10} {st.mean(ac5)*100:.1f}% ± {st.stdev(ac5)*100:.1f}%{'':<5} {sota['AC@5']*100:.1f}%{'':<8} {(st.mean(ac5)-sota['AC@5'])*100:+.1f}%")
        print(f"{'MRR':<10} {st.mean(mrr):.3f} ± {st.stdev(mrr):.3f}{'':<6} {sota['MRR']:.3f}{'':<9} {st.mean(mrr)-sota['MRR']:+.3f}")
        
        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)
        if st.mean(ac1) > sota['AC@1']:
            print(f"✅ AC@1: We BEAT SOTA by {(st.mean(ac1)-sota['AC@1'])*100:.1f}%")
        else:
            print(f"❌ AC@1: Behind SOTA by {(sota['AC@1']-st.mean(ac1))*100:.1f}%")
            
        if st.mean(mrr) > sota['MRR']:
            print(f"✅ MRR: We BEAT SOTA by {st.mean(mrr)-sota['MRR']:.3f}")
        else:
            print(f"❌ MRR: Behind SOTA by {sota['MRR']-st.mean(mrr):.3f}")

if __name__ == "__main__":
    main()
