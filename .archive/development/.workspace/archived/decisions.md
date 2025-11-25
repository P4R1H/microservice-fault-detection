### D009: CloudWatch Integration Approach
**Date**: 2025-11-14
**Decision**: Implement CloudWatch-compatible ingestion layer with synthetic data
**Rationale**:
- **User requirement**: CloudWatch integration REQUIRED but with synthetic data
- **Scope**: Show adapters + code structure demonstrating production-readiness
- **Data source**: Self-generated synthetic telemetry (not real AWS)
- **Presentation**: Frame as "CloudWatch-compatible ingestion layer" in documentation

**Implementation Strategy**:
1. Create adapter interfaces following AWS CloudWatch API patterns
2. Implement synthetic data generators for:
   - CloudWatch Metrics API (PutMetricData)
   - X-Ray Traces API (PutTraceSegments)
   - CloudWatch Logs with EMF (PutLogEvents)
3. Build ServiceLens visualization layer
4. Document as production-ready integration pattern

**Components to Build**:
- `src/utils/cloudwatch_adapter.py` - API interfaces
- `src/utils/synthetic_data_generator.py` - Telemetry generation
- `src/utils/servicelens_viz.py` - Service graph visualization
- Documentation showing CloudWatch compatibility

**Impact**: Demonstrates production-readiness and cloud-native design without AWS costs

### D010: Clean Slate Implementation Strategy
**Date**: 2025-11-14
**Decision**: Build from scratch, frame as "Phase 2 improvement" over midsem
**Rationale**:
- **Reality**: No Phase 1 code exists (midsem results were illustrative)
- **Opportunity**: Clean slate allows optimal architecture without legacy constraints
- **Presentation**: Show as natural evolution:
  - Phase 1 (midsem): LSTM-AE, RF, IF on metrics-only
  - Phase 2 (current): Chronos/TCN, GCN, PCMCI on multimodal data

**Narrative Arc for Report**:
```
Phase 1 Limitations Identified → Phase 2 Improvements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LSTM-AE (25.4s bottleneck)  →  Chronos-Bolt-Tiny (zero-shot, 100MB)
Random Forest (overfitting)  →  CatBoost (regularization) + TCN
Metrics-only                 →  Multimodal (metrics + logs + traces)
No causal inference         →  PCMCI causal discovery
No graph learning           →  GCN on service dependency graphs
Limited baselines (3)       →  Comprehensive (7+)
No ablations               →  Systematic (10-12 configs)
```

**Impact**: Professional project narrative showing research maturity and iterative improvement

---

## Implementation Constraints

### Known Constraints (Updated)