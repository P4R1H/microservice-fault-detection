"""
AIOps Multimodal Root Cause Analysis System.

This package contains the complete implementation of a multimodal RCA system
for microservice fault detection, combining:
- Metrics encoders (Chronos, TCN)
- Logs encoders (Drain3, embeddings)
- Traces encoders (GCN, GAT)
- Multimodal fusion with cross-attention
- PCMCI causal discovery
- End-to-end RCA model
"""

__version__ = "0.1.0"
