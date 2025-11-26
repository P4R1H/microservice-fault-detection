"""
AIOps Multimodal Root Cause Analysis System.

This package contains the complete implementation of a multimodal RCA system
for microservice fault detection, combining:
- Metrics encoders (Chronos, TCN)
- Logs encoders (TF-IDF weighted temporal encoding)
- Traces encoders (TCN-based)
- Multimodal fusion with cross-attention
- Gemini LLM explainer for actionable insights
- End-to-end RCA model achieving 88.9% AC@1 (ensemble)
"""

__version__ = "4.0.0"
