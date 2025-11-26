"""
Gemini-powered RCA Explainer.

Generates natural language explanations for root cause analysis predictions
using Google's Gemini LLM. This provides interpretable insights for operators.

Usage:
    explainer = GeminiExplainer()
    explanation = explainer.explain(
        prediction={"root_cause": "checkout-service", "confidence": 0.85},
        context={
            "metrics": {...},
            "logs": [...],
            "traces": {...}
        }
    )
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


class GeminiExplainer:
    """
    Generates natural language explanations for RCA predictions using Gemini.
    
    Features:
    - Summarizes evidence from metrics, logs, and traces
    - Explains why the predicted root cause is likely
    - Suggests remediation actions
    - Provides confidence analysis
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.3,  # Lower for more factual responses
        max_tokens: int = 1024
    ):
        """
        Initialize the Gemini explainer.
        
        Args:
            config_path: Path to LLM config YAML (optional)
            model_name: Gemini model to use (default: gemini-2.0-flash)
            temperature: Generation temperature (0-1)
            max_tokens: Maximum response tokens
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai required. Install with: pip install google-generativeai")
        
        # Load config
        self._load_config(config_path, model_name)
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)  # type: ignore
        self.model = genai.GenerativeModel(self.model_name)  # type: ignore
        
        # System prompt for RCA explanations
        self.system_prompt = self._build_system_prompt()
    
    def _load_config(self, config_path: Optional[str], model_name: Optional[str]):
        """Load API configuration from file or environment."""
        # Try loading from .env first
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.model_name = model_name or config.get('chat_model', 'gemini-2.0-flash')
        else:
            # Try default config location
            default_config = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'
            if default_config.exists():
                with open(default_config) as f:
                    config = yaml.safe_load(f)
                    self.model_name = model_name or config.get('chat_model', 'gemini-2.0-flash')
            else:
                self.model_name = model_name or os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')
        
        # Get API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set in .env file or environment.")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for RCA explanations."""
        return """You are an expert SRE providing ROOT CAUSE ANALYSIS for microservice failures.

CRITICAL: Be BRIEF. Focus on ACTIONS, not descriptions.

Response format (STRICTLY follow this):

## Root Cause: [Service Name]
One sentence: What failed and why.

## Evidence
- [Metric/Log/Trace finding 1]
- [Metric/Log/Trace finding 2]
- [Metric/Log/Trace finding 3]

## Immediate Actions
1. [First action to take NOW]
2. [Second action]
3. [Third action]

## Prevention
- [How to prevent recurrence]

Rules:
- Max 150 words total
- No speculation without evidence
- Every sentence must be actionable or evidential
- Skip sections if no relevant data
"""
    
    def explain(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any],
        include_evidence: bool = True
    ) -> str:
        """
        Generate explanation for an RCA prediction.
        
        Args:
            prediction: Dict with 'root_cause', 'confidence', 'ranking', etc.
            context: Dict with 'metrics', 'logs', 'traces', 'services' info
            include_evidence: Whether to include detailed evidence summary
        
        Returns:
            Natural language explanation string
        """
        # Build the prompt
        prompt = self._build_prompt(prediction, context, include_evidence)
        
        # Generate response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(  # type: ignore
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _build_prompt(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any],
        include_evidence: bool
    ) -> str:
        """Build the full prompt for explanation generation."""
        
        # Extract prediction info
        root_cause = prediction.get('root_cause', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        ranking = prediction.get('ranking', [])
        
        prompt_parts = [self.system_prompt, "\n---\n"]
        
        # Add prediction summary
        prompt_parts.append(f"""
## Prediction to Explain

**Predicted Root Cause:** {root_cause}
**Confidence Score:** {confidence:.1%}
""")
        
        if ranking:
            prompt_parts.append("\n**Top 3 Candidates:**\n")
            for i, (service, score) in enumerate(ranking[:3]):
                prompt_parts.append(f"{i+1}. {service}: {score:.1%}\n")
        
        # Add context evidence
        if include_evidence and context:
            prompt_parts.append("\n---\n## Available Evidence\n")
            
            # Service list
            if 'services' in context:
                prompt_parts.append(f"\n**Services in System:** {', '.join(context['services'])}\n")
            
            # Metrics summary
            if 'metrics' in context and context['metrics']:
                prompt_parts.append("\n### Metrics Anomalies\n")
                metrics = context['metrics']
                if isinstance(metrics, dict):
                    for service, data in metrics.items():
                        if isinstance(data, dict) and data.get('anomaly_score', 0) > 0.5:
                            prompt_parts.append(
                                f"- **{service}**: Anomaly score {data['anomaly_score']:.2f}"
                            )
                            if 'details' in data:
                                prompt_parts.append(f" ({data['details']})")
                            prompt_parts.append("\n")
            
            # Logs summary  
            if 'logs' in context and context['logs']:
                prompt_parts.append("\n### Error Logs\n")
                logs = context['logs']
                if isinstance(logs, list):
                    for log in logs[:10]:  # Limit to 10 logs
                        if isinstance(log, dict):
                            service = log.get('service', 'unknown')
                            message = log.get('message', str(log))[:200]
                            level = log.get('level', 'INFO')
                            prompt_parts.append(f"- [{level}] **{service}**: {message}\n")
                        else:
                            prompt_parts.append(f"- {str(log)[:200]}\n")
                elif isinstance(logs, dict):
                    for service, service_logs in logs.items():
                        if service_logs:
                            log_sample = str(service_logs[0])[:100] if isinstance(service_logs, list) else str(service_logs)[:100]
                            prompt_parts.append(f"- **{service}**: {log_sample}...\n")
            
            # Traces summary
            if 'traces' in context and context['traces']:
                prompt_parts.append("\n### Trace Patterns\n")
                traces = context['traces']
                if isinstance(traces, dict):
                    for service, data in traces.items():
                        if isinstance(data, dict):
                            latency = data.get('avg_latency', 0)
                            error_rate = data.get('error_rate', 0)
                            if latency > 0 or error_rate > 0:
                                prompt_parts.append(
                                    f"- **{service}**: Avg latency {latency:.0f}ms, "
                                    f"Error rate {error_rate:.1%}\n"
                                )
        
        prompt_parts.append("\n---\n\nPlease provide a clear explanation of this root cause analysis result.")
        
        return ''.join(prompt_parts)
    
    def explain_batch(
        self,
        predictions: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of prediction dicts
            contexts: List of context dicts
        
        Returns:
            List of explanation strings
        """
        explanations = []
        for pred, ctx in zip(predictions, contexts):
            explanation = self.explain(pred, ctx)
            explanations.append(explanation)
        return explanations
    
    def explain_with_comparison(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate explanation with optional comparison to ground truth.
        
        Args:
            prediction: Prediction dict
            context: Context dict
            ground_truth: Actual root cause (if known)
        
        Returns:
            Dict with 'explanation' and optionally 'comparison'
        """
        result = {'explanation': self.explain(prediction, context)}
        
        if ground_truth:
            pred_rc = prediction.get('root_cause', 'Unknown')
            is_correct = pred_rc == ground_truth
            
            if is_correct:
                result['comparison'] = f"✅ Prediction matches ground truth: {ground_truth}"
            else:
                # Generate analysis of the discrepancy
                comparison_prompt = f"""Model predicted '{pred_rc}', actual was '{ground_truth}'.

In 2-3 bullet points:
- Why might '{pred_rc}' have shown symptoms?
- How did '{ground_truth}' likely cause the cascade?
- What signal was missed?

Max 50 words."""
                try:
                    response = self.model.generate_content(
                        comparison_prompt,
                        generation_config=genai.GenerationConfig(  # type: ignore
                            temperature=0.3,
                            max_output_tokens=512
                        )
                    )
                    result['comparison'] = f"❌ Predicted: {pred_rc}, Actual: {ground_truth}\n\n{response.text}"
                except Exception as e:
                    result['comparison'] = f"❌ Predicted: {pred_rc}, Actual: {ground_truth}"
        
        return result


def create_explainer(**kwargs) -> GeminiExplainer:
    """Factory function to create explainer."""
    return GeminiExplainer(**kwargs)


if __name__ == '__main__':
    # Test the explainer
    print("Testing GeminiExplainer...")
    
    try:
        explainer = GeminiExplainer()
        
        # Sample prediction and context
        prediction = {
            'root_cause': 'checkout-service',
            'confidence': 0.85,
            'ranking': [
                ('checkout-service', 0.85),
                ('payment-service', 0.10),
                ('cart-service', 0.05)
            ]
        }
        
        context = {
            'services': ['frontend', 'cart-service', 'checkout-service', 'payment-service', 'shipping-service'],
            'metrics': {
                'checkout-service': {'anomaly_score': 0.92, 'details': 'CPU spike to 95%, latency 10x baseline'},
                'payment-service': {'anomaly_score': 0.45, 'details': 'Slight latency increase'},
            },
            'logs': [
                {'service': 'checkout-service', 'level': 'ERROR', 'message': 'Connection timeout to database after 30s'},
                {'service': 'checkout-service', 'level': 'ERROR', 'message': 'Failed to process order: DB connection pool exhausted'},
                {'service': 'payment-service', 'level': 'WARN', 'message': 'Upstream service checkout-service responding slowly'},
            ],
            'traces': {
                'checkout-service': {'avg_latency': 5200, 'error_rate': 0.45},
                'payment-service': {'avg_latency': 1500, 'error_rate': 0.12},
                'cart-service': {'avg_latency': 150, 'error_rate': 0.01},
            }
        }
        
        print("\n" + "="*60)
        print("Generating explanation...")
        print("="*60 + "\n")
        
        explanation = explainer.explain(prediction, context)
        print(explanation)
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
