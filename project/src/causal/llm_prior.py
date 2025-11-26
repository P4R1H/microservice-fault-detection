"""
LLM Causal Prior Generator for Root Cause Analysis.

Uses LLM domain knowledge to generate causal relationship priors
between microservices. This complements the statistical PCMCI
approach with semantic understanding of service dependencies.

The key insight: LLMs have learned common patterns like:
- "Database failures cascade to all dependent services"
- "Frontend services rarely cause backend failures"
- "Payment services are critical failure points"
"""

import os
import json
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMCausalPrior:
    """
    Generates causal prior matrices using LLM domain knowledge.
    
    The LLM is asked to reason about which services are likely to
    cause failures in other services based on:
    1. Service names (semantic meaning)
    2. Common microservice patterns
    3. Typical failure propagation paths
    
    Output is cached to avoid redundant API calls.
    """
    
    # Class-level cache
    _cache: Dict[str, np.ndarray] = {}
    _cache_loaded: bool = False
    _cache_path: Optional[Path] = None
    
    CAUSAL_PROMPT = """You are an expert Site Reliability Engineer analyzing a microservice system.

## Task
Given the following list of microservices, estimate the CAUSAL INFLUENCE between each pair.
A high causal influence from Service A to Service B means: "If Service A fails, it is likely to CAUSE problems in Service B".

## Services
{services}

## System Type
{system_type}

## Instructions
1. Consider common microservice patterns:
   - Databases/caches are upstream dependencies (failures cascade downstream)
   - API gateways/frontends are entry points (rarely cause backend failures)
   - Payment/checkout services are critical paths
   - Message queues can cause delayed cascading failures

2. For each pair (A, B), estimate P(B fails | A fails) on a scale of 0.0 to 1.0:
   - 0.0: A's failure has NO effect on B
   - 0.3: A's failure MIGHT affect B (weak dependency)
   - 0.6: A's failure LIKELY affects B (moderate dependency)
   - 0.9: A's failure ALMOST CERTAINLY affects B (strong dependency)

3. Output ONLY a JSON object with the causal matrix.
   Format: {{"causal_matrix": [[row1], [row2], ...], "reasoning": "brief explanation"}}
   
   Matrix[i][j] = causal influence from service i to service j
   Diagonal should be 0.0 (service doesn't cause itself)

## Example for 3 services [frontend, api, database]:
{{"causal_matrix": [[0.0, 0.2, 0.1], [0.1, 0.0, 0.1], [0.7, 0.8, 0.0]], "reasoning": "Database failures cascade strongly to API and frontend"}}

## Your Response (JSON only):"""

    def __init__(
        self,
        cache_path: str = "outputs/llm_causal_cache.pkl",
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,  # Low for consistency
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM Causal Prior generator.
        
        Args:
            cache_path: Path to cache file
            model_name: Gemini model to use
            temperature: Generation temperature (lower = more deterministic)
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Gemini if available
        self.model = None
        if GEMINI_AVAILABLE:
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
        
        # Load cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cached causal priors from disk."""
        if LLMCausalPrior._cache_loaded and LLMCausalPrior._cache_path == self.cache_path:
            return
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    LLMCausalPrior._cache = pickle.load(f)
                print(f"Loaded {len(LLMCausalPrior._cache)} cached LLM causal priors")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                LLMCausalPrior._cache = {}
        
        LLMCausalPrior._cache_loaded = True
        LLMCausalPrior._cache_path = self.cache_path
    
    def _save_cache(self):
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(LLMCausalPrior._cache, f)
    
    def _get_cache_key(self, services: List[str], system_type: str) -> str:
        """Generate cache key from services and system type."""
        key_str = f"{sorted(services)}_{system_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _parse_llm_response(self, response: str, n_services: int) -> Optional[np.ndarray]:
        """Parse LLM response to extract causal matrix."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            matrix = np.array(data['causal_matrix'], dtype=np.float32)
            
            # Validate shape
            if matrix.shape != (n_services, n_services):
                print(f"Warning: Matrix shape {matrix.shape} != expected ({n_services}, {n_services})")
                return None
            
            # Normalize to [0, 1]
            matrix = np.clip(matrix, 0.0, 1.0)
            
            # Zero diagonal (service doesn't cause itself)
            np.fill_diagonal(matrix, 0.0)
            
            return matrix
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return None
    
    def _generate_fallback_prior(self, services: List[str]) -> np.ndarray:
        """
        Generate a heuristic-based fallback prior when LLM is unavailable.
        
        Uses service name patterns to infer dependencies.
        """
        n = len(services)
        matrix = np.ones((n, n), dtype=np.float32) * 0.3  # Default weak influence
        np.fill_diagonal(matrix, 0.0)
        
        # Keyword-based heuristics
        db_keywords = ['db', 'database', 'mysql', 'postgres', 'mongo', 'redis', 'cache']
        gateway_keywords = ['frontend', 'gateway', 'nginx', 'web', 'ui']
        critical_keywords = ['payment', 'checkout', 'order', 'auth', 'user']
        
        for i, src in enumerate(services):
            src_lower = src.lower()
            
            for j, dst in enumerate(services):
                if i == j:
                    continue
                    
                dst_lower = dst.lower()
                
                # Database/cache failures cascade strongly
                if any(kw in src_lower for kw in db_keywords):
                    matrix[i, j] = 0.7
                
                # Gateway/frontend failures don't cascade much
                if any(kw in src_lower for kw in gateway_keywords):
                    matrix[i, j] = min(matrix[i, j], 0.2)
                
                # Critical services affect many others
                if any(kw in src_lower for kw in critical_keywords):
                    matrix[i, j] = max(matrix[i, j], 0.5)
                
                # Same-name prefix suggests dependency
                if src_lower.split('-')[0] == dst_lower.split('-')[0]:
                    matrix[i, j] = max(matrix[i, j], 0.4)
        
        return matrix
    
    def generate_prior(
        self,
        services: List[str],
        system_type: str = "generic microservice"
    ) -> np.ndarray:
        """
        Generate causal prior matrix for given services.
        
        Args:
            services: List of service names
            system_type: Type of system (e.g., "e-commerce", "social media")
            
        Returns:
            (n_services, n_services) causal prior matrix
        """
        n_services = len(services)
        cache_key = self._get_cache_key(services, system_type)
        
        # Check cache first
        if cache_key in LLMCausalPrior._cache:
            cached = LLMCausalPrior._cache[cache_key]
            if cached.shape == (n_services, n_services):
                return cached
        
        # Try LLM generation
        if self.model is not None:
            try:
                prompt = self.CAUSAL_PROMPT.format(
                    services=json.dumps(services),
                    system_type=system_type
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=1024
                    )
                )
                
                matrix = self._parse_llm_response(response.text, n_services)
                
                if matrix is not None:
                    # Cache and return
                    LLMCausalPrior._cache[cache_key] = matrix
                    self._save_cache()
                    return matrix
                    
            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")
        
        # Fallback to heuristic
        print("Using heuristic-based causal prior (LLM unavailable)")
        matrix = self._generate_fallback_prior(services)
        LLMCausalPrior._cache[cache_key] = matrix
        self._save_cache()
        
        return matrix
    
    def get_combined_weights(
        self,
        pcmci_weights: np.ndarray,
        services: List[str],
        system_type: str = "generic microservice",
        lambda_pcmci: float = 0.7,
        lambda_prior: float = 0.3
    ) -> np.ndarray:
        """
        Combine PCMCI statistical weights with LLM prior.
        
        Args:
            pcmci_weights: (n_services, n_services) from PCMCI
            services: List of service names
            system_type: Type of system
            lambda_pcmci: Weight for statistical causal discovery
            lambda_prior: Weight for LLM prior
            
        Returns:
            Combined causal weight matrix
        """
        assert lambda_pcmci + lambda_prior == 1.0, "Weights must sum to 1"
        
        llm_prior = self.generate_prior(services, system_type)
        
        # Ensure same shape
        if llm_prior.shape != pcmci_weights.shape:
            # Resize if needed
            n = pcmci_weights.shape[0]
            if llm_prior.shape[0] < n:
                padded = np.eye(n, dtype=np.float32) * 0.3
                padded[:llm_prior.shape[0], :llm_prior.shape[1]] = llm_prior
                llm_prior = padded
            else:
                llm_prior = llm_prior[:n, :n]
        
        # Combine
        combined = lambda_pcmci * pcmci_weights + lambda_prior * llm_prior
        
        # Normalize
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined


class CausalWeightManager:
    """
    Unified manager for both PCMCI and LLM causal weights.
    
    Handles:
    1. PCMCI weight computation/caching
    2. LLM prior generation/caching
    3. Weight combination with configurable lambdas
    """
    
    def __init__(
        self,
        pcmci_cache_path: str = "outputs/causal_cache_multimodal.pkl",
        llm_cache_path: str = "outputs/llm_causal_cache.pkl",
        lambda_pcmci: float = 0.7,
        lambda_prior: float = 0.3,
        use_llm_prior: bool = True
    ):
        """
        Initialize causal weight manager.
        
        Args:
            pcmci_cache_path: Path to PCMCI cache
            llm_cache_path: Path to LLM prior cache
            lambda_pcmci: Weight for PCMCI (statistical)
            lambda_prior: Weight for LLM prior (domain knowledge)
            use_llm_prior: Whether to use LLM prior
        """
        from .pcmci import CausalWeightComputer
        
        self.pcmci_computer = CausalWeightComputer(cache_path=pcmci_cache_path)
        self.llm_prior = LLMCausalPrior(cache_path=llm_cache_path) if use_llm_prior else None
        
        self.lambda_pcmci = lambda_pcmci
        self.lambda_prior = lambda_prior
        self.use_llm_prior = use_llm_prior
    
    def get_weights(
        self,
        case_id: str,
        services: List[str],
        system_type: str = "generic microservice"
    ) -> np.ndarray:
        """
        Get combined causal weights for a case.
        
        Args:
            case_id: Failure case identifier
            services: List of service names
            system_type: Type of system
            
        Returns:
            Combined causal weight matrix
        """
        n_services = len(services)
        
        # Get PCMCI weights
        pcmci_weights = self.pcmci_computer.get_weights(case_id, n_services)
        
        if not self.use_llm_prior or self.llm_prior is None:
            return pcmci_weights
        
        # Combine with LLM prior
        return self.llm_prior.get_combined_weights(
            pcmci_weights=pcmci_weights,
            services=services,
            system_type=system_type,
            lambda_pcmci=self.lambda_pcmci,
            lambda_prior=self.lambda_prior
        )
    
    def get_batch_weights(
        self,
        case_ids: List[str],
        services: List[str],
        system_type: str = "generic microservice",
        device: str = 'cpu'
    ):
        """
        Get causal weights for a batch of cases.
        
        Args:
            case_ids: List of case identifiers
            services: List of service names
            system_type: Type of system
            device: PyTorch device
            
        Returns:
            torch.Tensor of shape (batch, n_services, n_services)
        """
        import torch
        
        weights = np.stack([
            self.get_weights(cid, services, system_type) for cid in case_ids
        ])
        
        return torch.from_numpy(weights).float().to(device)


# Convenience functions
def get_system_type(system_name: str) -> str:
    """Map system name to descriptive type."""
    mappings = {
        'trainticket': 'train ticket booking e-commerce system',
        'sockshop': 'e-commerce sock shop with microservices',
        'onlineboutique': 'google cloud online boutique e-commerce',
        'gaia': 'qr code login microservice system'
    }
    return mappings.get(system_name.lower(), 'generic microservice system')


if __name__ == '__main__':
    # Test the LLM causal prior
    print("=" * 60)
    print("Testing LLM Causal Prior Generator")
    print("=" * 60)
    
    # Example services from OnlineBoutique
    services = [
        'frontend',
        'cartservice', 
        'productcatalogservice',
        'currencyservice',
        'paymentservice',
        'shippingservice',
        'emailservice',
        'checkoutservice',
        'recommendationservice',
        'adservice'
    ]
    
    prior_gen = LLMCausalPrior()
    
    print(f"\nGenerating causal prior for {len(services)} services...")
    print(f"Services: {services}")
    
    prior_matrix = prior_gen.generate_prior(
        services=services,
        system_type="Google Cloud Online Boutique e-commerce"
    )
    
    print(f"\nCausal Prior Matrix Shape: {prior_matrix.shape}")
    print("\nTop causal relationships (A -> B with strength):")
    
    # Find top relationships
    relationships = []
    for i, src in enumerate(services):
        for j, dst in enumerate(services):
            if i != j and prior_matrix[i, j] > 0.4:
                relationships.append((src, dst, prior_matrix[i, j]))
    
    relationships.sort(key=lambda x: x[2], reverse=True)
    
    for src, dst, strength in relationships[:10]:
        print(f"  {src} -> {dst}: {strength:.2f}")
    
    print("\n✅ LLM Causal Prior test complete!")
