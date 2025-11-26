"""
Logs encoder with Drain3 parsing and embeddings.

V4.1 Update: Proper TF-IDF based logs encoding (replaces placeholder)
V5.0 Update: Gemini LLM embeddings for semantic understanding

Pipeline options:
1. TF-IDF: Template counts → TF-IDF weighted → Temporal encoding → 64d
2. Gemini: Template counts → LLM semantic embeddings → Projection → 64d
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import pickle
import hashlib
from pathlib import Path

# Optional imports for Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TFIDFLogsEncoder(nn.Module):
    """
    TF-IDF weighted logs encoder for log template time series.
    
    This encoder properly processes log template counts by:
    1. Learning TF-IDF-like weights for each template
    2. Applying temporal convolution to capture patterns
    3. Projecting to embedding dimension
    
    Input: (batch, n_services, seq_len, n_log_features) - template counts per timestep
    Output: (batch * n_services, embed_dim)
    """
    
    def __init__(self,
                 n_log_features: int = 32,
                 hidden_dim: int = 48,
                 embed_dim: int = 64,
                 num_layers: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_log_features = n_log_features
        self.embed_dim = embed_dim
        
        # Learnable IDF-like weights for each template
        # Templates that appear in few cases should have higher weight
        self.template_weights = nn.Parameter(torch.ones(n_log_features))
        
        # Input projection with template weighting
        self.input_proj = nn.Sequential(
            nn.Linear(n_log_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal convolution layers (TCN-style)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.extend([
                # Depthwise separable convolution
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                         padding=padding, dilation=dilation, groups=min(hidden_dim, 8)),
                nn.Conv1d(hidden_dim, hidden_dim, 1),  # Pointwise
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.temporal = nn.Sequential(*layers)
        
        # Pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Error pattern detector (high-weight templates often indicate errors)
        self.error_detector = nn.Sequential(
            nn.Linear(n_log_features, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, n_log_features) - template counts
            
        Returns:
            (batch * n_services, embed_dim)
        """
        # Apply learnable template weights (IDF-like)
        # Softmax to normalize weights, then scale
        weights = torch.softmax(self.template_weights, dim=0)
        x_weighted = x * weights.unsqueeze(0).unsqueeze(0)  # (B*S, T, F)
        
        # Extract error pattern features from raw counts
        # Use mean counts across time to detect persistent patterns
        mean_counts = x.mean(dim=1)  # (B*S, F)
        error_features = self.error_detector(mean_counts)  # (B*S, embed_dim)
        
        # Temporal encoding
        h = self.input_proj(x_weighted)  # (B*S, T, hidden)
        h = h.permute(0, 2, 1)  # (B*S, hidden, T)
        h = self.temporal(h)  # (B*S, hidden, T)
        h = self.pool(h).squeeze(-1)  # (B*S, hidden)
        
        # Output projection
        temporal_features = self.output_proj(h)  # (B*S, embed_dim)
        
        # Combine temporal and error pattern features
        output = temporal_features + 0.3 * error_features
        
        return output


class FallbackLogsEncoder(nn.Module):
    """
    Simple fallback logs encoder using learned embeddings.
    
    DEPRECATED: Use TFIDFLogsEncoder instead.
    Kept for backward compatibility with old checkpoints.
    """

    def __init__(self, embedding_dim: int = 64, n_log_features: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_log_features = n_log_features
        
        # Simple MLP on mean log counts
        self.encoder = nn.Sequential(
            nn.Linear(n_log_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, n_log_features) or None
            
        Returns:
            (batch * n_services, embedding_dim)
        """
        if x is None or x.numel() == 0:
            # Return zeros if no input
            return torch.zeros(1, self.embedding_dim, device=next(self.parameters()).device)
        
        # Take mean across time
        x_mean = x.mean(dim=1)  # (B*S, n_log_features)
        return self.encoder(x_mean)


class GeminiLogsEncoder(nn.Module):
    """
    Gemini LLM-based logs encoder with semantic embeddings.
    
    Pipeline:
    1. Log template names (from column headers) → Gemini text-embedding-004
    2. Template embeddings (768d) → Learned projection → 64d
    3. Template counts × embeddings → Weighted sum
    4. Temporal convolution → Output embedding
    
    The key insight: Log template NAMES contain semantic meaning.
    E.g., "error_database_connection" vs "info_user_login"
    
    Caching: Embeddings are cached to disk to avoid redundant API calls.
    """
    
    # Class-level embedding cache (shared across instances)
    _embedding_cache: Dict[str, np.ndarray] = {}
    _cache_loaded: bool = False
    _cache_path: Optional[Path] = None
    
    def __init__(
        self,
        n_log_features: int = 32,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        gemini_embed_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.3,
        cache_dir: str = "outputs/llm_cache",
        api_key: Optional[str] = None
    ):
        super().__init__()
        
        self.n_log_features = n_log_features
        self.embed_dim = embed_dim
        self.gemini_embed_dim = gemini_embed_dim
        
        # Initialize Gemini API
        self._init_gemini(api_key)
        
        # Setup cache
        self._setup_cache(cache_dir)
        
        # Projection: Gemini 768d → hidden_dim
        self.gemini_proj = nn.Sequential(
            nn.Linear(gemini_embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learnable template embeddings (fallback if API fails)
        # Also used for weighted combination
        self.template_embed = nn.Embedding(n_log_features, hidden_dim)
        
        # Temporal encoding (same as TF-IDF encoder)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (3 - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 3,
                         padding=padding, dilation=dilation, groups=min(hidden_dim, 8)),
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.temporal = nn.Sequential(*layers)
        
        # Output projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Cached projected embeddings (populated in first forward pass)
        self._projected_embeddings: Optional[torch.Tensor] = None
        
        # Default template names (will be set from data config)
        self._template_names: List[str] = [f"template_{i}" for i in range(n_log_features)]
    
    def _init_gemini(self, api_key: Optional[str] = None):
        """Initialize Gemini API."""
        self._gemini_available = False
        
        if not GEMINI_AVAILABLE:
            print("Warning: google-generativeai not installed. Using fallback embeddings.")
            return
        
        # Get API key from parameter, env, or .env file
        key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not key:
            # Try loading from .env file
            env_file = Path("project/.env")
            if not env_file.exists():
                env_file = Path(".env")
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            key = line.strip().split("=", 1)[1].strip('"\'')
                            break
        
        if key:
            try:
                genai.configure(api_key=key)
                self._gemini_available = True
            except Exception as e:
                print(f"Warning: Failed to configure Gemini API: {e}")
        else:
            print("Warning: No Gemini API key found. Using fallback embeddings.")
    
    def _setup_cache(self, cache_dir: str):
        """Setup embedding cache."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        GeminiLogsEncoder._cache_path = cache_path / "template_embeddings.pkl"
        
        # Load existing cache
        if not GeminiLogsEncoder._cache_loaded and GeminiLogsEncoder._cache_path.exists():
            try:
                with open(GeminiLogsEncoder._cache_path, 'rb') as f:
                    GeminiLogsEncoder._embedding_cache = pickle.load(f)
                GeminiLogsEncoder._cache_loaded = True
                print(f"Loaded {len(GeminiLogsEncoder._embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Failed to load embedding cache: {e}")
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if GeminiLogsEncoder._cache_path:
            try:
                with open(GeminiLogsEncoder._cache_path, 'wb') as f:
                    pickle.dump(GeminiLogsEncoder._embedding_cache, f)
            except Exception as e:
                print(f"Warning: Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for template text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def set_template_names(self, names: List[str]):
        """
        Set template names for semantic embedding.
        
        Should be called with log column names from the dataset.
        E.g., ["frontend_error_conn", "db_slow_query", "auth_login_success"]
        """
        self._template_names = names[:self.n_log_features]
        # Pad with generic names if needed
        while len(self._template_names) < self.n_log_features:
            self._template_names.append(f"template_{len(self._template_names)}")
        # Reset projected embeddings to force recomputation
        self._projected_embeddings = None
    
    def _get_gemini_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from Gemini API with caching.
        
        Args:
            texts: List of template names/descriptions
            
        Returns:
            (len(texts), 768) numpy array of embeddings
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in GeminiLogsEncoder._embedding_cache:
                embeddings.append((i, GeminiLogsEncoder._embedding_cache[cache_key]))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Get embeddings for uncached texts
        if texts_to_embed and self._gemini_available:
            try:
                # Format templates into more descriptive text for better embeddings
                formatted_texts = []
                for t in texts_to_embed:
                    # Convert template name to description
                    # e.g., "frontend_error_conn" → "frontend error connection"
                    desc = t.replace('_', ' ').replace('-', ' ')
                    formatted_texts.append(f"Log message template: {desc}")
                
                # Batch embed
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=formatted_texts,
                    task_type="SEMANTIC_SIMILARITY"
                )
                
                # Cache and collect results
                for idx, (orig_idx, text) in enumerate(zip(text_indices, texts_to_embed)):
                    emb = np.array(result['embedding'][idx] if isinstance(result['embedding'][0], list) 
                                   else result['embedding'], dtype=np.float32)
                    if emb.ndim == 1 and len(result['embedding']) > 1:
                        emb = np.array(result['embedding'][idx], dtype=np.float32)
                    cache_key = self._get_cache_key(text)
                    GeminiLogsEncoder._embedding_cache[cache_key] = emb
                    embeddings.append((orig_idx, emb))
                
                # Save updated cache
                self._save_cache()
                
            except Exception as e:
                print(f"Warning: Gemini embedding failed: {e}. Using fallback.")
                # Use random but consistent embeddings as fallback
                for orig_idx, text in zip(text_indices, texts_to_embed):
                    np.random.seed(hash(text) % 2**32)
                    emb = np.random.randn(self.gemini_embed_dim).astype(np.float32)
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append((orig_idx, emb))
        
        elif texts_to_embed:
            # No Gemini, use consistent random embeddings
            for orig_idx, text in zip(text_indices, texts_to_embed):
                np.random.seed(hash(text) % 2**32)
                emb = np.random.randn(self.gemini_embed_dim).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                embeddings.append((orig_idx, emb))
        
        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([e[1] for e in embeddings])
    
    def _ensure_projected_embeddings(self, device: torch.device):
        """Ensure template embeddings are computed and projected."""
        if self._projected_embeddings is not None:
            # Move to correct device if needed
            if self._projected_embeddings.device != device:
                self._projected_embeddings = self._projected_embeddings.to(device)
            return
        
        # Get Gemini embeddings for template names
        gemini_emb = self._get_gemini_embeddings(self._template_names)
        gemini_emb = torch.from_numpy(gemini_emb).to(device)  # (n_templates, 768)
        
        # Project to hidden dim
        with torch.no_grad():
            self._projected_embeddings = self.gemini_proj(gemini_emb)  # (n_templates, hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, n_log_features) - template counts
            
        Returns:
            (batch * n_services, embed_dim)
        """
        batch_size, seq_len, n_features = x.shape
        device = x.device
        
        # Ensure we have projected template embeddings
        self._ensure_projected_embeddings(device)
        
        # Weight template embeddings by counts
        # x: (B*S, T, F), projected: (F, hidden)
        # Result: (B*S, T, hidden)
        template_emb = self._projected_embeddings[:n_features]  # Handle feature count mismatch
        
        # Pad if needed
        if n_features < self.n_log_features:
            # Use learned embeddings for padding
            padding = self.template_embed.weight[n_features:self.n_log_features]
            template_emb = torch.cat([template_emb, padding], dim=0)
        
        # Weighted combination: counts × embeddings
        # Normalize counts per timestep for stability
        x_norm = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        h = torch.einsum('btf,fh->bth', x_norm, template_emb)  # (B*S, T, hidden)
        
        # Also add count magnitude as a feature
        count_magnitude = x.sum(dim=-1, keepdim=True)  # (B*S, T, 1)
        count_scale = torch.tanh(count_magnitude / 10.0)  # Normalize
        h = h * (1 + 0.1 * count_scale)
        
        # Temporal encoding
        h = h.permute(0, 2, 1)  # (B*S, hidden, T)
        h = self.temporal(h)
        h = self.pool(h).squeeze(-1)  # (B*S, hidden)
        
        # Output projection
        output = self.output_proj(h)  # (B*S, embed_dim)
        
        return output


class LogsEncoder(nn.Module):
    """
    Main logs encoder interface.
    
    Supports multiple backends:
    - 'tfidf': TFIDFLogsEncoder (default for V4.1)
    - 'gemini': GeminiLogsEncoder (V5.0, requires API)
    - 'fallback': Simple learned embedding (deprecated)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_log_features: int = 32,
        encoder_type: str = 'tfidf',
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        
        if encoder_type == 'tfidf':
            self.encoder = TFIDFLogsEncoder(
                n_log_features=n_log_features,
                hidden_dim=hidden_dim,
                embed_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif encoder_type == 'fallback':
            self.encoder = FallbackLogsEncoder(
                embedding_dim=embedding_dim,
                n_log_features=n_log_features
            )
        elif encoder_type == 'gemini':
            self.encoder = GeminiLogsEncoder(
                n_log_features=n_log_features,
                hidden_dim=hidden_dim,
                embed_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                cache_dir=kwargs.get('cache_dir', 'outputs/llm_cache'),
                api_key=kwargs.get('api_key', None)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode log data.

        Args:
            x: (batch * n_services, seq_len, n_log_features) template counts

        Returns:
            (batch * n_services, embedding_dim)
        """
        return self.encoder(x)


# Factory function
def create_logs_encoder(
    embedding_dim: int = 64,
    n_log_features: int = 32,
    encoder_type: str = 'tfidf',
    **kwargs
) -> nn.Module:
    """
    Create a logs encoder.

    Args:
        embedding_dim: Output embedding dimension
        n_log_features: Number of log template features
        encoder_type: 'tfidf', 'gemini', or 'fallback'
        **kwargs: Additional arguments

    Returns:
        LogsEncoder instance
    """
    return LogsEncoder(
        embedding_dim=embedding_dim,
        n_log_features=n_log_features,
        encoder_type=encoder_type,
        **kwargs
    )
