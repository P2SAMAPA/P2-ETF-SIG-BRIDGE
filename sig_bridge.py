"""
sig_bridge.py  —  Sig-Bridge Engine
====================================

Implements:
- Path signature computation
- Neural SDE with signature conditioning
- Schrödinger Bridge via conditional flow matching
- Path interpolation between market states
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings("ignore")


class SignatureComputer:
    """
    Path signature computation for conditioning.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.depth = config.get("depth", 3)
        self.n_landmarks = config.get("n_landmarks", 100)
        self.include_time = config.get("include_time", True)
        self.normalize = config.get("normalize", True)
        
    def compute_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute truncated signature of a path.
        
        Args:
            path: (n_steps, n_dim) array representing the path
        
        Returns:
            signature: truncated signature vector
        """
        n_steps, n_dim = path.shape
        
        if n_steps < 2:
            return np.zeros(self._get_signature_dimension(n_dim))
        
        # Normalize path
        if self.normalize:
            path = (path - np.mean(path, axis=0)) / (np.std(path, axis=0) + 1e-6)
        
        # Add time dimension if requested
        if self.include_time:
            t = np.linspace(0, 1, n_steps).reshape(-1, 1)
            path_with_time = np.hstack([t, path])
            n_dim = path_with_time.shape[1]
        else:
            path_with_time = path
        
        # Compute increments
        increments = np.diff(path_with_time, axis=0)
        
        # Compute signature terms recursively
        sig_terms = [1.0]  # Level 0
        
        # Level 1: ∫ dX
        level1 = np.sum(increments, axis=0)
        sig_terms.extend(level1.tolist())
        
        if self.depth >= 2 and len(increments) > 1:
            # Level 2: ∫∫ dX⊗dX
            level2 = np.zeros((n_dim, n_dim))
            cumsum = np.zeros(n_dim)
            for i in range(len(increments)):
                cumsum += increments[i]
                level2 += np.outer(cumsum, increments[i])
            level2 = level2 / len(increments)
            sig_terms.extend(level2.flatten().tolist())
        
        if self.depth >= 3 and len(increments) > 2:
            # Level 3: ∫∫∫ dX⊗dX⊗dX
            level3 = np.zeros((n_dim, n_dim, n_dim))
            cumsum1 = np.zeros(n_dim)
            cumsum2 = np.zeros((n_dim, n_dim))
            for i in range(len(increments)):
                cumsum1 += increments[i]
                cumsum2 += np.outer(cumsum1, increments[i])
                for j in range(n_dim):
                    for k in range(n_dim):
                        level3[j, k, :] += cumsum2[j, k] * increments[i]
            level3 = level3 / (len(increments) ** 1.5)
            sig_terms.extend(level3.flatten().tolist())
        
        # Truncate to a fixed size
        max_size = self._get_signature_dimension(n_dim)
        sig_array = np.array(sig_terms)
        if len(sig_array) > max_size:
            sig_array = sig_array[:max_size]
        elif len(sig_array) < max_size:
            sig_array = np.pad(sig_array, (0, max_size - len(sig_array)))
        
        return sig_array
    
    def _get_signature_dimension(self, n_dim: int) -> int:
        """Calculate signature dimension for given state dimension."""
        # Level 0: 1
        # Level 1: n_dim
        # Level 2: n_dim^2
        # Level 3: n_dim^3
        total = 1
        for d in range(1, self.depth + 1):
            total += n_dim ** d
        return min(total, 1024)  # Cap at 1024


class ConditionalFlowMatcher:
    """
    Conditional Flow Matching for Schrödinger Bridge.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_time_steps = config.get("n_time_steps", 30)
        self.n_paths = config.get("n_paths", 100)
        self.temperature = config.get("temperature", 1.0)
        self.convergence_threshold = config.get("convergence_threshold", 1e-4)
        
        # Neural network weights (simplified)
        self.drift_weights = None
        self.diffusion_weights = None
        
    def solve_bridge(self, start_state: np.ndarray, end_state: np.ndarray,
                     signature: np.ndarray, n_steps: int = 30) -> np.ndarray:
        """
        Solve the Schrödinger Bridge between start and end states.
        
        Returns:
            path: (n_steps, state_dim) array representing the bridge
        """
        state_dim = start_state.shape[0]
        
        # Initialize path as linear interpolation
        path = np.linspace(start_state, end_state, n_steps).T
        
        # Add noise for exploration
        noise_scale = 0.1 * np.std(path)
        path += np.random.normal(0, noise_scale, path.shape)
        
        # Iterative refinement using flow matching
        for iteration in range(50):
            # Compute drift from signature-conditioned network
            drift = self._compute_drift(path, signature)
            
            # Update path using drift
            dt = 1.0 / n_steps
            for i in range(n_steps - 1):
                path[i + 1] = path[i] + drift[i] * dt
            
            # Enforce boundary conditions
            path[0] = start_state
            path[-1] = end_state
            
            # Check convergence
            if iteration > 10 and np.max(np.abs(drift)) < self.convergence_threshold:
                break
        
        return path
    
    def _compute_drift(self, path: np.ndarray, signature: np.ndarray) -> np.ndarray:
        """
        Compute drift using signature-conditioned neural network.
        """
        n_steps, state_dim = path.shape
        sig_dim = len(signature)
        
        # Simplified drift computation
        # In practice, this would be a neural network
        drift = np.zeros_like(path)
        
        for i in range(n_steps - 1):
            # Combine state and signature features
            state = path[i]
            combined = np.concatenate([state, signature])
            
            # Simple linear drift (simplified)
            # Neural network would be here
            drift[i] = 0.1 * (np.mean(path) - state) + 0.01 * np.random.randn(state_dim)
        
        return drift
    
    def generate_paths(self, start_state: np.ndarray, end_state: np.ndarray,
                        signature: np.ndarray, n_paths: int = 50) -> List[np.ndarray]:
        """
        Generate multiple bridge paths.
        """
        paths = []
        for _ in range(n_paths):
            # Add small random perturbation for diversity
            perturbed_start = start_state + 0.01 * np.random.randn(len(start_state))
            perturbed_end = end_state + 0.01 * np.random.randn(len(end_state))
            
            path = self.solve_bridge(perturbed_start, perturbed_end, signature)
            paths.append(path)
        
        return paths


class SigBridgeEngine:
    """
    Signature-Conditioned Neural Bridge Engine.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.signature_computer = SignatureComputer(config.get("signature", {}))
        self.flow_matcher = ConditionalFlowMatcher(config.get("bridge", {}))
        self.state_dim = config.get("state_dim", 16)
        
    def encode_state(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """
        Encode market state into latent state vector.
        """
        features = []
        
        # Return features
        if len(returns) > 0:
            recent = returns[-min(20, len(returns)):]
            features.extend([
                np.mean(recent),
                np.std(recent),
                np.percentile(recent, 25) if len(recent) > 1 else 0,
                np.percentile(recent, 75) if len(recent) > 1 else 0,
                recent[-1] if len(recent) > 0 else 0,
                np.mean(recent[-5:]) if len(recent) >= 5 else 0,
                np.mean(recent[-10:]) if len(recent) >= 10 else 0,
            ])
        else:
            features.extend([0] * 7)
        
        # Macro features
        if len(macro) > 0:
            macro_flat = macro.flatten()[:9]
            features.extend(macro_flat.tolist())
        else:
            features.extend([0] * 9)
        
        # Pad to state_dim
        if len(features) < self.state_dim:
            features.extend([0] * (self.state_dim - len(features)))
        else:
            features = features[:self.state_dim]
        
        return np.array(features)
    
    def compute_bridge(self, start_returns: np.ndarray, end_returns: np.ndarray,
                       macro: np.ndarray, window: int = 63) -> Dict:
        """
        Compute signature-conditioned bridge between two market states.
        """
        # Encode start and end states
        start_state = self.encode_state(start_returns, macro)
        end_state = self.encode_state(end_returns, macro)
        
        # Build path from start to end
        path_length = min(len(start_returns), len(end_returns), 100)
        path = np.column_stack([start_returns[-path_length:], end_returns[-path_length:]])
        
        # Compute signature of the path
        signature = self.signature_computer.compute_signature(path)
        
        # Generate bridge paths
        n_paths = min(self.flow_matcher.n_paths, 50)
        paths = self.flow_matcher.generate_paths(start_state, end_state, signature, n_paths)
        
        # Compute bridge statistics
        path_means = np.array([np.mean(p, axis=0) for p in paths])
        path_stds = np.array([np.std(p, axis=0) for p in paths])
        
        # Compute bridge metrics
        bridge_width = np.mean(np.std(paths, axis=0))
        bridge_curvature = np.mean(np.abs(np.diff(paths, axis=1)), axis=(0, 1))
        
        return {
            "bridge_paths": [p.tolist() for p in paths[:5]],  # Store first 5
            "n_paths": len(paths),
            "bridge_width": bridge_width,
            "bridge_curvature": bridge_curvature,
            "start_state": start_state.tolist(),
            "end_state": end_state.tolist(),
            "signature": signature.tolist(),
            "mean_path": np.mean(paths, axis=0).tolist(),
            "std_path": np.std(paths, axis=0).tolist(),
            "error": None
        }


def compute_sig_bridge(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 63
) -> Dict:
    """
    Compute Sig-Bridge for a single ticker.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    macro = macro_df.values
    
    if len(returns) < window + 20:
        return {
            "bridge_width": 0,
            "bridge_curvature": 0,
            "n_paths": 0,
            "z_score": 0,
            "error": "Insufficient data"
        }
    
    try:
        # Use recent window
        recent_returns = returns[-window:]
        macro_window = macro[-min(window, len(macro)):] if len(macro) > 0 else np.zeros((1, 6))
        
        # Split into start and end states
        half_window = window // 2
        start_returns = recent_returns[:half_window]
        end_returns = recent_returns[half_window:]
        
        # Initialize engine
        engine = SigBridgeEngine(config)
        
        # Compute bridge
        result = engine.compute_bridge(start_returns, end_returns, macro_window, window)
        
        if result.get("error"):
            return {
                "bridge_width": 0,
                "bridge_curvature": 0,
                "n_paths": 0,
                "z_score": 0,
                "error": result["error"]
            }
        
        # Compute z-score (bridge width normalized)
        bridge_width = result.get("bridge_width", 0)
        z_score = bridge_width / (np.std(bridge_width) + 1e-6) if bridge_width > 0 else 0
        
        return {
            "bridge_width": bridge_width,
            "bridge_curvature": result.get("bridge_curvature", 0),
            "n_paths": result.get("n_paths", 0),
            "bridge_paths": result.get("bridge_paths", []),
            "mean_path": result.get("mean_path", []),
            "std_path": result.get("std_path", []),
            "z_score": z_score,
            "error": None
        }
    except Exception as e:
        return {
            "bridge_width": 0,
            "bridge_curvature": 0,
            "n_paths": 0,
            "z_score": 0,
            "error": str(e)
        }


def compute_universe_sig_bridge(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 63
) -> Dict:
    """
    Compute Sig-Bridge for all ETFs in a universe.
    """
    results = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_sig_bridge(prices, macro_df, config, window)
        
        results[ticker] = {
            "bridge_width": result.get("bridge_width", 0),
            "bridge_curvature": result.get("bridge_curvature", 0),
            "n_paths": result.get("n_paths", 0),
            "z_score": result.get("z_score", 0),
            "bridge_paths": result.get("bridge_paths", []),
            "mean_path": result.get("mean_path", []),
            "std_path": result.get("std_path", [])
        }
    
    # Normalize z-scores
    z_scores = np.array([r["z_score"] for r in results.values()])
    if len(z_scores) > 1 and np.std(z_scores) > 1e-6:
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        # Use bridge width as fallback
        widths = np.array([r["bridge_width"] for r in results.values()])
        if len(widths) > 1 and np.std(widths) > 1e-6:
            mean_w = np.mean(widths)
            std_w = np.std(widths)
            for ticker, r in results.items():
                r["z_score"] = (r["bridge_width"] - mean_w) / std_w
        else:
            for r in results.values():
                r["z_score"] = 0
    
    return results
