"""
src/allocation — portfolio weight engine for gestionHvsA-adrsArgy.

Public API:
    compute_weights   — translate regime + asset signals into monthly portfolio weights
    AllocationResult  — dataclass: weights, active_adrs, excluded_adrs, regime, probability
"""
from .allocator import AllocationResult, compute_weights

__all__ = ["compute_weights", "AllocationResult"]
