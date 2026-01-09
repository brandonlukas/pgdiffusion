"""
PGD: Pseudotime Graph Diffusion

Random-walk feature diffusion on pseudotime graphs for single-cell analysis.
Bridges trajectory inference with trajectory-aware smoothing of cell embeddings.
"""

from .build_graph import build_graph
from .diffuse import diffuse

__version__ = "0.1.1"
__all__ = [
    "build_graph",
    "diffuse",
]
