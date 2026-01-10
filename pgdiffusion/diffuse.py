import torch
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def diffuse(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    alpha: float = 0.6,
    n_steps: int = 1,
    add_self_loops: bool = True,
    edge_weight: torch.Tensor | None = None,
    M: torch.Tensor | None = None,
    U: torch.Tensor | None = None,
    beta: float = 0.0,
):
    """
    Random-walk feature diffusion on a pseudotime graph.

    At each step, features are updated as a blend of their own value and the mean
    of their neighbors, with optional feature-to-feature mixing:

    Explicit (square matrix):
        X^{(t+1)} = (1 - alpha) X^{(t)} + alpha * P X^{(t)} M
    Low-rank (efficient for large d):
        X^{(t+1)} = (1 - alpha) X^{(t)} + alpha * P X^{(t)} (I + beta U U^T)

    where:
      - P is the propagation operator (neighbor mean)
      - M is an explicit (d, d) feature-mixing matrix (for small d)
      - U is a (d, r) low-rank feature coupling (for large d)
      - beta is the low-rank coupling strength
      - Only one of M or U may be provided

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix (n_cells, n_features)
    edge_index : torch.Tensor
        Sparse edges (2, n_edges)
    alpha : float
        Blending weight in (0, 1] (default: 0.6)
    n_steps : int
        Number of diffusion iterations (default: 1)
    add_self_loops : bool
        Add residual self-loops (default: True)
    edge_weight : torch.Tensor | None
        Optional per-edge weights for weighted aggregation (default: None)
    M : torch.Tensor | None
        Optional explicit feature-mixing matrix (n_features, n_features).
        Use for small feature spaces (e.g., PCA).
    U : torch.Tensor | None
        Optional low-rank feature coupling (n_features, r).
        Use for large feature spaces (e.g., genes).
    beta : float
        Coupling strength for low-rank U. Ignored if U is None. Default: 0.0.

    Returns
    -------
    X_diffused : torch.Tensor
        Smoothed features, same shape as X

    Raises
    ------
    ValueError
        If both M and U are provided, or if their shapes are invalid.

    Examples
    --------
    >>> import torch
    >>> import scanpy as sc
    >>> import pgdiffusion as pgd
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> trajectories = {"branch": ["cell_0", "cell_1", "cell_2"]}
    >>> X = torch.tensor(adata.obsm["X_pca"], dtype=torch.float32)
    >>> edge_index = pgd.build_graph(adata, trajectories)
    >>> edge_index = torch.tensor(edge_index, dtype=torch.long)
    >>> X_diffused = pgd.diffuse(X, edge_index, alpha=0.6, n_steps=1)
    >>> adata.obsm["X_pseudotime"] = X_diffused.cpu().numpy()

    Weighted edges example
    ----------------------
    Use positional steps from `build_graph(..., include_step_attr=True)` to derive
    per-edge weights (e.g., inverse-distance) and pass them via `edge_weight`.

    >>> import torch
    >>> import numpy as np
    >>> import scanpy as sc
    >>> import pgdiffusion as pgd
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> trajectories = {"branch": ["cell_0", "cell_1", "cell_2"]}
    >>> X = torch.tensor(adata.obsm["X_pca"], dtype=torch.float32)
    >>> edge_index, edge_steps = pgd.build_graph(
    ...     adata, trajectories, include_step_attr=True
    ... )
    >>> edge_index = torch.tensor(edge_index, dtype=torch.long)
    >>> edge_weight = torch.tensor(1.0 / (edge_steps + 1.0), dtype=torch.float32)
    >>> X_diffused = pgd.diffuse(
    ...     X,
    ...     edge_index,
    ...     alpha=0.6,
    ...     n_steps=1,
    ...     add_self_loops=True,
    ...     edge_weight=edge_weight,
    ... )
    >>> adata.obsm["X_pseudotime"] = X_diffused.cpu().numpy()
    """
    N, d = X.shape

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
    if M is not None and U is not None:
        raise ValueError("Provide either M (explicit) or U (low-rank), not both.")
    if M is not None:
        if M.dim() != 2 or M.shape[0] != d or M.shape[1] != d:
            raise ValueError(
                f"M must have shape (n_features, n_features) = ({d}, {d}), "
                f"got {tuple(M.shape)}"
            )
    elif U is not None:
        if U.dim() != 2 or U.shape[0] != d:
            raise ValueError(f"U must have shape (d, r)=({d}, r), got {tuple(U.shape)}")
        if beta < 0:
            raise ValueError("beta must be >= 0")

    src, dst = edge_index

    # Prepare edge weights (default to ones)
    if edge_weight is not None:
        if edge_weight.dim() != 1 or edge_weight.shape[0] != src.shape[0]:
            raise ValueError(
                f"edge_weight must be 1D with length equal to number of edges ({src.shape[0]}), "
                f"got shape {tuple(edge_weight.shape)}"
            )
        w = edge_weight.to(device=X.device, dtype=X.dtype)
    else:
        w = torch.ones(src.shape[0], device=X.device, dtype=X.dtype)

    if add_self_loops:
        self_edges = torch.arange(N, device=X.device)
        src = torch.cat([src, self_edges])
        dst = torch.cat([dst, self_edges])
        w = torch.cat([w, torch.ones(N, device=X.device, dtype=X.dtype)])

    # Precompute in-degree for mean aggregation
    deg = torch.zeros(N, device=X.device, dtype=X.dtype)
    deg.index_add_(0, dst, w)
    deg = deg.clamp_min(1.0)

    # Move M/U to correct device/dtype if provided
    if M is not None:
        M = M.to(device=X.device, dtype=X.dtype)
    elif U is not None:
        U = U.to(device=X.device, dtype=X.dtype)

    H = X
    with Progress(
        SpinnerColumn(),
        TextColumn("Diffusing"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task("diffuse", total=n_steps)
        for _ in range(n_steps):
            agg = torch.zeros_like(H)
            agg.index_add_(0, dst, H[src] * w.unsqueeze(-1))
            agg = agg / deg.unsqueeze(-1)
            # Feature coupling
            if M is not None:
                agg = agg @ M
            elif U is not None and beta != 0.0:
                # agg @ (I + beta U U^T) = agg + beta (agg U) U^T
                agg = agg + beta * ((agg @ U) @ U.T)

            H = (1 - alpha) * H + alpha * agg
            progress.advance(task_id)

    return H


if __name__ == "__main__":
    # Simple test case
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]],
        dtype=torch.long,
    )
    X_diffused = diffuse(X, edge_index, alpha=0.5, n_steps=1)
    print("Original X:\n", X)
    print("Diffused X:\n", X_diffused)
