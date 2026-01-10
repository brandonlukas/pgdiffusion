import math
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
    edge_weight: torch.Tensor | None = None,
    add_self_loops: bool = False,
    self_loop_weight: float = 1.0,
    M: torch.Tensor | None = None,
    U: torch.Tensor | None = None,
    beta: float = 0.0,
):
    """
    Random-walk feature diffusion on a pseudotime graph.

    We update features by blending a residual term with an incoming-neighbor
    (src -> dst) weighted mean aggregation, followed by optional feature coupling.

    **Mathematical formulation**: In the simplest case (unweighted graph, no feature
    coupling), let H^(t) in R^(N x d) denote the feature matrix at iteration t. One
    diffusion step is given by

        H^(t+1) = (1 - alpha) H^(t) + alpha * P H^(t),

    where the propagation operator P is defined as

        P = D^(-1) A.

    Here A is the adjacency matrix with entries A[i, s] = 1 if there is an edge
    s -> i and 0 otherwise, and D is the diagonal in-degree matrix with
    D[i, i] = sum_s A[i, s].

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix (n_cells, n_features)
    edge_index : torch.Tensor
        Sparse edges (2, n_edges), with (src, dst)
    alpha : float
        Blend weight in [0, 1]
    n_steps : int
        Number of diffusion iterations
    edge_weight : torch.Tensor | None
        Optional per-edge weights aligned with edge_index columns (n_edges,)
    add_self_loops : bool
        If True, include self-loops in the aggregation operator (default: False)
    self_loop_weight : float
        Weight assigned to self-loop edges when add_self_loops=True (default: 1.0)
    M : torch.Tensor | None
        Optional explicit feature mixing matrix (d, d)
    U : torch.Tensor | None
        Optional low-rank feature coupling (d, r) (e.g., PCA loadings)
    beta : float
        Coupling strength for low-rank U; ignored if U is None

    Returns
    -------
    torch.Tensor
        Smoothed features with the same shape as ``X``.

    Raises
    ------
    ValueError
        If both ``M`` and ``U`` are provided, or if shapes are invalid.

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
    >>> X_smooth = pgd.diffuse(X, edge_index, alpha=0.6, n_steps=1)
    >>> adata.obsm["X_pseudotime"] = X_smooth.cpu().numpy()

    Weighted edges example
    ----------------------
    Use positional steps from `build_graph(..., include_step_attr=True)` to derive
    per-edge weights (e.g., inverse-distance) and pass them via `edge_weight`.

    >>> edge_index, edge_steps = pgd.build_graph(
    ...     adata, trajectories, include_step_attr=True
    ... )
    >>> edge_index = torch.tensor(edge_index, dtype=torch.long)
    >>> edge_weight = torch.tensor(1.0 / (edge_steps + 1.0), dtype=torch.float32)
    >>> X_smooth = pgd.diffuse(
    ...     X,
    ...     edge_index,
    ...     alpha=0.6,
    ...     n_steps=1,
    ...     edge_weight=edge_weight,
    ... )

    Feature coupling (PCA loadings) example
    ---------------------------------------
    Apply low-rank feature coupling using PCA loadings U (e.g., Scanpy stores gene
    loadings in ``adata.varm["PCs"]``). This biases diffusion toward the principal
    subspace while preserving full feature dimensionality.

    >>> X = torch.tensor(adata.X.toarray(), dtype=torch.float32)  # (n_cells, n_genes)
    >>> U = torch.tensor(adata.varm["PCs"], dtype=torch.float32)  # (n_genes, n_pcs)
    >>> X_smooth = pgd.diffuse(X, edge_index, alpha=0.5, n_steps=5, U=U, beta=0.3)
    """
    N, d = X.shape

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if edge_index.min() < 0 or edge_index.max() >= N:
        raise ValueError(f"edge_index contains node ids outside [0, {N-1}]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if add_self_loops:
        if not math.isfinite(self_loop_weight):
            raise ValueError("self_loop_weight must be finite")
        if self_loop_weight < 0:
            raise ValueError("self_loop_weight must be >= 0")
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

    edge_index = edge_index.to(device=X.device)
    src, dst = edge_index

    # Prepare edge weights (default to ones)
    if edge_weight is not None:
        if edge_weight.dim() != 1 or edge_weight.shape[0] != src.shape[0]:
            raise ValueError(
                f"edge_weight must be 1D with length equal to number of edges ({src.shape[0]}), "
                f"got shape {tuple(edge_weight.shape)}"
            )
        if not torch.isfinite(edge_weight).all():
            raise ValueError("edge_weight must contain only finite values")
        if (edge_weight < 0).any():
            raise ValueError("edge_weight must be >= 0")
        w = edge_weight.to(device=X.device, dtype=X.dtype)
    else:
        w = torch.ones(src.shape[0], device=X.device, dtype=X.dtype)

    # (optional) include self-loops inside the aggregation operator
    if add_self_loops:
        self_edges = torch.arange(N, device=X.device)
        src = torch.cat([src, self_edges])
        dst = torch.cat([dst, self_edges])
        w = torch.cat(
            [
                w,
                torch.full(
                    (N,),
                    float(self_loop_weight),
                    device=X.device,
                    dtype=X.dtype,
                ),
            ]
        )

    # Precompute weighted in-degree for mean aggregation into dst
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
    X_smooth = diffuse(X, edge_index, alpha=0.5, n_steps=1)
    print("Original X:\n", X)
    print("Smoothed X:\n", X_smooth)
