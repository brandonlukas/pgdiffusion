import torch


def diffuse(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    alpha: float = 0.6,
    n_steps: int = 1,
    add_self_loops: bool = True,
):
    """Random-walk feature diffusion on pseudotime graph.

    Smooths feature values across neighboring cells in the pseudotime graph using
    iterative aggregation. At each step, each cell's features are updated as a
    weighted blend of its own features and the mean features of its neighbors.

    **Mathematical operation**: At iteration :math:`t`,
    :math:`X^{(t+1)}_i = (1 - \\alpha) X^{(t)}_i + \\alpha \\cdot \\text{mean}_{j \\in N(i)} X^{(t)}_j`

    where :math:`N(i)` is the set of neighbors of node :math:`i` in the graph.
    Higher :math:`\\alpha` increases smoothing; :math:`\\alpha = 0` returns unchanged features,
    :math:`\\alpha = 1` returns pure neighbor aggregation.

    Uses in-place PyTorch scatter operations (``index_add_``) for GPU efficiency.
    Degree clamping prevents division by zero for isolated nodes.

    Parameters
    ----------
    X
        Feature matrix of shape (n_cells, n_features) with dtype float32 or float64.
        Typically embeddings (e.g., PCA coordinates).
    edge_index
        Sparse graph edges of shape (2, n_edges) with dtype int64.
        Row 0: source indices, Row 1: destination indices.
    alpha
        Blending weight for neighbor features, in [0, 1]. Default: 0.6.
        - 0.0 = no diffusion (identity, returns original X)
        - 0.6 = balanced blend (recommended)
        - 1.0 = full neighbor averaging
    n_steps
        Number of diffusion iterations. Default: 1.
        More steps = more smoothing (cumulative effect).
    add_self_loops
        If True, add self-loops to graph before diffusion (residual connection).
        Default: True.

    Returns
    -------
    X_diffused
        Diffused feature matrix, same shape and dtype as input X.

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
    """
    N, d = X.shape

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")

    src, dst = edge_index

    if add_self_loops:
        self_edges = torch.arange(N, device=X.device)
        src = torch.cat([src, self_edges])
        dst = torch.cat([dst, self_edges])

    # Precompute in-degree for mean aggregation
    deg = torch.zeros(N, device=X.device, dtype=X.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=X.dtype))
    deg = deg.clamp_min(1.0)

    H = X
    for _ in range(n_steps):
        # aggregate neighbor features into dst
        agg = torch.zeros_like(H)
        agg.index_add_(0, dst, H[src])  # sum over incoming neighbors
        agg = agg / deg.unsqueeze(-1)  # mean

        H = (1 - alpha) * H + alpha * agg  # residual mixing

    return H
