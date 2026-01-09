from typing import Mapping, Sequence
import warnings
import numpy as np
from anndata import AnnData


def build_graph(
    adata: AnnData,
    branch_trajectories: Mapping[str, Sequence[str]],
    *,
    neighbors_per_side: int = 50,
    include_step_attr: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build sparse pseudotime graph from branched cell trajectories.

    Constructs an undirected graph where edges connect cells within a sliding window
    along their pseudotime ordering. Each branch in the input trajectories defines a
    sequential ordering; cells are connected only to neighbors within ``neighbors_per_side``
    positions along this trajectory. Multiple branches can share edges; the shortest
    positional distance is retained.

    **Mathematical formulation**: For branch trajectory :math:`s = (c_1, c_2, ..., c_L)`,
    cell :math:`c_i` is connected to all :math:`c_j` where :math:`|i - j| \\leq k`,
    excluding self-loops. Edges are bidirectional.

    Parameters
    ----------
    adata
        AnnData object. Cell names (``adata.obs_names``) must match cell identifiers
        in ``branch_trajectories``. Missing cells are silently skipped.
    branch_trajectories
        Mapping from branch name to ordered cell sequences. Example:
        ``{"Epithelial": ["cell_001", "cell_003", ...], "Endothelial": [...]}``
    neighbors_per_side
        Radius of sliding window. Each cell connects to neighbors within this distance
        in trajectory order. Default: 50.
    include_step_attr
        If True, return edge attributes as positional steps within branches.
        Default: False.

    Returns
    -------
    edge_index
        Array of shape (2, E) with dtype int64. Rows are source/destination node indices.
    edge_attr
        (Optional) Array of shape (E,) with dtype float32 containing positional steps.
        Only returned if ``include_step_attr=True``.

    Examples
    --------
    >>> import scanpy as sc
    >>> import pgdiffusion as pgd
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> trajectories = {
    ...     "branch1": ["cell_0", "cell_1", "cell_2"],
    ...     "branch2": ["cell_2", "cell_3", "cell_4"],
    ... }
    >>> edge_index = pgd.build_graph(adata, trajectories, neighbors_per_side=1)
    >>> print(edge_index.shape)
    (2, 8)
    """

    if not branch_trajectories:
        raise ValueError("branch_trajectories cannot be empty")
    if neighbors_per_side < 0:
        raise ValueError(f"neighbors_per_side must be >= 0, got {neighbors_per_side}")

    name_to_idx = {name: i for i, name in enumerate(adata.obs_names)}
    edge_steps: dict[tuple[int, int], int] = {}

    for branch_name, ordered_cells in branch_trajectories.items():
        seq = [name_to_idx[c] for c in ordered_cells if c in name_to_idx]
        missing = [c for c in ordered_cells if c not in name_to_idx]
        if missing:
            warnings.warn(
                f"Branch '{branch_name}': {len(missing)} cell(s) not found in adata.obs_names. "
                f"Skipping: {missing[:3]}{'...' if len(missing) > 3 else ''}",
                UserWarning,
            )
        L = len(seq)
        if L == 0:
            continue

        # Connect within a sliding window of size neighbors_per_side on each side
        for pos, cell_idx in enumerate(seq):
            q_lo = max(0, pos - neighbors_per_side)
            q_hi = min(L, pos + neighbors_per_side + 1)
            for other_pos in range(q_lo, q_hi):
                if other_pos == pos:
                    continue  # skip self-loop

                other_cell_idx = seq[other_pos]
                a, b = (
                    (cell_idx, other_cell_idx)
                    if cell_idx < other_cell_idx
                    else (other_cell_idx, cell_idx)
                )
                step = abs(other_pos - pos)

                prev = edge_steps.get((a, b))
                if prev is None or step < prev:
                    edge_steps[(a, b)] = step

    rows: list[int] = []
    cols: list[int] = []
    steps: list[int] = []
    for (a, b), step in edge_steps.items():
        rows += [a, b]
        cols += [b, a]
        steps += [step, step]

    edge_index = np.array([rows, cols], dtype=np.int64)
    if include_step_attr:
        edge_attr = np.array(steps, dtype=np.float32)
        return edge_index, edge_attr

    return edge_index
