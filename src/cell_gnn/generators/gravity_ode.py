
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


@register_simulator("gravity_ode")
class GravityODE(nn.Module):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of cells according to the gravity law as a function of their relative position.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dd_pos : float
        the acceleration of the cells (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], clamp=[], pred_limit=[], bc_dpos=[], dimension=2):
        super().__init__()

        self.p = p
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.bc_dpos = bc_dpos
        self.dimension = dimension

    def forward(self, state: CellState, edge_index: torch.Tensor):
        edge_index = remove_self_loops(edge_index)

        mass = self.p[to_numpy(state.cell_type)]
        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        mass_j = mass[src, None]

        messages = self.message(pos_i, pos_j, mass_j)
        dd_pos = scatter_aggregate(messages, dst, state.n_cells, 'add')
        return dd_pos

    def message(self, pos_i, pos_j, mass_j):
        distance_ij = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1))
        distance_ij = torch.clamp(distance_ij, min=self.clamp)
        direction_ij = self.bc_dpos(pos_j - pos_i) / distance_ij[:,None]
        dd_pos = mass_j * direction_ij / (distance_ij[:,None] ** 2)

        return torch.clamp(dd_pos, max=self.pred_limit)

    def psi(self, r, p):
        r_ = torch.clamp(r, min=self.clamp)
        psi = p * r / r_ ** 3
        psi = torch.clamp(psi, max=self.pred_limit)

        return psi[:, None]
