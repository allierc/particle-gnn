import torch
import numpy as np
import torch.nn as nn

from particle_gnn.models.MLP import MLP
from particle_gnn.models.registry import get_model_class
from particle_gnn.utils import to_numpy, fig_init, choose_boundary_values


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the Apache License, Version 2.0
    # found in the LICENSE file in the root directory of this source tree.

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):

            # student_output = F.normalize(student_output, eps=eps, p=2, dim=0)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()

        return loss


def choose_training_model(model_config=None, device=None):
    """Create and return a model based on the configuration.

    Uses the model registry to look up the appropriate class.

    Args:
        model_config: Configuration object containing simulation and graph model parameters.
        device: Torch device to place the model on.

    Returns:
        Tuple of (model, bc_pos, bc_dpos).
    """

    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension
    name = model_config.graph_model.particle_model_name

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model_cls = get_model_class(name)
    model = model_cls(
        aggr_type=aggr_type,
        config=model_config,
        device=device,
        bc_dpos=bc_dpos,
        dimension=dimension,
    )
    model.edges = []

    return model, bc_pos, bc_dpos


def get_type_list(x, dimension):
    from particle_gnn.particle_state import ParticleState
    if isinstance(x, ParticleState):
        return x.particle_type.clone().detach().unsqueeze(-1).float()
    type_col = 1 + 2 * dimension
    type_list = x[:, type_col:type_col + 1].clone().detach()
    return type_list


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size
    return get_batch_size


def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size
    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[], lr_update=[], lr_W=[], lr_modulation=[], learning_rate_nnr=[], learning_rate_edge_embedding=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    if lr_update == []:
        lr_update = lr

    optimizer = torch.optim.Adam([model.a], lr=lr_embedding)
    for name, parameter in model.named_parameters():
        if (parameter.requires_grad) & (name != 'a'):
            if (name == 'b') or ('lin_modulation' in name):
                optimizer.add_param_group({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_W})
            elif 'NNR' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_nnr})
            elif 'edges_embedding' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_edge_embedding})
            else:
                optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params
