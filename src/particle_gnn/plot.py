"""Centralized plot functions and vectorized helpers for particle-gnn.

All plot functions that were previously scattered across models/utils.py,
models/graph_trainer.py, and generators/graph_data_generator.py are
consolidated here.
"""
from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import torch
import torch.nn as nn
import umap

from particle_gnn.figure_style import default_style, dark_style
from particle_gnn.utils import to_numpy


# --------------------------------------------------------------------------- #
#  Vectorized helpers
# --------------------------------------------------------------------------- #

def build_edge_features(rr, embedding, model_name, max_radius):
    """Build input features for the edge MLP, supporting batched embeddings.

    Args:
        rr: (n_pts,) tensor of radial distances
        embedding: (N, embed_dim) or (n_pts, embed_dim) tensor
        model_name: one of PDE_A, PDE_A_bis, PDE_B, PDE_G, PDE_ParticleField_A, PDE_ParticleField_B
        max_radius: float

    Returns:
        (N, n_pts, input_dim) or (n_pts, input_dim) tensor of features
    """
    # Handle batched case: embedding is (N, embed_dim), rr is (n_pts,)
    if embedding.dim() == 2 and rr.dim() == 1 and embedding.shape[0] != rr.shape[0]:
        N, embed_dim = embedding.shape
        n_pts = rr.shape[0]
        rr_exp = rr[None, :].expand(N, n_pts)  # (N, n_pts)
        emb_exp = embedding[:, None, :].expand(N, n_pts, embed_dim)  # (N, n_pts, embed_dim)
        z = torch.zeros_like(rr_exp)

        match model_name:
            case 'PDE_A' | 'PDE_ParticleField_A':
                return torch.cat((
                    rr_exp.unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    rr_exp.unsqueeze(-1) / max_radius,
                    emb_exp,
                ), dim=-1)
            case 'PDE_A_bis':
                return torch.cat((
                    rr_exp.unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    rr_exp.unsqueeze(-1) / max_radius,
                    emb_exp,
                    emb_exp,
                ), dim=-1)
            case 'PDE_B' | 'PDE_ParticleField_B':
                return torch.cat((
                    rr_exp.unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    torch.abs(rr_exp).unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    emb_exp,
                ), dim=-1)
            case 'PDE_G':
                return torch.cat((
                    rr_exp.unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    rr_exp.unsqueeze(-1) / max_radius,
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    z.unsqueeze(-1),
                    emb_exp,
                ), dim=-1)
            case _:
                raise ValueError(f'Unknown model name in build_edge_features: {model_name}')
    else:
        # Original non-batched path (embedding is (n_pts, embed_dim))
        match model_name:
            case 'PDE_A' | 'PDE_ParticleField_A':
                return torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                  rr[:, None] / max_radius, embedding), dim=1)
            case 'PDE_A_bis':
                return torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                  rr[:, None] / max_radius, embedding, embedding), dim=1)
            case 'PDE_B' | 'PDE_ParticleField_B':
                return torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                  torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                  0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            case 'PDE_G':
                return torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                  rr[:, None] / max_radius, 0 * rr[:, None],
                                  0 * rr[:, None],
                                  0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
            case _:
                raise ValueError(f'Unknown model name in build_edge_features: {model_name}')


def _batched_mlp_eval(mlp, embeddings, rr, model_name, max_radius, device, chunk_size=512):
    """Evaluate an MLP for all particles in batched mode.

    Args:
        mlp: nn.Module — the edge MLP
        embeddings: (N, embed_dim) tensor of particle embeddings
        rr: (n_pts,) tensor of radial sample points
        model_name: str — model name for feature construction
        max_radius: float
        device: torch device
        chunk_size: number of particles per chunk to avoid OOM

    Returns:
        (N, n_pts) tensor of MLP output (first output dim)
    """
    N = embeddings.shape[0]
    n_pts = rr.shape[0]
    results = []

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            emb_chunk = embeddings[start:end]  # (chunk, embed_dim)

            # Build features: (chunk, n_pts, input_dim)
            features = build_edge_features(rr, emb_chunk, model_name, max_radius)
            chunk_n = features.shape[0]

            # Flatten to (chunk * n_pts, input_dim), run MLP, reshape back
            features_flat = features.reshape(chunk_n * n_pts, -1)
            out = mlp(features_flat.float())[:, 0]  # (chunk * n_pts,)
            results.append(out.reshape(chunk_n, n_pts))

    return torch.cat(results, dim=0)  # (N, n_pts)


def _plot_curves_fast(ax, rr, func_matrix, type_list, cmap, ynorm=1.0, subsample=None, alpha=0.25, linewidth=1):
    """Plot N curves using a single LineCollection.

    Args:
        ax: matplotlib Axes
        rr: (n_pts,) numpy array of x values
        func_matrix: (N, n_pts) numpy array of y values
        type_list: (N,) numpy array of int type labels for coloring
        cmap: CustomColorMap instance
        ynorm: scalar or numpy array to multiply y values by
        subsample: int or None — plot every `subsample`-th curve. None plots all.
        alpha: float
        linewidth: float
    """
    N = func_matrix.shape[0]
    if subsample is not None:
        indices = np.arange(0, N, subsample)
    else:
        indices = np.arange(N)

    if len(indices) == 0:
        return

    # Build line segments for LineCollection
    rr_np = np.asarray(rr)
    ynorm_val = float(ynorm) if np.isscalar(ynorm) else np.asarray(ynorm)
    segments = []
    colors = []
    for i in indices:
        y_vals = func_matrix[i] * ynorm_val
        pts = np.column_stack([rr_np, y_vals])
        segments.append(pts)
        colors.append(cmap.color(int(type_list[i])))

    lc = mcoll.LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale_view()


def _vectorized_linear_fit(x, y):
    """Vectorized closed-form least-squares linear fit.

    Args:
        x: (N,) tensor
        y: (N,) tensor

    Returns:
        slope, intercept as scalars
    """
    N = x.shape[0]
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sx2 = (x * x).sum()
    denom = N * sx2 - sx * sx
    slope = (N * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / N
    return slope, intercept


# --------------------------------------------------------------------------- #
#  Embedding helpers
# --------------------------------------------------------------------------- #

def get_embedding(model_a=None, dataset_number=0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())
    return embedding


def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())
    indexes = np.arange(n_frames) * n_particles + cell_id
    return embedding[indexes]


def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    indexes = np.arange(n_frames) * n_particles + cell_id
    return new_labels[indexes]


# --------------------------------------------------------------------------- #
#  analyze_edge_function — vectorized
# --------------------------------------------------------------------------- #

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_particles=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    dimension = config.simulation.dimension
    config_model = config.graph_model.particle_model_name

    if rr == []:
        if config_model == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')

    # Build all embeddings at once: (N, embed_dim)
    if len(model.a.shape) == 3:
        all_embeddings = model.a[0, :n_particles, :]  # (N, embed_dim)
    else:
        all_embeddings = model.a[:n_particles, :]  # (N, embed_dim)

    if config.training.do_tracking:
        pass  # embeddings used directly
    elif (update_type != 'NA') & model.embedding_trial:
        b_rep = model.b[0].clone().detach().repeat(1, 1).expand(n_particles, -1)
        all_embeddings = torch.cat((all_embeddings, b_rep), dim=1)

    # Batched MLP evaluation: (N, 1000)
    func_list = _batched_mlp_eval(model_MLP, all_embeddings, rr, config_model, max_radius, device)

    func_list_ = to_numpy(func_list)

    if vizualize:
        fig = plt.gcf()
        ax = plt.gca()

        # Determine subsampling
        if n_particles <= 200:
            subsample = 1
        else:
            subsample = max(1, n_particles // 200)

        _plot_curves_fast(
            ax, to_numpy(rr), func_list_,
            type_list.flatten() if type_list is not None else np.zeros(n_particles),
            cmap, ynorm=to_numpy(ynorm),
            subsample=subsample, alpha=0.25, linewidth=1,
        )

        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xlim([1E-3, 0.02])
        plt.ylim(config.plotting.ylim)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)

    return func_list, proj_interaction


# --------------------------------------------------------------------------- #
#  plot_training — vectorized
# --------------------------------------------------------------------------- #

def plot_training(config, pred, gt, log_dir, epoch, N, x, index_particles, n_particles, n_particle_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    do_tracking = train_config.do_tracking
    max_radius = simulation_config.max_radius
    n_runs = train_config.n_runs
    dimension = simulation_config.dimension
    type_col = 1 + 2 * dimension

    matplotlib.rcParams['savefig.pad_inches'] = 0

    # --- Embedding scatter plot ---
    if n_runs == 3:
        fig = plt.figure(figsize=(24, 8))
        ax = fig.add_subplot(1, 3, 1)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(1, 3, 3)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(1, 3, 2)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
    elif n_runs > 10:
        fig = plt.figure(figsize=(8, 8))
        for m in range(1, n_runs):
            embedding = get_embedding(model.a, m)
            plt.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=1)
    else:
        fig = plt.figure(figsize=(8, 8))
        if do_tracking:
            embedding = to_numpy(model.a)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=1)
        elif simulation_config.state_type == 'sequence':
            embedding = to_numpy(model.a[0].squeeze())
            plt.scatter(embedding[:-200, 0], embedding[:-200, 1], color='k', s=0.1)
        else:
            embedding = get_embedding(model.a, plot_config.data_embedding)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=1)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    # --- Pred vs true scatter ---
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(to_numpy(gt[:, 0]), to_numpy(pred[:, 0]), c='r', s=1)
    plt.scatter(to_numpy(gt[:, 1]), to_numpy(pred[:, 1]), c='g', s=1)
    plt.xlabel('true value', fontsize=14)
    plt.ylabel('pred value', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/prediction/{epoch}_{N}.tif", dpi=87)
    plt.close()

    # --- Interaction function curves (vectorized) ---
    if n_runs > 10:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 1000)).to(device)

        # Build all (n, k) pair features: n_runs-1 * n_particles^2 combinations
        all_funcs = []
        for m in range(1, n_runs):
            # Batched: for each m, build (n_particles * n_particles, n_pts) via pairs
            emb_n = model.a[m, :n_particles, :]  # (N, embed_dim)
            emb_k = model.a[m, :n_particles, :]  # (N, embed_dim)
            # Expand to all pairs (N*N, embed_dim)
            emb_n_rep = emb_n.unsqueeze(1).expand(-1, n_particles, -1).reshape(-1, emb_n.shape[-1])
            emb_k_rep = emb_k.unsqueeze(0).expand(n_particles, -1, -1).reshape(-1, emb_k.shape[-1])

            n_pts = rr.shape[0]
            n_pairs = emb_n_rep.shape[0]

            # Build features for all pairs
            rr_exp = rr[None, :].expand(n_pairs, n_pts)
            z = torch.zeros_like(rr_exp)
            emb_n_exp = emb_n_rep[:, None, :].expand(-1, n_pts, -1)
            emb_k_exp = emb_k_rep[:, None, :].expand(-1, n_pts, -1)

            features = torch.cat((
                rr_exp.unsqueeze(-1),
                z.unsqueeze(-1),
                z.unsqueeze(-1),
                emb_n_exp,
                emb_k_exp,
            ), dim=-1)

            # Chunk MLP evaluation
            chunk_size = 512
            funcs = []
            with torch.no_grad():
                for start in range(0, n_pairs, chunk_size):
                    end = min(start + chunk_size, n_pairs)
                    feat_flat = features[start:end].reshape(-1, features.shape[-1])
                    out = model.lin_edge(feat_flat.float())[:, 0]
                    funcs.append(out.reshape(end - start, n_pts))
            funcs = torch.cat(funcs, dim=0)  # (n_pairs, n_pts)
            all_funcs.append(funcs)

        all_funcs = torch.cat(all_funcs, dim=0)
        rr_np = to_numpy(rr)
        ynorm_np = to_numpy(ynorm)

        # Plot with LineCollection
        segments = []
        for i in range(all_funcs.shape[0]):
            y_vals = to_numpy(all_funcs[i]) * ynorm_np
            pts = np.column_stack([rr_np, y_vals])
            segments.append(pts)
        colors = ['b'] * len(segments)
        lc = mcoll.LineCollection(segments, colors=colors, linewidths=2, alpha=0.1)
        ax.add_collection(lc)
        ax.autoscale_view()

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.tif", dpi=87)
        plt.close()
    else:
        match model_config.particle_model_name:

            case 'PDE_A' | 'PDE_A_bis' | 'PDE_ParticleField_A' | 'PDE_E' | 'PDE_G':
                fig = plt.figure(figsize=(12, 12))
                if axis:
                    ax = fig.add_subplot(1, 1, 1)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    plt.xticks(fontsize=32)
                    plt.yticks(fontsize=32)
                    plt.xlim([0, simulation_config.max_radius])
                    plt.tight_layout()
                else:
                    ax = fig.add_subplot(1, 1, 1)

                rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 1000)).to(device)

                # Vectorized: build all embeddings and eval MLP in batch
                if do_tracking:
                    all_embeddings = model.a[:n_particles, :]
                else:
                    all_embeddings = model.a[0, :n_particles, :]

                func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                              config.graph_model.particle_model_name,
                                              simulation_config.max_radius, device)

                # Plot with LineCollection
                rr_np = to_numpy(rr)
                ynorm_np = to_numpy(ynorm)
                type_arr = to_numpy(x[:n_particles, type_col]).astype(int)

                subsample = 5 if n_runs <= 5 else 1
                _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                                  ynorm=ynorm_np, subsample=subsample, alpha=0.25, linewidth=2)

                if (model_config.particle_model_name == 'PDE_G') | (model_config.particle_model_name == 'PDE_E'):
                    plt.xlim([0, 0.02])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.tif", dpi=87)
                plt.close()

            case 'PDE_B' | 'PDE_ParticleField_B':
                max_radius_plot = 0.04
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1)
                rr = torch.tensor(np.linspace(-max_radius_plot, max_radius_plot, 1000)).to(device)

                # Vectorized MLP evaluation
                if do_tracking:
                    all_embeddings = model.a[:n_particles, :]
                else:
                    all_embeddings = model.a[0, :n_particles, :]

                func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                              config.graph_model.particle_model_name,
                                              max_radius_plot, device)

                # Plot with LineCollection
                rr_np = to_numpy(rr)
                ynorm_np = to_numpy(ynorm)
                type_arr = np.array([int(n // (n_particles / n_particle_types)) for n in range(n_particles)])

                _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                                  ynorm=ynorm_np, subsample=5, alpha=1.0, linewidth=2)

                if not do_tracking:
                    plt.ylim(config.plotting.ylim)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                from matplotlib.ticker import FormatStrFormatter
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.tif", dpi=170.7)
                plt.close()


# --------------------------------------------------------------------------- #
#  plot_training_particle_field — vectorized
# --------------------------------------------------------------------------- #

def plot_training_particle_field(config, has_siren, has_siren_time, model_f, n_frames, model_name, log_dir, epoch, N, x, x_mesh, index_particles, n_neurons, n_neuron_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    dimension = simulation_config.dimension
    type_col = 1 + 2 * dimension

    max_radius = simulation_config.max_radius
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))

    # --- Embedding scatter ---
    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
    else:
        plt.axis('off')
    embedding = get_embedding(model.a, dataset_num)
    if n_neuron_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, type_col]) / n_neurons, s=1, cmap='viridis')
    else:
        for n in range(n_neuron_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=1)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{model_name}_embedding_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    # --- Interaction function curves (vectorized) ---
    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, simulation_config.max_radius])
        plt.tight_layout()
    else:
        ax = fig.add_subplot(1, 1, 1)

    match model_config.particle_model_name:
        case 'PDE_ParticleField_A':
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
        case 'PDE_ParticleField_B':
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 200)).to(device)

    # Vectorized: all neurons at once
    all_embeddings = model.a[dataset_num, :n_neurons, :]  # (N, embed_dim)
    func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                  model_config.particle_model_name,
                                  max_radius, device)

    # Plot with LineCollection
    rr_np = to_numpy(rr)
    ynorm_np = to_numpy(ynorm)
    type_arr = to_numpy(x[:n_neurons, type_col]).astype(int)

    _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                      ynorm=ynorm_np, subsample=5, alpha=0.25, linewidth=8)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/{model_name}_function_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    # --- Siren field visualization ---
    if has_siren:
        if has_siren_time:
            frame_list = [54, 58, 62, 66]
        else:
            frame_list = [0]

        for frame in frame_list:
            if has_siren_time:
                with torch.no_grad():
                    tmp = model_f(time=frame / n_frames) ** 2
            else:
                with torch.no_grad():
                    tmp = model_f() ** 2
            tmp = torch.reshape(tmp, (n_nodes_per_axis, n_nodes_per_axis))
            tmp = to_numpy(torch.sqrt(tmp))
            if has_siren_time:
                tmp = np.rot90(tmp, k=1)
            fig_ = plt.figure(figsize=(14, 12))
            axf = fig_.add_subplot(1, 1, 1)
            plt.imshow(tmp, cmap='grey')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/external_input/{model_name}_{epoch}_{N}_{frame}.tif", dpi=80)
            plt.close()


# --------------------------------------------------------------------------- #
#  Vectorized sparsity MLP evaluation
# --------------------------------------------------------------------------- #

def batched_sparsity_mlp_eval(model, rr, n_particles, config, device):
    """Evaluate the edge MLP for all particles in batch mode, for sparsity fitting.

    Returns:
        pred: (N, n_pts, output_dim) tensor
    """
    mc = config.graph_model
    sim = config.simulation
    all_embeddings = model.a[0, :n_particles, :].clone().detach()  # (N, embed_dim)

    # Build features: (N, n_pts, input_dim)
    features = build_edge_features(rr, all_embeddings, mc.particle_model_name, sim.max_radius)
    N, n_pts, input_dim = features.shape

    # Flatten, run MLP, reshape
    features_flat = features.reshape(N * n_pts, input_dim)
    pred_flat = model.lin_edge(features_flat.float())  # (N * n_pts, output_dim)
    output_dim = pred_flat.shape[1]
    pred = pred_flat.reshape(N, n_pts, output_dim)

    return pred
