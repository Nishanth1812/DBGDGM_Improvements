import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
import math

import networkx as nx

from .utils import sample_pos_neg_edges, gumbel_softmax, bce_loss, kld_z_loss, kld_gauss, reparameterized_sample, \
    get_status, divide_graph_snapshots

LOSS_KEYS = ['nll', 'kld_z', 'kld_alpha', 'kld_beta', 'kld_phi']


class Model(nn.Module):
    """
    DBGDGM model definition with Improvements (Fix 1 and Fix 2).

    Attributes:
    ------------
    num_samples: int
        The number of subjects in dataset.
    num_nodes: int
        The number of nodes in the brain graphs.
    embedding_dim: int
        Embedding dimensionality for node (phi), community (beta), subject (alpha) embeddings.
    categorical_dim: int
        Number of communities in brain graphs.
    gamma: float
        Temporal smoothness for community embeddings.
    sigma: float
        Temporal smoothness for node embeddings.
    device: str
        Training/Inference device.
    """

    def __init__(self, num_samples, num_nodes,
                 embedding_dim=32,
                 categorical_dim=3,
                 gamma=0.1,
                 sigma=1.,
                 device=torch.device("cpu")):

        super().__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.gamma = gamma
        self.sigma = sigma
        self.device = device

        self.beta_mean = nn.Linear(self.embedding_dim, embedding_dim)
        self.beta_std = nn.Sequential(nn.Linear(self.embedding_dim, embedding_dim), nn.Softplus())
        self.phi_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.phi_std = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Softplus())
        self.nn_pi = nn.Linear(self.embedding_dim, self.categorical_dim, bias=False)

        # Fix 1: Input size remains 2*embedding_dim, second slot mapped to neighbor aggregates
        self.rnn_nodes = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)
        self.rnn_comms = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)

        self.alpha_mean = nn.Embedding(self.num_samples, self.embedding_dim)
        self.alpha_std = nn.Embedding(self.num_samples, self.embedding_dim)

        self.subject_to_phi = nn.Linear(self.embedding_dim, self.num_nodes * self.embedding_dim)
        self.subject_to_beta = nn.Linear(self.embedding_dim, self.categorical_dim * self.embedding_dim)

        self.alpha_mean_prior = torch.zeros(self.embedding_dim)
        self.alpha_std_scalar = 1.

        self.decoder = nn.Sequential(nn.Linear(embedding_dim, num_nodes, bias=False))

        # Fix 2: Learned Transition Priors for temporal embedding shift
        self.phi_transition = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.beta_transition = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def _aggregate_neighbors(self, phi: torch.Tensor, graph: nx.Graph) -> torch.Tensor:
        """Fix 1: Aggregates embeddings of neighbors via vectorized scatter_add."""
        num_nodes = phi.shape[0]

        edges = list(graph.edges())
        if not edges:
            return phi.clone()

        # Build symmetric edge index (u->v and v->u)
        src = torch.tensor([u for u, v in edges] + [v for u, v in edges],
                           dtype=torch.long, device=self.device)
        dst = torch.tensor([v for u, v in edges] + [u for u, v in edges],
                           dtype=torch.long, device=self.device)

        # Scatter-add neighbor embeddings then normalise by degree
        agg = torch.zeros_like(phi)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, phi.shape[1]), phi[src])

        degree = torch.zeros(num_nodes, device=self.device)
        degree.scatter_add_(0, dst, torch.ones(len(dst), device=self.device))

        # Isolated nodes (degree==0) fall back to their own embedding
        isolated = (degree == 0)
        agg[isolated] = phi[isolated]
        degree = degree.clamp(min=1).unsqueeze(1)

        return agg / degree


    def _update_hidden_states(self, phi_prior_mean, beta_prior_mean, h_phi, h_beta, graph=None):
        """Fix 1 implementation feeding graph neighbor means to Node GRU."""
        if graph is not None:
            neighbor_context = self._aggregate_neighbors(phi_prior_mean, graph)
        else:
            neighbor_context = phi_prior_mean

        nodes_in = torch.cat([phi_prior_mean, neighbor_context], dim=-1).view(1, self.num_nodes, 2 * self.embedding_dim)
        _, h_phi = self.rnn_nodes(nodes_in, h_phi)

        comms_in = torch.cat([beta_prior_mean, beta_prior_mean], dim=-1).view(1, self.categorical_dim, 2 * self.embedding_dim)
        _, h_beta = self.rnn_comms(comms_in, h_beta)

        return h_phi, h_beta

    def _sample_embeddings(self, h_phi, h_beta):
        beta_mean_t = self.beta_mean(h_beta[-1])
        beta_std_t = self.beta_std(h_beta[-1])

        phi_mean_t = self.phi_mean(h_phi[-1])
        phi_std_t = self.phi_std(h_phi[-1])

        beta_sample = reparameterized_sample(beta_mean_t, beta_std_t)
        phi_sample = reparameterized_sample(phi_mean_t, phi_std_t)

        return (beta_sample, beta_mean_t, beta_std_t), (phi_sample, phi_mean_t, phi_std_t)

    def _edge_reconstruction(self, w, c, phi_sample, beta_sample, temp):
        q = self.nn_pi(phi_sample[w] * phi_sample[c])  # q(z|w, c)
        p_prior = self.nn_pi(phi_sample[w])  # p(z|w)

        if self.training:
            z = gumbel_softmax(q, self.device, tau=temp, hard=True)
        else:
            z = F.softmax(q, dim=-1)

        beta_mixture = torch.mm(z, beta_sample)  # Community mixture embeddings
        p_c_given_z = self.decoder(beta_mixture)  # p(c|z)

        return p_c_given_z, F.softmax(q, dim=-1), F.softmax(p_prior, dim=-1)

    def _initialize_subject(self, subject_idx):
        alpha_mean_n = self.alpha_mean.weight[subject_idx]
        alpha_std_n = F.softplus(self.alpha_std.weight[subject_idx])
        alpha_n = alpha_mean_n  # Converges faster than sampling but might overfit
        kld_alpha = kld_gauss(alpha_mean_n, alpha_std_n, self.alpha_mean_prior.to(self.device), self.alpha_std_scalar)
        phi_0_mean = self.subject_to_phi(alpha_n).view(self.num_nodes, self.embedding_dim)
        beta_0_mean = self.subject_to_beta(alpha_n).view(self.categorical_dim, self.embedding_dim)
        return alpha_n, kld_alpha, phi_0_mean, beta_0_mean

    def forward(self, batch_data, valid_prop=0.1, test_prop=0.1, temp=1.):
        loss = {key: 0 for key in LOSS_KEYS}
        for data in batch_data:
            subject_idx, dynamic_graph, _ = data
            subject_loss = self._forward(subject_idx, dynamic_graph, valid_prop, test_prop, temp)
            for loss_name in LOSS_KEYS:
                loss[loss_name] += subject_loss[loss_name]
        return loss

    def _forward(self, subject_idx, batch_graphs, valid_prop, test_prop, temp):
        loss = {key: 0 for key in LOSS_KEYS}
        edge_counter = 0

        train_time, valid_time, test_time = divide_graph_snapshots(len(batch_graphs), valid_prop, test_prop)
        alpha_n, kld_alpha, phi_prior_mean, beta_prior_mean = self._initialize_subject(subject_idx)

        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        for snapshot_idx in range(0, train_time):
            graph = batch_graphs[snapshot_idx]
            train_edges = [(u, v) for u, v in graph.edges()]
            if self.training:
                np.random.shuffle(train_edges)

            batch = torch.LongTensor(train_edges).to(self.device)
            assert batch.shape == (len(train_edges), 2)
            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))

            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta, graph=graph)

            (beta_sample, beta_mean_t, beta_std_t), (phi_sample, phi_mean_t, phi_std_t) = self._sample_embeddings(h_phi, h_beta)

            recon, posterior_z, prior_z = self._edge_reconstruction(w, c, phi_sample, beta_sample, temp)

            # Fix 2: Learned prior updates transition
            beta_prior_mean = self.beta_transition(beta_sample)
            phi_prior_mean = self.phi_transition(phi_sample)

            loss['nll'] += bce_loss(recon, c)
            loss['kld_z'] += kld_z_loss(posterior_z, prior_z)
            loss['kld_alpha'] += kld_alpha
            loss['kld_beta'] += kld_gauss(beta_sample, beta_std_t, beta_prior_mean, self.gamma)
            loss['kld_phi'] += kld_gauss(phi_sample, phi_std_t, phi_prior_mean, self.sigma)
            edge_counter += c.shape[0]

        for loss_name in loss.keys():
            loss[loss_name] = loss[loss_name] / edge_counter

        return loss

    def predict_auc_roc_precision(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        aucroc = {'train': 0, 'valid': 0, 'test': 0}
        ap = {'train': 0, 'valid': 0, 'test': 0}
        nll = {'train': 0, 'valid': 0, 'test': 0}

        num_subjects = len(subject_graphs)
        for subject in subject_graphs:
            subject_idx, subject_graphs_s, gender_label = subject
            pred, label, _nll = self._predict_auc_roc_precision(subject_idx,
                                                                subject_graphs_s,
                                                                valid_prop,
                                                                test_prop)

            for status in ['train', 'valid', 'test']:
                aucroc[status] += roc_auc_score(label[status], pred[status]) / num_subjects
                ap[status] += average_precision_score(label[status], pred[status]) / num_subjects
                nll[status] += _nll[status].mean() / num_subjects

        return nll, aucroc, ap

    def _predict_auc_roc_precision(self, subject_idx, batch_graphs, valid_prop, test_prop):
        def _get_edge_reconstructions(edges, phi_sample, beta_sample):
            batch = torch.LongTensor(edges).to(self.device)
            assert batch.shape == (len(edges), 2)
            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))
            p_c_given_z, _, _ = self._edge_reconstruction(w, c, phi_sample, beta_sample, 1)  # Model not in train
            p_c_gt = p_c_given_z.gather(1, c.unsqueeze(dim=1)).squeeze(dim=-1).detach().cpu()
            return p_c_given_z, p_c_gt, w, c

        _, _, phi_prior_mean, beta_prior_mean = self._initialize_subject(subject_idx)

        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        pred = {'train': [], 'valid': [], 'test': []}
        label = {'train': [], 'valid': [], 'test': []}
        nll = {'train': [], 'valid': [], 'test': []}

        train_time, valid_time, test_time = divide_graph_snapshots(len(batch_graphs), valid_prop, test_prop)

        for i, graph in enumerate(batch_graphs):
            status = get_status(i, train_time, valid_time)
            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta, graph=graph)
            (_, beta_mean, _), (_, phi_mean, _) = self._sample_embeddings(h_phi, h_beta)

            pos_edges, neg_edges = sample_pos_neg_edges(graph)

            p_c_pos_given_z, p_c_pos_gt, _, c_pos = _get_edge_reconstructions(pos_edges, phi_mean, beta_mean)
            p_c_neg_given_z, p_c_neg_gt, _, _ = _get_edge_reconstructions(neg_edges, phi_mean, beta_mean)

            recon_c_softmax = F.log_softmax(p_c_pos_given_z, dim=-1)
            bce = bce_loss(recon_c_softmax, c_pos, reduction='none').detach().cpu().numpy()

            pred[status] = np.hstack([pred[status], p_c_pos_gt.numpy(), p_c_neg_gt.numpy()])
            label[status] = np.hstack([label[status], np.ones(len(p_c_pos_gt)), np.zeros(len(p_c_neg_gt))])
            nll[status] = np.hstack([nll[status], bce])

            # Fix 2
            beta_prior_mean = self.beta_transition(beta_mean)
            phi_prior_mean = self.phi_transition(phi_mean)

        return pred, label, nll

    def predict_embeddings(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        subjects = {}
        for subject in subject_graphs:
            subject_idx, subject_graphs_s, gender_label = subject

            subject_data = self._predict_embeddings(subject_idx, subject_graphs_s, valid_prop, test_prop)
            subjects[subject_idx] = subject_data

        return subjects

    def _predict_embeddings(self, subject_idx, batch_graphs, valid_prop, test_prop):
        train_time, valid_time, _ = divide_graph_snapshots(len(batch_graphs), valid_prop, test_prop)

        embeddings = {
            'alpha_embedding': None,
            'p_c_given_z': {'train': [], 'valid': [], 'test': []},
            'beta_embeddings': {'train': [], 'valid': [], 'test': []},
            'phi_embeddings': {'train': [], 'valid': [], 'test': []}
        }

        alpha_n, _, phi_prior_mean, beta_prior_mean = self._initialize_subject(subject_idx)

        embeddings['alpha_embedding'] = alpha_n.cpu().detach().numpy()

        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        for i, graph in enumerate(batch_graphs):
            status = get_status(i, train_time, valid_time)

            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta, graph=graph)

            (_, beta_mean, _), (_, phi_mean, _) = self._sample_embeddings(h_phi, h_beta)

            p_c_given_z = self.decoder(beta_mean)  # p(c|z) for all communities. No need to sample.
            p_c_given_z = F.softmax(p_c_given_z, dim=-1).cpu().detach().numpy()

            embeddings['p_c_given_z'][status].append(p_c_given_z)
            embeddings['beta_embeddings'][status].append(beta_mean.cpu().detach().numpy())
            embeddings['phi_embeddings'][status].append(phi_mean.cpu().detach().numpy())

            # Fix 2: Updates transition
            beta_prior_mean = self.beta_transition(beta_mean)
            phi_prior_mean = self.phi_transition(phi_mean)

        return embeddings
