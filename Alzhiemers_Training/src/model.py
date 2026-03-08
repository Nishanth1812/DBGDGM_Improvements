import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

from .utils import (
    bce_loss,
    divide_graph_snapshots,
    get_status,
    gumbel_softmax,
    kld_gauss,
    kld_z_loss,
    reparameterized_sample,
    sample_pos_neg_edges,
    snapshot_edge_tensor,
)

LOSS_KEYS = ['nll', 'kld_z', 'kld_alpha', 'kld_beta', 'kld_phi', 'classification']


class Model(nn.Module):
    """
    DBGDGM model definition.

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
                 num_classes=4,
                 gamma=0.1,
                 sigma=1.,
                 device=torch.device("cpu")):

        super().__init__()

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.sigma = sigma
        self.device = device

        self.beta_mean = nn.Linear(self.embedding_dim, embedding_dim)
        self.beta_std = nn.Sequential(nn.Linear(self.embedding_dim, embedding_dim), nn.Softplus())
        self.phi_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.phi_std = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Softplus())
        self.nn_pi = nn.Linear(self.embedding_dim, self.categorical_dim, bias=False)

        self.rnn_nodes = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)
        self.rnn_comms = nn.GRU(2 * self.embedding_dim, self.embedding_dim, num_layers=1, bias=True)

        self.alpha_mean = nn.Embedding(self.num_samples, self.embedding_dim)
        self.alpha_std = nn.Embedding(self.num_samples, self.embedding_dim)

        self.subject_to_phi = nn.Linear(self.embedding_dim, self.num_nodes * self.embedding_dim)
        self.subject_to_beta = nn.Linear(self.embedding_dim, self.categorical_dim * self.embedding_dim)

        self.alpha_mean_prior = torch.zeros(self.embedding_dim)
        self.alpha_std_scalar = 1.

        self.decoder = nn.Sequential(nn.Linear(embedding_dim, num_nodes, bias=False))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def _update_hidden_states(self, phi_prior_mean, beta_prior_mean, h_phi, h_beta):
        nodes_in = torch.cat([phi_prior_mean, phi_prior_mean], dim=-1).view(1, self.num_nodes, 2 * self.embedding_dim)
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

    def _subject_representation(self, alpha_n, h_phi, h_beta):
        return alpha_n + h_phi[-1].mean(dim=0) + h_beta[-1].mean(dim=0)

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

    def forward(self, batch_data, valid_prop=0.1, test_prop=0.1, temp=1., class_weights=None):
        loss = {key: 0 for key in LOSS_KEYS}
        for data in batch_data:
            subject_idx, dynamic_graph, subject_label = data
            subject_loss = self._forward(subject_idx, dynamic_graph, subject_label, valid_prop, test_prop, temp,
                                         class_weights=class_weights)
            for loss_name in LOSS_KEYS:
                loss[loss_name] += subject_loss[loss_name]
        return loss

    def _forward(self, subject_idx, batch_graphs, subject_label, valid_prop, test_prop, temp, class_weights=None):
        loss = {key: 0 for key in LOSS_KEYS}
        edge_counter = 0

        train_time, valid_time, test_time = divide_graph_snapshots(len(batch_graphs), valid_prop, test_prop)
        alpha_n, kld_alpha, phi_prior_mean, beta_prior_mean = self._initialize_subject(subject_idx)

        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        for snapshot_idx in range(0, train_time):
            graph = batch_graphs[snapshot_idx]
            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta)

            batch = snapshot_edge_tensor(graph, device=self.device)
            if batch.numel() == 0:
                continue

            if self.training and batch.shape[0] > 1:
                permutation = torch.randperm(batch.shape[0], device=self.device)
                batch = batch[permutation]

            w = torch.cat((batch[:, 0], batch[:, 1]))
            c = torch.cat((batch[:, 1], batch[:, 0]))

            (beta_sample, beta_mean_t, beta_std_t), (phi_sample, phi_mean_t, phi_std_t) = self._sample_embeddings(h_phi, h_beta)

            recon, posterior_z, prior_z = self._edge_reconstruction(w, c, phi_sample, beta_sample, temp)

            loss['nll'] += bce_loss(recon, c)
            loss['kld_z'] += kld_z_loss(posterior_z, prior_z)
            loss['kld_alpha'] += kld_alpha
            loss['kld_beta'] += kld_gauss(beta_sample, beta_std_t, beta_prior_mean, self.gamma)
            loss['kld_phi'] += kld_gauss(phi_sample, phi_std_t, phi_prior_mean, self.sigma)

            beta_prior_mean = beta_sample
            phi_prior_mean = phi_sample
            edge_counter += c.shape[0]

        if edge_counter == 0:
            raise ValueError(f"Subject {subject_idx} has no positive edges in the training snapshots.")

        for loss_name in ['nll', 'kld_z', 'kld_alpha', 'kld_beta', 'kld_phi']:
            loss[loss_name] = loss[loss_name] / edge_counter

        subject_repr = self._subject_representation(alpha_n, h_phi, h_beta)
        logits = self.classifier(subject_repr.unsqueeze(0))
        target = torch.tensor([subject_label], dtype=torch.long, device=self.device)
        weight = class_weights.to(self.device) if class_weights is not None else None
        loss['classification'] = F.cross_entropy(logits, target, weight=weight)

        return loss

    def _encode_subject(self, subject_idx, batch_graphs, valid_prop, test_prop, use_train_snapshots_only=True):
        alpha_n, _, phi_prior_mean, beta_prior_mean = self._initialize_subject(subject_idx)

        h_beta = torch.zeros(1, self.categorical_dim, self.embedding_dim).to(self.device)
        h_phi = torch.zeros(1, self.num_nodes, self.embedding_dim).to(self.device)

        train_time, _, _ = divide_graph_snapshots(len(batch_graphs), valid_prop, test_prop)
        total_steps = train_time if use_train_snapshots_only else len(batch_graphs)

        for snapshot_idx in range(total_steps):
            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta)
            (_, beta_mean, _), (_, phi_mean, _) = self._sample_embeddings(h_phi, h_beta)
            beta_prior_mean = beta_mean
            phi_prior_mean = phi_mean

        return self._subject_representation(alpha_n, h_phi, h_beta)

    def predict_auc_roc_precision(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        aucroc = {'train': 0, 'valid': 0, 'test': 0}
        ap = {'train': 0, 'valid': 0, 'test': 0}
        nll = {'train': 0, 'valid': 0, 'test': 0}

        num_subjects = len(subject_graphs)
        for subject in subject_graphs:
            subject_idx, subject_graphs_s, _ = subject
            pred, label, _nll = self._predict_auc_roc_precision(subject_idx,
                                                                subject_graphs_s,
                                                                valid_prop,
                                                                test_prop)

            for status in ['train', 'valid', 'test']:
                aucroc[status] += roc_auc_score(label[status], pred[status]) / num_subjects
                ap[status] += average_precision_score(label[status], pred[status]) / num_subjects
                nll[status] += _nll[status].mean() / num_subjects

        return nll, aucroc, ap

    def predict_label_metrics(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        labels = []
        predictions = []
        probabilities = []
        total_loss = 0.0

        for subject_idx, subject_graphs_s, subject_label in subject_graphs:
            subject_repr = self._encode_subject(subject_idx, subject_graphs_s, valid_prop, test_prop)
            logits = self.classifier(subject_repr.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            target = torch.tensor([subject_label], dtype=torch.long, device=self.device)

            total_loss += F.cross_entropy(logits, target, reduction='sum').item()
            labels.append(int(subject_label))
            predictions.append(int(torch.argmax(probs).item()))
            probabilities.append(probs.detach().cpu().numpy())

        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)

        metrics = {
            'loss': total_loss / max(len(subject_graphs), 1),
            'accuracy': float((predictions == labels).mean()) if len(labels) else float('nan'),
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)) if len(labels) else float('nan'),
            'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)) if len(labels) else float('nan'),
            'macro_auc_ovr': float('nan'),
        }

        unique_labels = np.unique(labels)
        if len(unique_labels) == self.num_classes:
            metrics['macro_auc_ovr'] = float(
                roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
            )

        return metrics

    def _predict_auc_roc_precision(self, subject_idx, batch_graphs, valid_prop, test_prop):
        def _get_edge_reconstructions(edges, phi_sample, beta_sample):
            if not edges:
                return None, None, None, None
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
            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta)
            (_, beta_mean, _), (_, phi_mean, _) = self._sample_embeddings(h_phi, h_beta)

            pos_edges, neg_edges = sample_pos_neg_edges(graph)
            if not pos_edges or not neg_edges:
                continue

            p_c_pos_given_z, p_c_pos_gt, _, c_pos = _get_edge_reconstructions(pos_edges, phi_mean, beta_mean)
            p_c_neg_given_z, p_c_neg_gt, _, _ = _get_edge_reconstructions(neg_edges, phi_mean, beta_mean)

            bce = bce_loss(p_c_pos_given_z, c_pos, reduction='none').detach().cpu().numpy()

            pred[status] = np.hstack([pred[status], p_c_pos_gt.numpy(), p_c_neg_gt.numpy()])
            label[status] = np.hstack([label[status], np.ones(len(p_c_pos_gt)), np.zeros(len(p_c_neg_gt))])
            nll[status] = np.hstack([nll[status], bce])

            beta_prior_mean = beta_mean
            phi_prior_mean = phi_mean

        return pred, label, nll

    def predict_embeddings(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        subjects = {}
        for subject in subject_graphs:
            subject_idx, subject_graphs_s, _ = subject

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

            h_phi, h_beta = self._update_hidden_states(phi_prior_mean, beta_prior_mean, h_phi, h_beta)

            (_, beta_mean, _), (_, phi_mean, _) = self._sample_embeddings(h_phi, h_beta)

            p_c_given_z = self.decoder(beta_mean)  # p(c|z) for all communities. No need to sample.
            p_c_given_z = F.softmax(p_c_given_z, dim=-1).cpu().detach().numpy()

            embeddings['p_c_given_z'][status].append(p_c_given_z)
            embeddings['beta_embeddings'][status].append(beta_mean.cpu().detach().numpy())
            embeddings['phi_embeddings'][status].append(phi_mean.cpu().detach().numpy())

            beta_prior_mean = beta_mean
            phi_prior_mean = phi_mean

        return embeddings
