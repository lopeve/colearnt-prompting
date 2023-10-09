
import torch
import datasets
import json
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.data import BERTDataModule, LabelModelDataModule
from src.models.BERT import BERT

def get_conf_inds(labels, features, coverage, device='cuda',
                  K=20, scores=None, return_scores=False):
    N = labels.shape[0]
    if scores is None:
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)

        # move to CPU for memory issues on large dset
        pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

        dists_sorted = torch.argsort(pairwise_dists)
        neighbors = dists_sorted[:,:K]
        dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
        weights = 1/(1 + dists_nn)

        neighbors = neighbors.to(device)
        dists_nn = dists_nn.to(device)
        weights = weights.to(device)

        cut_vals = (labels[:,None] != labels[None,:]).long()
        cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
        Jp = (weights * cut_neighbors).sum(dim=1)

        weak_counts = torch.bincount(labels)
        weak_pct = weak_counts / weak_counts.sum()

        prior_probs = weak_pct[labels]
        mu_vals = (1-prior_probs) * weights.sum(dim=1)
        sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
        sigma_vals = torch.sqrt(sigma_vals)
        normalized = (Jp - mu_vals) / sigma_vals
        normalized = normalized.cpu()
    else:
        normalized = scores

    inds_sorted = torch.argsort(normalized)
    N_select = int(coverage * N)
    conf_inds = inds_sorted[:N_select]
    conf_inds = list(set(conf_inds.tolist()))
    if return_scores:
        return conf_inds, normalized
    else:
        return conf_inds


def get_conf_inds_per_class(labels, features, num_per_class, device='cuda',
                            K=20, scores=None, return_scores=False,
                            ref_labels=None, ref_features=None):
    N = labels.shape[0]
    uniq_labels, counts = np.unique(labels.numpy(), return_counts=True)
    if scores is None:
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)