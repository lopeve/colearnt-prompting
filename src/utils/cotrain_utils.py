
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

    conf_inds = []
    all_inds = torch.arange(labels.shape[0])

    for l in uniq_labels:
        l_inds = all_inds[labels == l]
        scores_l = normalized[labels == l]
        sorted_inds_l = l_inds[torch.argsort(scores_l)]
        N_select = min(len(l_inds), num_per_class)
        conf_inds.extend(sorted_inds_l[:N_select].tolist())

    conf_inds = list(set(conf_inds))
    if return_scores:
        return conf_inds, normalized
    else:
        return conf_inds


# select a minimum percentage per class,
# then select the rest from the unstratified ranking.
def get_conf_inds_minppc(labels, features, coverage, min_ppc, device='cuda',
                         K=20, return_scores=False):
    N = labels.shape[0]
    features = torch.FloatTensor(features).to(device)
    uniq_labels, _ = np.unique(labels.numpy(), return_counts=True)
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

    conf_inds = []
    all_inds = torch.arange(labels.shape[0])
    N_select = int(coverage*N)

    # first select the top min_ppc*Nselect for each label.
    for l in uniq_labels:
        l_inds = all_inds[labels == l]
        scores_l = normalized[labels == l]
        sorted_inds_l = l_inds[torch.argsort(scores_l)]
        min_select = min(len(l_inds), int(min_ppc*N_select))
        conf_inds.extend(sorted_inds_l[:min_select].tolist())

    # now select the rest from the pooled (non-stratified) ranking
    remaining_inds = list(set(all_inds).difference(set(conf_inds)))
    remaining_inds = torch.tensor(remaining_inds)

    scores = normalized[remaining_inds]
    inds_sorted = torch.argsort(remaining_inds)
    global_inds_sorted = remaining_inds[inds_sorted]
    num_select = N_select - len(conf_inds)
    conf_inds.extend(global_inds_sorted[:num_select])
    conf_inds = list(set(conf_inds))
    return conf_inds



add_special_tokens=True
def extract_psl_and_features(example, prompttokenizer=None, promptmodel=None, list_templates=None, template_idx=None, device='cuda'):
    promptmodel.eval()
    with torch.no_grad():
        if template_idx is not None:
            template = list_templates[template_idx]
        else:
            template = np.random.choice(list_templates)

        input_str, target_str = template.apply(example)
        answer_choices = template.get_answer_choices_list(example)
        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    prompttokenizer(
                        input_field, return_tensors="pt", truncation=True, add_special_tokens=False
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    prompttokenizer(
                        input_str[-1], return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            tok_outputs = prompttokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
            )
            input_ids = tok_outputs.input_ids.squeeze(0)
            tok_outputs = {k:v.to('cuda') for (k,v) in tok_outputs.items()}

        target_ids = prompttokenizer(
            target_str, return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens