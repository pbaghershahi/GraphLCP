import torch, random, os, logging, shutil
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_dense_adj, subgraph, remove_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import F1Score, AUROC
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import ipdb


@torch.no_grad()
def evaluate(model, dataset, target_dataset = "cal", metric_eval = True):

    if target_dataset == "train":
        data_loader = dataset.train_loader
        n_samples = dataset.n_train
        target_mask = dataset.train_mask_name
    elif target_dataset == "test":
        data_loader = dataset.test_loader
        n_samples = dataset.n_test
        target_mask = dataset.test_mask_name
    elif target_dataset == "cal":
        data_loader = dataset.cal_loader
        n_samples = dataset.n_cal
        target_mask = dataset.cal_mask_name
    else:
        data_loader = dataset.cal_loader
        n_samples = dataset.num_nsamples
        target_mask = "all"

        
    outs = {}
    logits = []
    labels = []
    mask = []
    x = []
    
    model.eval()
    for i, (batch, idxs) in enumerate(data_loader):
        batch = batch.to(model.device)

        temp_labels = batch.y.to(model.device)
        temp_logits, temp_x = model(
            data = batch,
            decoder = True,
        )

        x.append(temp_x.detach().cpu())
        logits.append(temp_logits.detach().cpu())
        labels.append(temp_labels.cpu())
        if hasattr(batch, target_mask):
            mask.append(getattr(batch, target_mask).cpu())
        else:
            mask.append(torch.ones(batch.x.shape[0], dtype=torch.bool))

    x = torch.cat(x, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels)
    n_cls = 1 if logits.ndim == 1 else logits.shape[-1]

    if dataset.level == "node":
        mask = torch.cat(mask)
        x = x[mask]
        logits = logits[mask]
        labels = labels[mask]

    n_samples = labels.shape[0]

    if dataset.task == "reg":
        scores = logits.view(n_samples, n_cls)
        loss_func = F.mse_loss
        labels = labels.view(n_samples, n_cls)
        pearson_ratio = pearsonr(scores.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy())
        outs["pearson_ratio"] = pearson_ratio.statistic
        logits = scores
    else:
        binary_task = False if dataset.n_classes > 2 else True
        f1 = F1Score(
            task = "binary" if binary_task else "multiclass",
            num_classes = dataset.n_classes, average = "micro"
        )
        auroc = AUROC(
            task = "binary" if binary_task else "multiclass",
            num_classes = dataset.n_classes, average = "weighted"
        )
        loss_func = F.binary_cross_entropy_with_logits if binary_task else F.cross_entropy

        if binary_task:
            logits = logits.view(-1)
            scores = F.sigmoid(logits)
            preds = (scores >= 0.5).long()
        else:
            scores = F.softmax(logits, dim=1)
            preds = scores.argmax(dim=1).long()
        
        outs["acc"] = int((labels == preds).sum()) / n_samples
        outs["f1"] = f1(preds, labels).item()
        outs["auroc"] = auroc(scores, labels).item()

        if binary_task:
            scores = scores.view(-1, 1)
            scores = torch.cat(((1. - scores).abs(), scores), dim=1)
            logits = logits.view(-1, 1)
        
    outs["x"] = x
    outs["labels"] = labels
    outs["loss"] = loss_func(logits, labels, reduction="mean")
    outs["scores"] = scores
    outs["logits"] = logits

    return outs



def compute_scores(
    self, 
    model,
    data_loader,
    n_classes,
):
    outs = {}
    scores = []
    labels = []
    x = []
    with torch.no_grad():
        for i, (batch, idxs) in enumerate(data_loader):
            logits, temp_x = model(
                data = batch,
                decoder = True,
            )
            
            scores.append(logits.detach().to("cpu"))
            labels.append(batch.y.to("cpu"))
            x.append(temp_x.detach().to("cpu"))

    scores = torch.cat(scores, dim=0)
    if n_classes == 1:
        scores = scores.view(-1)
    elif n_classes == 2:
        scores = F.sigmoid(scores.view(-1, 1))
        scores = torch.cat(((1. - scores).abs(), scores), dim=1)
    elif n_classes > 2:
        scores = F.softmax(scores, dim=1)
    else:
        raise Exception("Invalid number of classes!")
            
    outs["scores"] = scores.numpy()
    outs["x"] = torch.cat(x, dim=0).numpy()
    outs["labels"] = torch.cat(labels).int().numpy()

    return outs 