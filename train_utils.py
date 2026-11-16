import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *
from data_utils import *
from model import *
from eval.utils import evaluate
from munch import Munch
from model import get_model
from tqdm import tqdm

logger = logging.getLogger("global_logger")



def pretrain_model(
    dataset,
    optimizer_config,
    training_config,
    model = None,
    model_config = None,
    weights = None,
    eval_step = 1,
    save_path = None,
    mode = "train",
    eval_test = True
):
    if model is None:
        assert model_config is not None, "model and model_config cannot both be None!"
        model = get_model(model_config, dataset).to(model_config["device"])

    if dataset.n_classes == 1:
        obj_fun = F.mse_loss
    elif dataset.n_classes == 2:
        obj_fun = F.binary_cross_entropy_with_logits
    elif dataset.n_classes > 2:
        obj_fun = F.cross_entropy
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    if eval_test:
        test_outs = evaluate(model, dataset, target_dataset = "test")
        log_content = f"GNN Before Pretraining: -- Test Loss: {test_outs['loss']:.3f} "
        if dataset.task == "cls":
            log_content +=  f"-- Test ACC: {test_outs['acc']:.3f} -- Test F1-score: {test_outs['f1']:.3f}"
        else:
            log_content +=  f"-- Test Pearson Ratio: {test_outs['pearson_ratio']:.3f}"
        logger.info(log_content)

    assert mode in ["train", "cal", "test"], "Mode must be train, cal or test!"
    data_loader = getattr(dataset, f"{mode}_loader")

    n_epochs = training_config["n_epochs"]
    for epoch in tqdm(range(n_epochs)):
        model.train()

        for i, (batch, idxs) in enumerate(data_loader):
            optimizer.zero_grad()
            batch = batch.to(model.device)
            logits, _ = model(
                data = batch,
                decoder = True,
            )
            logits = logits.view(-1) if dataset.n_classes <= 2 else logits

            if dataset.level == "node":
                logits = logits[batch.train_mask]
                labels = batch.y[batch.train_mask]
            else:
                labels = batch.y

            if weights is None:
                loss = obj_fun(
                    logits, 
                    labels, 
                    reduction="mean"
                )
            else:
                if dataset.level == "node":
                    weights = torch.as_tensor(weights).to(model.device)
                else:
                    weights = torch.as_tensor(weights[idxs]).to(model.device)
                per_sample_loss = obj_fun(
                    logits, 
                    labels, 
                    reduction = "none"
                )
                weighted_loss = per_sample_loss * weights
                loss = weighted_loss.mean()

            loss.backward()
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0 and epoch > 0:
            cal_outs = evaluate(model, dataset, target_dataset = "cal")
            log_content = f"Epoch: {epoch}/{n_epochs} -- Train Loss: {loss:.3f} -- Calibration Loss: {cal_outs['loss']:.3f} "
            if dataset.task == "cls":
                log_content +=  f"-- Calibration ACC: {cal_outs['acc']:.3f} -- Calibration F1: {cal_outs['f1']:.3f}"
            else:
                log_content +=  f"-- Calibration Pearson Ratio: {cal_outs['pearson_ratio']:.3f}"
            logger.info(log_content)

    results = {}

    if eval_test:
        test_outs = evaluate(model, dataset, target_dataset = "test")
        train_outs = evaluate(model, dataset, target_dataset = "train")

        log_content = f"GNN After Pretraining: -- Test Loss: {test_outs['loss']:.3f} -- Train Loss: {train_outs['loss']:.3f} "

        if dataset.task == "cls":
            log_content +=  (
                f"-- Test ACC: {test_outs['acc']:.3f} "
                f"-- Test F1: {test_outs['f1']:.3f} "
                f"-- Test AUROC: {test_outs['auroc']:.3f} "
                f"-- Train ACC: {train_outs['acc']:.3f} "
                f"-- Train F1: {train_outs['f1']:.3f} "
                f"-- Train AUROC: {train_outs['auroc']:.3f} "
            )
            results["pretrained_test_acc"] = test_outs["acc"]
            results["pretrained_test_f1"] = test_outs["f1"]
            results["pretrained_test_auroc"] = test_outs["auroc"]
        else:
            log_content +=  (
                f"-- Test Pearson Ratio: {test_outs['pearson_ratio']:.3f} "
            )
            results["pretrained_test_loss"] = test_outs["loss"]
            results["pretrained_test_pearson_ratio"] = test_outs["pearson_ratio"]

        logger.info(log_content)

    if save_path is not None:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path
        )
        logger.info(f"Model saved to: {save_path}")
    else:
        save_path = "Won't be stored"
    return model, results
