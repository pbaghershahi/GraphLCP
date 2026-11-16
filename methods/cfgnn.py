import torch, random, os, ipdb, logging
import torch.nn as nn
from copy import deepcopy
from .split_cp import SplitCP
from .aps import APS
from model import GNN
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np

logger = logging.getLogger("global_logger")



class ConfGNN(nn.Module):
    def __init__(
        self, 
        base_gnn,
        confnn_config,
        **kwargs
    ):
        super().__init__()
        self.base_gnn = base_gnn
        self.model = GNN(**confnn_config)


    def forward(
        self, data_loader, base_out = None,
    ):
        logits = []
        labels = []
        base_outs = []
        
        self.base_gnn.eval()
        for i, (batch, idxs) in enumerate(data_loader):
            batch = batch.to(self.base_gnn.device)
            temp_labels = batch.y.to(self.base_gnn.device)

            if base_out is None:
                with torch.no_grad():
                    base_out, _ = self.base_gnn(
                        data = Data(
                            x = batch.x,
                            edge_index = batch.edge_index,
                            batch = batch.batch,
                        ),
                        decoder = True,
                    )
            else:
                base_out = base_out.to(self.base_gnn.device)

            temp_logits, _ = self.model(
                x = base_out,
                edge_index = batch.edge_index,
                batch = batch.batch,
                decoder = True,
            )

            logits.append(temp_logits)
            labels.append(temp_labels)
            base_outs.append(base_out)

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels)
        base_outs = torch.cat(base_outs, dim=0)

        return logits, labels, base_outs



class CFGNN(SplitCP, APS):

    def __init__(
        self,
        base_gnn,
        confnn_config,
        base_method = "split",
        **kwargs
    ):
        SplitCP.__init__(self, **kwargs)
        self.confmodel = ConfGNN(deepcopy(base_gnn), confnn_config)
        self.confmodel = self.confmodel.to(base_gnn.device)
        self.cp_method = self.cfgnn
        self.conf_outs = None
        self.base_method = base_method
        print(f"Initialized CFGNN with base_method={base_method}")
        if base_method == "aps":
            assert self.task == "cls", "APS-based weighting only supports classification tasks."


    def cfgnn(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights = None,
        **kwargs
    ):

        cal_scores, cal_x, cal_labels = [all_outs[k][cal_idxs.numpy()] for k in ["scores", "x", "labels"]]
        n_cal, n_cls = cal_scores.shape

        calib_fraction = 0.5
        n_cal_train = int(n_cal * calib_fraction)
        train_calib_idx = cal_idxs[:n_cal_train//2]
        train_test_idx = cal_idxs[n_cal_train//2:n_cal_train]
        train_idxs = dataset.train_idxs
        test_cal_idxs = cal_idxs[n_cal_train:]

        if self.conf_outs is None:

            best_size_loss = 10000
            best_val_acc = 0
            tau = 0.1
            target_size = 0.0
            seed = 1234
            epochs = 500
            size_loss_weight = 0.1
            reg_loss_weight = 1.0
            confgnn_lr = 0.0001
            conftr_calib_holdout = True
            conftr = True
            quantile = False

            optimizer = torch.optim.Adam(self.confmodel.parameters(), weight_decay=5e-4, lr=confgnn_lr)

            for epoch in range(1, epochs + 1):

                self.confmodel.train()
                optimizer.zero_grad()
                conf_outs, labels, base_outs = self.confmodel(
                    dataset.cal_loader, 
                    base_out = torch.from_numpy(all_outs['scores'])
                )

                if self.task == 'reg':
                    if quantile:
                        ### use only train_train nodes
                        mid = conf_outs[:, 0][train_idxs].reshape(-1,1)
                        label = labels[train_idxs].reshape(-1,1)
                        mse_loss = F.mse_loss(mid, label)
                        low_bound = alpha / 2
                        upp_bound = 1 - alpha / 2
                        lower = conf_outs[:, 1][train_idxs].reshape(-1,1)
                        upper = conf_outs[:, 2][train_idxs].reshape(-1,1)
                        low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
                        upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
                        pred_loss = mse_loss + low_loss + upp_loss

                        n_temp = len(train_calib_idx)
                        ## CQR loss
                        lower_calib = conf_outs[:, 1][train_calib_idx].reshape(-1,1)
                        upper_calib = conf_outs[:, 2][train_calib_idx].reshape(-1,1)
                        label_calib = labels[train_calib_idx].reshape(-1,1)

                        cal_scores = torch.maximum(label_calib-upper_calib, lower_calib-label_calib)
                        # Get the score quantile
                        qhat = torch.quantile(cal_scores, np.ceil((n_temp + 1) * (1 - alpha)) / n_temp, interpolation='higher')

                        lower_test = conf_outs[:, 1][train_test_idx].reshape(-1,1)
                        upper_test = conf_outs[:, 2][train_test_idx].reshape(-1,1)
                        
                        lower_deviate_loss = F.mse_loss(conf_outs[:, 1].reshape(-1,1), base_outs[:, 1].reshape(-1,1))
                        upper_deviate_loss = F.mse_loss(conf_outs[:, 2].reshape(-1,1), base_outs[:, 2].reshape(-1,1))
                                                
                        size_loss = torch.mean(upper_test + qhat - (lower_test - qhat))
                                        
                        if epoch <= 1000:
                            loss = pred_loss
                        else:
                            loss = pred_loss + size_loss_weight * size_loss
                            loss += reg_loss_weight * lower_deviate_loss
                            loss += reg_loss_weight * upper_deviate_loss

                    else:

                        conf_outs = conf_outs.squeeze()
                        base_outs = base_outs.squeeze()

                        cal_residuals = torch.abs(labels[train_calib_idx] - conf_outs[train_calib_idx])

                        n_temp = len(train_calib_idx)
                        q_level = np.ceil((n_temp + 1) * (1 - self.alpha)) / n_temp
                        qhat = torch.quantile(cal_residuals, q_level, interpolation='higher')
                        size_loss = qhat

                        deviate_loss = F.mse_loss(conf_outs.reshape(-1,1), base_outs.reshape(-1,1))
                        pred_loss = F.mse_loss(conf_outs[train_idxs], labels[train_idxs])
                        
                        if epoch <= 1000:
                            loss = pred_loss
                        else:
                            loss = pred_loss + size_loss_weight * size_loss
                            loss += reg_loss_weight * deviate_loss

                else:
                    conf_softmax = F.softmax(conf_outs, dim = 1)
                    base_softmax = F.softmax(base_outs, dim = 1)

                    n_cal_temp = train_calib_idx.shape[0]
                    cal_temp_labels = labels[train_calib_idx]
                    q_level = np.ceil((n_cal_temp + 1) * (1 - self.alpha)) / n_cal_temp

                    tps_conformal_score = conf_softmax[train_calib_idx][torch.arange(n_cal_temp), cal_temp_labels]
                    qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')

                    c = torch.sigmoid((conf_softmax[train_test_idx] - qhat)/tau)
                    size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))

                    pred_loss = F.cross_entropy(conf_outs[train_idxs], labels[train_idxs])

                    if epoch <= 1000:
                        loss = pred_loss
                    elif cond_cov_loss:
                        if epoch <=3000:
                            loss = pred_loss + size_loss_weight * size_loss
                        else:
                            loss = pred_loss + size_loss_weight * size_loss + loss_cov
                    else:
                        loss = pred_loss + size_loss_weight * size_loss
                    
                loss.backward()
                optimizer.step()
            
            self.confmodel.eval()
            with torch.no_grad():
                conf_outs, labels, _ = self.confmodel(
                    dataset.cal_loader, 
                    base_out = torch.from_numpy(all_outs['scores'])
                )
                if self.task == 'cls':
                    conf_outs = F.softmax(conf_outs, dim = 1)

            self.conf_outs = conf_outs.cpu().numpy()

        all_outs["scores"] = self.conf_outs

        if self.base_method == "split":
            cp_out_dict = self.split_cp(
                all_outs = all_outs, 
                cal_idxs = test_cal_idxs,
                test_idxs = test_idxs,
                pre_weights = kwargs.get("pre_weights", None),
                **kwargs,
            )
        elif self.base_method == "aps":
            cp_out_dict = self.aps(
                all_outs = all_outs, 
                cal_idxs = test_cal_idxs,
                test_idxs = test_idxs,
                pre_weights = kwargs.get("pre_weights", None),
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown base_method: {self.base_method}")

        outputs = {
            "test_predictions": cp_out_dict["test_predictions"],
            "quantile_thresholds": cp_out_dict["quantile_thresholds"],
        }
        
        return outputs