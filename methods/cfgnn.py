import torch, random, os, ipdb, logging
import torch.nn as nn
from copy import deepcopy
from .split_cp import SplitCP
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

            # ipdb.set_trace()
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



class CFGNN(SplitCP):

    def __init__(
        self,
        base_gnn,
        confnn_config,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confmodel = ConfGNN(deepcopy(base_gnn), confnn_config)
        self.confmodel = self.confmodel.to(base_gnn.device)
        self.cp_method = self.cfgnn
        self.conf_outs = None

    
    def cfgnn(
        self,
        all_outs, 
        cal_idxs,
        test_idxs,
        dataset,
        pre_weights = None,
        **kwargs
    ):

        # test_scores, test_x, test_labels = [all_outs[k][test_idxs.numpy()] for k in ["scores", "x", "labels"]]
        cal_scores, cal_x, cal_labels = [all_outs[k][cal_idxs.numpy()] for k in ["scores", "x", "labels"]]
        # n_test = test_scores.shape[0]

        n_cal, n_cls = cal_scores.shape

        # if self.task == "reg":
        #     cal_nc_scores = np.abs(cal_labels - cal_scores).reshape(-1)
        # else:
        #     cal_nc_scores = 1 - cal_scores[np.arange(n_cal), cal_labels]
        #     test_nc_scores = 1 - test_scores

        # if pre_weights is None:
        #     pre_weights = np.ones_like(cal_nc_scores) / n_cal
        # else:
        #     pre_weights = pre_weights / pre_weights.sum()

        # ord_idxs = np.argsort(cal_nc_scores)
        # sorted_nc_scores = cal_nc_scores[ord_idxs]

        # quantile_level = (1 - self.alpha) * (1 + 1 / n_cal)

        # cumulative_probs = np.cumsum(pre_weights[ord_idxs])
        # quantile_idx = np.min(np.where(cumulative_probs >= quantile_level))
        
        # quantile_thresh = sorted_nc_scores[quantile_idx]

        # if self.task == "reg":
        #     test_conf_pred = np.zeros((n_test, n_cls, 2))
        #     test_conf_pred[:, 0] = np.concatenate(
        #         (test_scores - quantile_thresh.reshape(-1, 1), test_scores + quantile_thresh.reshape(-1, 1)),
        #         axis=1
        #     )
        # else:
        #     test_conf_pred = test_nc_scores <= quantile_thresh

        # if args.bnn:
        #     pred = pred.detach().cpu().numpy()
        #     pred_all = pred[:, 0].reshape(-1)
        #     pred_logvar = pred[:, 1].reshape(-1)
        #     pred_std = np.sqrt(np.exp(pred_logvar))
        #     mu = pred_all
        #     pred = np.vstack([mu, mu-1.96 * pred_std,mu+1.96 * pred_std]).T        
        
        # if task == 'regression':
        #     result_this_run['gnn']['CQR'] = run_conformal_regression(pred, data, n, alpha, calib_eval = False)
        # else:
        #     result_this_run['gnn']['APS'] = run_conformal_classification(pred, data, n, alpha, score = 'aps', calib_eval = False)
        #     result_this_run['gnn']['RAPS'] = run_conformal_classification(pred, data, n, alpha, score = 'raps', calib_eval = False)
        
        # condcov_epochs = []
        # if args.optimal_examine:
        #     optimal_examine_res['gnn_pred'] = pred
        #     optimal_examine_res['data'] = data
        # result_this_run['conf_gnn'] = {}
        # if args.bnn:
        #     result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred, data, n, alpha, score = 'qr', calib_eval = False)        
        
        # elif args.conf_correct_model == 'mcdropout':
        #     pred_all = []
        #     model.train()
        #     for i in tqdm(range(1000)):
        #         pred_all.append(model(data.x, data.edge_index).detach().cpu().numpy())
        #     model.eval()
            
        #     pred_all = [i[:,0] for i in pred_all]
        #     pred_mcdropout = np.vstack([np.mean(pred_all, axis = 0), np.quantile(pred_all, q = alpha/2, axis = 0), np.quantile(pred_all, q = 1-alpha/2, axis = 0)]).T
        #     result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred_mcdropout, data, n, alpha, score = 'qr', calib_eval = False)
        # elif args.conf_correct_model == 'mcdropout_std':
        #     pred_all = []
        #     model.train()
        #     for i in tqdm(range(1000)):
        #         pred_all.append(model(data.x, data.edge_index).detach().cpu().numpy())
        #     model.eval()
            
        #     pred_all = [i[:,0] for i in pred_all]
        #     std = np.std(pred_all, axis = 0)
        #     mu = np.mean(pred_all, axis = 0)
        #     pred_mcdropout = np.vstack([mu, 
        #                                 mu - 1.96 * std, 
        #                                 mu + 1.96 * std]).T
        #     result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred_mcdropout, data, n, alpha, score = 'qr', calib_eval = False)

            
        # elif args.conf_correct_model == 'QR':
        #     result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred, data, n, alpha, score = 'qr', calib_eval = False)
        
        # elif args.conf_correct_model == 'Calibrate':
        #     #print('Use calibration model...')
        #     if task == 'regression':
        #         raise ValueError('Unavailable for regression task...')
        #     model_to_correct = copy.deepcopy(model)
        #     if args.calibrator == 'TS':
        #         temp_model = TS(model_to_correct, device)
        #     elif args.calibrator == 'VS':
        #         temp_model = VS(model_to_correct, output_dim, device)
        #     elif args.calibrator == 'ETS':
        #         temp_model = ETS(model_to_correct, output_dim, device)
        #     elif args.calibrator == 'CaGCN':
        #         temp_model = CaGCN(model_to_correct, data.x.shape[0], output_dim, 0.5, device)
        #     elif args.calibrator == 'GATS':
        #         temp_model = GATS(model_to_correct, data.edge_index, 
        #                           data.x.shape[0], torch.tensor(data.train_mask), 
        #                           output_dim, None, 2, 1, device)
        #     else:
        #         raise ValueError
        #     cal_wdecay = 0
        #     temp_model.fit(data, data['valid_mask'], data['train_mask'], cal_wdecay)
        #     with torch.no_grad():
        #         temp_model.eval()
        #         best_pred = temp_model(data.x, data.edge_index)
        #     result_this_run['conf_gnn']['Raw'] = run_conformal_classification(best_pred, data, n, args.alpha, score = 'threshold', calib_eval = False)
        
        # model_to_correct = copy.deepcopy(model)
        # if args.conf_correct_model == 'gnn':
        #     confmodel = ConfGNN(model_to_correct, data, args, num_conf_layers, base_model, output_dim, task).to(args.device)
        # elif args.conf_correct_model == 'mlp':
        #     confmodel = ConfMLP(model_to_correct, data, output_dim, task).to(args.device)
        # pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []

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

            optimizer = torch.optim.Adam(self.confmodel.parameters(), weight_decay=5e-4, lr=confgnn_lr)  # Only perform weight-decay on first convolution.
        
            # rng = np.random.default_rng(seed)
            # cal_range = np.arange(n_cal)
            # rng.shuffle(cal_idxs)

            #     calib_test_idx = np.where(data.calib_test_mask)[0]
            #     np.random.seed(run)
            #     np.random.shuffle(calib_test_idx)
            #     calib_eval_idx = calib_test_idx[:int(n * args.calib_fraction)]
            #     calib_test_real_idx = calib_test_idx[int(n * args.calib_fraction):]

            #     data.calib_eval_mask = np.array([False] * len(y))
            #     data.calib_eval_mask[calib_eval_idx] = True
            #     data.calib_test_real_mask = np.array([False] * len(y))
            #     data.calib_test_real_mask[calib_test_real_idx] = True
            #     if args.verbose:
            #         print('Using a separate calibration holdout...')
            #     calib_eval_idx = np.where(data.calib_eval_mask)[0]
            #     np.random.seed(run)
            #     np.random.shuffle(calib_eval_idx)
            #     train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
            #     train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
            #     train_train_idx = np.where(data.train_mask)[0]

            # if args.conftr_valid_holdout:
            #     if args.verbose:
            #         print('Using the validation set as holdout...')
            #     calib_eval_idx = np.where(data.valid_mask)[0]
            #     np.random.seed(run)
            #     np.random.shuffle(calib_eval_idx)
            #     train_calib_idx = calib_eval_idx[int(len(calib_eval_idx)/2):]
            #     train_test_idx = calib_eval_idx[:int(len(calib_eval_idx)/2)]
            #     train_train_idx = np.where(data.train_mask)[0]

            # if args.conftr_holdout:
            #     train_idx = np.where(data.train_mask)[0]
            #     np.random.seed(run)
            #     np.random.shuffle(train_idx)

            #     train_train_idx = train_idx[:int(len(train_idx)/2)]

            #     if args.conftr_sep_test:
            #         train_calib_test_idx = train_idx[int(len(train_idx)/2):]
            #         np.random.seed(run)
            #         np.random.shuffle(train_calib_test_idx)
            #         train_calib_idx = train_calib_test_idx[int(len(train_calib_test_idx)/2):]
            #         train_test_idx = train_calib_test_idx[:int(len(train_calib_test_idx)/2)]
            #     else:
            #         train_calib_idx = train_idx[int(len(train_idx)/2):]
            #         train_test_idx = train_train_idx
            
            # print('Starting topology-aware conformal correction...')
            # ipdb.set_trace()
            for epoch in range(1, epochs + 1):  
                # if (not args.conftr_holdout) and (not args.conftr_calib_holdout) and (not args.conftr_valid_holdout):
                #     train_idx = np.where(data.train_mask)[0]
                #     np.random.seed(epoch)
                #     np.random.shuffle(train_idx)
                #     train_train_idx = train_idx[:int(len(train_idx)/2)]
                #     train_calib_idx = train_idx[int(len(train_idx)/2):]
                #     train_test_idx = train_train_idx

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

        # cal_outs = dict(
        #     scores = conf_outs[test_cal_idxs].cpu().numpy(),
        #     labels = labels[test_cal_idxs].cpu().long().numpy()
        # )
        # test_outs = dict(
        #     scores = conf_outs[dataset.test_idxs].cpu().numpy(),
        #     labels = labels[dataset.test_idxs].cpu().long().numpy()
        # )

        # ipdb.set_trace()
        all_outs["scores"] = self.conf_outs
        test_predictions, quantile_thresholds = self.split_cp(
            all_outs = all_outs, 
            cal_idxs = test_cal_idxs,
            test_idxs = test_idxs,
            pre_weights = kwargs.get("pre_weights", None),
            **kwargs,
        )

        #     if args.verbose:
        #         log(Epoch = epoch, Prediction_loss = pred_loss.item(), size_loss = size_loss.item())
        #     loss = float(loss)
        #     pred_loss_hist.append(pred_loss.item())
        #     size_loss_hist.append(size_loss.item())
            
        #     (train_acc, val_acc, tmp_test_calib_acc), pred, size_loss = test(self.confmodel, data, alpha, tau, target_size, size_loss = True)
            
        #     if task == 'regression':
        #         eff_valid = run_conformal_regression(pred, data, n, alpha, validation_set = True)[1]
        #     else:
        #         eff_valid = run_conformal_classification(pred, data, n, alpha, score = 'aps', validation_set = True)[1]
                
        #     if args.wandb:
        #         wandb.log({'run_' + str(run) + '_eff_valid': eff_valid})
                
        #     val_size_loss_hist.append(size_loss)
        #     if args.conftr:
        #         if eff_valid < best_size_loss: 
        #             best_size_loss = eff_valid
        #             test_acc = tmp_test_calib_acc
        #             best_pred = pred
        #             best_epoch = epoch
        #     else:
        #         if val_acc > best_val_acc:
        #             best_val_acc = val_acc
        #             test_acc = tmp_test_calib_acc
        #             best_pred = pred    

        # result_this_run['conf_gnn'] = {}
            
        #     if task == 'regression':
        #         result_this_run['conf_gnn']['CQR'] = run_conformal_regression(best_pred, data, n, alpha, calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
        #         result_this_run['conf_gnn']['eff_valid'] = run_conformal_regression(best_pred, data, n, alpha, validation_set = True)[1]
            
        #     else:
        #         result_this_run['conf_gnn']['APS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
        #         result_this_run['conf_gnn']['RAPS'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
        #         result_this_run['conf_gnn']['eff_valid'] = run_conformal_classification(best_pred, data, n, alpha, score = 'aps', validation_set = True)[1]
        #         result_this_run['conf_gnn']['eff_valid_raps'] = run_conformal_classification(best_pred, data, n, alpha, score = 'raps', validation_set = True)[1]
            
        # if args.optimal_examine:
        #     optimal_examine_res['confgnn_pred'] = best_pred
        #     optimal_examine_res['condcov_epochs'] = condcov_epochs
        #     return optimal_examine_res
        # tau2res[run] = result_this_run
        # print('Finished training this run!')

        return test_predictions, quantile_thresholds