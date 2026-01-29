import os
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[0])
sys.path.insert(0, ROOT_PATH)

from train_utils import *
from utils import *
from kmm import kernel_mean_matching_parallel
from methods import SplitCP, RandomlyLocalizedCP, CalibratedLocalizedCP, GraphRLCP, TestCP, CFGNN
from eval.utils import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_utils import *
import yaml
import json
import seaborn as sns
import argparse
from scipy import stats
from model import get_model
from iw_utils import get_weights, mmd
from munch import Munch
import ipdb
import warnings
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data`"
)



def get_parser():
    parser = argparse.ArgumentParser(description="CertainGNN")

    # GNN Model Args
    parser.add_argument("-gwb", "--gnn_with_bn", action="store_true")
    parser.add_argument("-gwn", "--gnn_with_nonlinearity", action="store_true")
    parser.add_argument("-gwid", "--gnn_with_input_dropout", action="store_true")
    parser.add_argument("-gwd", "--gnn_weight_decay", type=float, help="Rate of regularization")
    parser.add_argument("-gt", "--gnn_type", type=str, help="Type of base GNN: [gcn, gat, gin, sage]")
    parser.add_argument("-gpm", "--gnn_pooling_mode", type=str, help="Type of base GNN: [gcn, gat, gin, sage]")
    parser.add_argument("-gnhl", "--gnn_num_hid_layers", type=int, help="Number of layers of the base GNN")
    parser.add_argument("-gne", "--gnn_n_epochs", type=int, help="Number of epochs for pretraining")
    parser.add_argument("-ges", "--gnn_eval_step", type=int, help="Evaluation step for pretrained model")
    parser.add_argument("-ghd", "--gnn_h_dim", type=int, help="Hidden dim of the GNN")
    parser.add_argument("-glr", "--gnn_lr", type=float, help="Learning rate for pretraining")
    parser.add_argument("-gsss", "--gnn_scheduler_step_size", type=int, help="Learning rate step size for pretraining")
    parser.add_argument("-gsg", "--gnn_scheduler_gamma", type=float, help="Learning rate gamma for pretraining")
    parser.add_argument("-gbs", "--gnn_batch_size", type=int, help="Batch size for pretraining")
    parser.add_argument("-gdr", "--gnn_dropout_rate", type=float, help="Dropout for GNN")

    # Experiment / CP Args
    parser.add_argument("-tt", "--total_iters", type=int, help="Total number of trials with random initialization of datasets")
    parser.add_argument("-m", "--method", type=str, help="CP method: [split_cp, rand_lcp, cal_lcp]")
    parser.add_argument("-ds", "--dataset", type=str, help="Target dataset")
    parser.add_argument("-dsd", "--dataset_domain", type=str)
    parser.add_argument("-dss", "--dataset_shift", type=str)
    parser.add_argument("-dsg", "--dataset_generate", action="store_true")
    parser.add_argument("-dcd", "--dataset_cal_domain", type=str)
    parser.add_argument("-dtd", "--dataset_test_domain", type=str)
    parser.add_argument("-tlt", "--train_loader_type", type=str)
    parser.add_argument("-dnp", "--dataset_n_partitions", type=int)
    parser.add_argument("-dnm", "--data_normal_mode", type=str, help="Config file to save to")
    parser.add_argument("-pgl", "--partitioning_granularity_level", type=float)
    parser.add_argument("-nsm", "--not_save_model", action="store_true")
    parser.add_argument("-pp", "--pretrained_path", nargs="*", default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument("-cpbs", "--cp_batch_size", type=int, help="Batch size for pretraining")
    parser.add_argument("-nr", "--num_runs", type=int, help="Numer of runs per trial")
    parser.add_argument("-wnv", "--wsc_num_vectors", type=int, help="Number of vectors for wsc optimization")
    parser.add_argument("-tts", "--train_test_split", nargs="*", type=float, help="[train_percentage, test_percentage]")
    parser.add_argument("-s", "--seed", nargs="*", type=int, default=None, help="Seed for random")
    
    # Config / Logging Args
    parser.add_argument("-cff", "--config_from_file", type=str, default="", help="Config file to read from")
    parser.add_argument("-ctf", "--config_to_file", type=str, default="", help="Config file to save to")
    parser.add_argument("-nv", "--not_verbose", action="store_true")
    parser.add_argument("-lpd", "--log_path_dir", default=None, type=str)
    parser.add_argument("-opd", "--output_path_dir", default=None, type=str)
    parser.add_argument("-mpd", "--model_path_dir", default=None, type=str)
    parser.add_argument("-eid", "--experiment_id", type=str, help="ID to group outputs of a specific run script")

    # Weighting / PCA / Advanced Args
    parser.add_argument("-iw", "--importance_weighting", type=str)
    parser.add_argument("-wss", "--with_standard_scalar", action="store_true")
    parser.add_argument("-wdb", "--with_dynamic_bandwidth", action="store_true")
    parser.add_argument("-sos", "--sample_original_space", action="store_true")
    parser.add_argument("-itt", "--input_transformation_type", type=str)
    parser.add_argument("-ltt", "--latent_transformation_type", type=str)
    parser.add_argument("-iwbs", "--importance_weighting_batch_size", type=int)
    parser.add_argument("-iwrf", "--importance_weighting_reg_fac", type=float)
    parser.add_argument("-iwmi", "--importance_weighting_max_iters", type=int)
    parser.add_argument("-iwkq", "--importance_weighting_kernelwidth_quantile", type=float)
    parser.add_argument("-iwkb", "--importance_weighting_kmm_b", type=float)
    parser.add_argument("-ipnc", "--input_pca_n_components", type=int)
    parser.add_argument("-lpnc", "--latent_pca_n_components", type=int)
    parser.add_argument("-a", "--alpha", type=float, help="Alpha value")
    parser.add_argument("-b", "--bandwidth", type=float, help="Bandwidth value")
    parser.add_argument("-ecp", "--eval_cp", action="store_true")
    parser.add_argument("-vd", "--visualize_data", action="store_true")
    parser.add_argument("-kt", "--kernel_type", type=str, help="[gaussian, ppr]")
    parser.add_argument("-pa", "--ppr_alpha", type=float)
    parser.add_argument("-pk", "--ppr_k", type=int)
    parser.add_argument("-cdc", "--convex_dist_coeff", type=float)
    parser.add_argument("-ccm", "--convex_coeff_method", type=str)
    parser.add_argument("-gdt", "--graph_densification_threshold", type=float)
    parser.add_argument("-gdm", "--graph_densification_method", type=str)
    parser.add_argument("-adt", "--adapt_dense_threshold", action="store_true")
    parser.add_argument("-dmer", "--dense_max_edge_ratio", default=None)
    parser.add_argument("-cem", "--conformal_eval_method", nargs="*", type=str)
    parser.add_argument("-wm", "--weighting_method", type=str, help="Weighting method for weighted CP")
    
    return parser



def process_config(args, parser):
    """Merges args from file and CLI."""
    if args.config_from_file:
        # Mapping flags to their destination variable names
        flag_to_dest_map = {}
        for action in parser._actions:
            for option_string in action.option_strings:
                flag_to_dest_map[option_string] = action.dest

        input_dests = set()
        # Simple heuristic to detect which flags were actually passed on CLI
        # This prevents the config file from overriding specific CLI flags
        for arg in sys.argv[1:]:
            if arg in flag_to_dest_map:
                input_dests.add(flag_to_dest_map[arg])

        with open(args.config_from_file, 'r') as infile:
            all_args = vars(args)
            file_args = yaml.safe_load(infile) or {}

        merged_args = {
            key: file_args[key] if (key in file_args and key not in input_dests) else value
            for key, value in all_args.items()
        }
        
        merged_args.update({k: v for k, v in file_args.items() if k not in merged_args})
        args = argparse.Namespace(**merged_args)
    return args

    

def main(parser) -> None:
    
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    
    configs_path = os.path.join(ROOT_PATH, "config")
    os.makedirs(configs_path, exist_ok=True)    

    if args.log_path_dir is None:
        log_dir = Path(ROOT_PATH) / "log/"
    else:
        log_dir = Path(args.log_path_dir) / "CertainGNN/log"
    
    if args.experiment_id is not None:
        if args.output_path_dir is None:
            ouputs_path = Path(ROOT_PATH) / "outputs" / args.experiment_id
        else:
            ouputs_path = Path(args.output_path_dir) / "CertainGNN" / "outputs" / args.experiment_id
    else:
        if args.output_path_dir is None:
            ouputs_path = Path(ROOT_PATH) / "outputs"
        else:
            ouputs_path = Path(args.output_path_dir) / "CertainGNN" / "outputs"

    if args.model_path_dir is None:
        models_path = Path(ROOT_PATH) / "pretrained/"
    else:
        models_path = Path(args.model_path_dir) / "CertainGNN/pretrained"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(models_path / "predictive", exist_ok=True)
    os.makedirs(models_path / "weighting", exist_ok=True)
    os.makedirs(ouputs_path, exist_ok=True)
    weights_path = ouputs_path / "weights" / args.importance_weighting
    os.makedirs(weights_path, exist_ok=True)
        
    if args.experiment_id:
        json_out_path = ouputs_path / f"{args.method}_{args.dataset}_{exec_name}.json"
    else:
        os.makedirs(ouputs_path / "json_outs", exist_ok=True)
        json_out_path = ouputs_path / "json_outs" / f"{args.method}_{args.dataset}_{exec_name}.json"

    if args.not_verbose:
        global_logger = DummyLogger()

    else:
        general_log_dir = Path(log_dir) / "general"
        evaluations_log_dir = Path(log_dir) / "evaluation"
        os.makedirs(general_log_dir, exist_ok=True)
        os.makedirs(evaluations_log_dir, exist_ok=True)

        log_file_paths = [
            general_log_dir / f"global_{args.method}_{args.dataset}_{exec_name}.log", 
            evaluations_log_dir / f"eval_{args.method}_{args.dataset}_{exec_name}.log"
        ]
        global_logger = setup_logger(
            name="global_logger", 
            level=logging.INFO, 
            log_file=log_file_paths[0],
            stream_handler=True,
        )
        # eval_logger = setup_logger(
        #     name="eval_logger", 
        #     level=logging.INFO, 
        #     log_file=log_file_paths[1],
        #     stream_handler=False,
        #     formatter=logging.Formatter(fmt='%(message)s')
        # )
        global_logger.info(f"Logging to: {log_file_paths}\nOutputs to: {ouputs_path}")

    if args.config_from_file != "":
        global_logger.info(f"Reading config from: {args.config_from_file}")
    
    if args.seed is not None:
        arg_seeds = np.array(args.seed)
        total_iters = len(arg_seeds)
    else:
        total_iters = args.total_iters
        arg_seeds = np.random.randint(1000, 5000, (total_iters,))

    # global_logger.info(args)
    # global_logger.info(f"Seeds: {arg_seeds.tolist()}")
    # global_logger.info("#"*100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_paths = args.pretrained_path if len(args.pretrained_path) > 0 else []
    all_results = {}

    def update_results(temp_results):
        for key, value in temp_results.items():
            if key in all_results:
                all_results[key].append(value)
            else:
                all_results[key] = [value]

    for i in range(total_iters):

        exec_trial_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        fix_seed(arg_seeds[i])

        gen_ds = GenDataset()
        if args.dataset in ["ENZYMES", "PROTEINS", "DHFR"]:
            dataset, embeds_dists = gen_ds.get_graph_dataset(
                args.dataset,
                store_to_path = "./data",
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.data_normal_mode,
                data_config = dict(
                    node_attributes = True,
                ),
                n_partitions = args.dataset_n_partitions, 
                partitioning_ds = "test", 
                granularity_level = args.partitioning_granularity_level,
                seed = arg_seeds[i],
            )
        elif args.dataset in ["GOODHIV", "GOODCMNIST", "GOODMotif", "GOODZINC"]:
            dataset, embeds_dists = gen_ds.get_graph_dataset(
                args.dataset,
                store_to_path = "./data",
                batch_size = args.gnn_batch_size,
                normal_mode = args.data_normal_mode,
                data_config = dict(
                    domain = args.dataset_domain,
                    shift = args.dataset_shift,
                    generate = args.dataset_generate,
                    cal_domain = args.dataset_cal_domain,
                    test_domain = args.dataset_test_domain,
                ),
                n_partitions = args.dataset_n_partitions,  
                partitioning_ds = "test", 
                granularity_level = args.partitioning_granularity_level,
                seed = arg_seeds[i],
                transformation_type = args.input_transformation_type,
                transformation_config = dict(n_components = args.input_pca_n_components),
                report_ds_stats = True if i == 0 else False,
                encoder_config = dict(
                    num_itrs = args.gnn_num_hid_layers,
                )
            )
        elif args.dataset in ["GOODTwitch", "GOODWebKB", "GOODCBAS"]:
            dataset, embeds_dists = gen_ds.get_graph_dataset(
                args.dataset,
                store_to_path = "./data",
                batch_size = args.gnn_batch_size,
                normal_mode = args.data_normal_mode,
                data_config = dict(
                    domain = args.dataset_domain,
                    shift = args.dataset_shift,
                    generate = args.dataset_generate,
                    cal_domain = args.dataset_cal_domain,
                    test_domain = args.dataset_test_domain,
                ),
                n_partitions = args.dataset_n_partitions, 
                partitioning_ds = "test", 
                granularity_level = args.partitioning_granularity_level,
                seed = arg_seeds[i],
                transformation_type = args.input_transformation_type,
                transformation_config = dict(n_components = args.input_pca_n_components),
                loader_config = dict(
                    train_loader_type = args.train_loader_type,
                    batch_size = args.gnn_batch_size,
                    walk_length = args.gnn_num_hid_layers,
                    num_neighbors = [-1]*args.gnn_num_hid_layers, 
                    num_steps = args.node_loader_walker_steps,
                    sample_coverage = args.node_loader_walker_coverage,
                ),
                report_ds_stats = True if i == 0 else False,
                encoder_config = dict(
                    num_itrs = args.gnn_num_hid_layers,
                )
            )
        elif args.dataset in ["Cora_ML", "CiteSeer", "DBLP", "PubMed", "Cora", "Computers", "Photo", "Physics", "CS"]:
            dataset, embeds_dists = gen_ds.get_graph_dataset(
                args.dataset,
                store_to_path = "./data",
                normal_mode = args.data_normal_mode,
                train_test_split = args.train_test_split,
                data_config = dict(
                    domain = args.dataset_domain,
                    shift = args.dataset_shift,
                    generate = args.dataset_generate,
                    cal_domain = args.dataset_cal_domain,
                    test_domain = args.dataset_test_domain,
                ),
                n_partitions = args.dataset_n_partitions, 
                partitioning_ds = "test", 
                granularity_level = args.partitioning_granularity_level,
                seed = arg_seeds[i],
                transformation_type = args.input_transformation_type,
                transformation_config = dict(n_components = args.input_pca_n_components),
                loader_config = dict(
                    train_loader_type = args.train_loader_type,
                    batch_size = args.gnn_batch_size,
                    walk_length = args.gnn_num_hid_layers,
                    num_neighbors = [-1]*args.gnn_num_hid_layers, 
                    num_steps = args.node_loader_walker_steps,
                    sample_coverage = args.node_loader_walker_coverage,
                ),
                report_ds_stats = True if i == 0 else False,
                encoder_config = dict(
                    num_itrs = args.gnn_num_hid_layers,
                )
            )
        elif args.dataset in ["Anaheim", "ChicagoSketch", "twitch_PTBR", "county_education_2012", "county_election_2016", "county_income_2012", "county_unemployment_2012"]:
            dataset, embeds_dists = gen_ds.get_graph_dataset(
                args.dataset,
                store_to_path = "./data/cfgnn_regression",
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.data_normal_mode,
                data_config = dict(
                    domain = args.dataset_domain,
                    shift = args.dataset_shift,
                    generate = args.dataset_generate,
                    cal_domain = args.dataset_cal_domain,
                    test_domain = args.dataset_test_domain,
                ),
                n_partitions = args.dataset_n_partitions, 
                partitioning_ds = "test", 
                granularity_level = args.partitioning_granularity_level,
                seed = arg_seeds[i],
                transformation_type = args.input_transformation_type,
                transformation_config = dict(n_components = args.input_pca_n_components),
                loader_config = dict(
                    train_loader_type = args.train_loader_type,
                    batch_size = args.gnn_batch_size,
                    walk_length = args.gnn_num_hid_layers,
                    num_neighbors = [-1]*args.gnn_num_hid_layers, 
                    num_steps = args.node_loader_walker_steps,
                    sample_coverage = args.node_loader_walker_coverage,
                ),
                report_ds_stats = True if i == 0 else False,
                encoder_config = dict(
                    num_itrs = args.gnn_num_hid_layers,
                )
            )
        else:
            raise Exception("Dataset is not supported")

        if i == 0:
            embeds_dists_memory = embeds_dists

        if args.dataset.lower().startswith("good"):
            model_config = Munch(dict(
                dim_hidden = args.gnn_h_dim,
                model_layer = args.gnn_num_hid_layers,
                dropout_rate = args.gnn_dropout_rate,
                dim_ffn = args.gnn_h_dim,
                global_pool = args.gnn_pooling_mode,
                dataset_level = dataset.level,
                dim_node = dataset.n_feats,
                dim_edge = dataset.ne_feats,
                num_classes = dataset.n_classes if dataset.n_classes != 2 else 1,
                dataset_name = dataset.name,
                domain = args.dataset_domain,
                shift_type = args.dataset_shift,
                dataset_type = dataset.type,
                device = device,
            ))
        else:
            model_config = Munch(dict(
                gnn_type = args.gnn_type,
                in_channels = dataset.n_feats, 
                hidden_channels = args.gnn_h_dim, 
                out_channels = dataset.n_classes if dataset.n_classes != 2 else 1, 
                gnn_num_hid_layers = args.gnn_num_hid_layers,
                device = device,
                pooling = args.gnn_pooling_mode if dataset.level == "graph" else None,
                decoder = "linear" if dataset.level == "graph" else "gnn",
                with_nonlinearity = args.gnn_with_nonlinearity,
                with_bn = args.gnn_with_bn,
                in_dropout = args.gnn_dropout_rate,
                hid_dropout = args.gnn_dropout_rate,
                out_dropout = args.gnn_dropout_rate,
            ))
        
        optimizer_config = Munch(dict(
            lr = args.gnn_lr,
            scheduler_step_size = args.gnn_scheduler_step_size,
            scheduler_gamma = args.gnn_scheduler_gamma,
            weight_decay = args.gnn_weight_decay
        ))
        training_config = Munch(dict(
            n_epochs = args.gnn_n_epochs
        ))

        if len(args.pretrained_path) == 0:
            global_logger.info(f"Pretraining {args.gnn_type} on {args.dataset} started for {args.gnn_n_epochs} epochs")
            pred_p_path = "" if args.not_save_model else os.path.join(models_path, "predictive", f"{args.gnn_type}_{args.dataset}_{exec_trial_name}.pt")
            model, p_results = pretrain_model(
                dataset,
                optimizer_config,
                training_config,
                model_config = model_config,
                eval_step = args.gnn_eval_step,
                save_path = pred_p_path
            )
            pretrained_paths.append(pred_p_path)
            model.eval()
        else:
            p_results = {}
            model = get_model(
                model_config, dataset
            ).to(device).eval()
            load_model(model, read_checkpoint=True, pretrained_path=pretrained_paths[i])
            test_outs = evaluate(model, dataset, target_dataset = "test")

            log_content = f"GNN After Pretraining: -- Test Loss: {test_outs['loss']:.3f} "
            if dataset.task == "cls":
                log_content +=  (
                    f"-- Test ACC: {test_outs['acc']:.3f} "
                    f"-- Test F1: {test_outs['f1']:.3f} "
                    f"-- Test AUROC: {test_outs['auroc']:.3f} "
                )
                p_results["pretrained_test_acc"] = test_outs["acc"]
                p_results["pretrained_test_f1"] = test_outs["f1"]
                p_results["pretrained_test_auroc"] = test_outs["auroc"]
            else:
                log_content +=  (
                    f"-- Test Pearson Ratio: {test_outs['pearson_ratio']:.3f} "
                )
                p_results["pretrained_test_loss"] = test_outs["loss"]
                p_results["pretrained_pearson_ratio"] = test_outs["pearson_ratio"]
            # global_logger.info(log_content)
            global_logger.info(
                f"Exp. {i+1}/{total_iters} **** Pretrained Results: " + \
                str(list(map(lambda x:f"{' '.join(x.split('_')).title()}: {p_results[x]:.3f}", p_results)))
            )
        
        update_results(p_results)

        cal_outs = evaluate(model, dataset, target_dataset = "cal", metric_eval = False)
        test_outs = evaluate(model, dataset, target_dataset = "test", metric_eval = False)
        all_outs = evaluate(model, dataset, target_dataset = "all", metric_eval = False)

        n_cal = cal_outs["x"].shape[0]
        n_test = test_outs["x"].shape[0]

        if args.with_standard_scalar or (args.latent_pca_n_components > 0):
            scaler = StandardScaler(with_std=False)
            cal_outs["x"] = scaler.fit_transform(cal_outs["x"])
            test_outs["x"] = scaler.transform(test_outs["x"])
            all_outs["x"] = scaler.transform(all_outs["x"])

        if args.importance_weighting in ["input", "latent"]:
            split_char = args.dataset + "_"
            w_path = Path(weights_path) / (
                f"{args.dataset}_B{str(round(args.importance_weighting_kmm_b))}"
                f"_q{str(args.importance_weighting_kernelwidth_quantile).replace('.', '')}_{pretrained_paths[i].split(split_char)[-1]}"
            )

            pre_weights = get_weights(
                w_path,
                dataset,
                iw_config = dict(
                    batch_size = args.importance_weighting_batch_size,
                    max_iters = args.importance_weighting_max_iters,
                    kernel_width_q = args.importance_weighting_kernelwidth_quantile,
                    B = args.importance_weighting_kmm_b,
                    reg_fac = args.importance_weighting_reg_fac,
                    base_seed = arg_seeds[i]
                ), 
                setup = dict(
                    space = args.importance_weighting,
                    device = device
                ),
                encoder_config = dict(
                    num_itrs = args.gnn_num_hid_layers,
                    level = dataset.level 
                ), 
                latent_outs = dict(
                    cal_outs = cal_outs,
                    test_outs = test_outs
                )
            )

            weight_stats = {}
            normalized_pre_weights = pre_weights / (pre_weights.max() + 1e-9)
            weight_stats["weights_mean"] = normalized_pre_weights.mean()
            weight_stats["weights_median"] = np.median(normalized_pre_weights)
            weight_stats["weights_05_precentage"] = ((normalized_pre_weights >= 0.5).sum() / normalized_pre_weights.shape[0]).item()
            logger.info(f"Percentage of calibration samples with weights above 0.5: {weight_stats['weights_05_precentage']:.3f}")

            if embeds_dists_memory is not None:
                iw_cal_test_mmd = mmd(
                    embeds_dists_memory["cal_x"].cpu(), embeds_dists_memory["train_x"].cpu(), sigma_sq=1.0, 
                    iw=torch.from_numpy(normalized_pre_weights)
                ).numpy()
                # Normalized MMD Reduction (NMR)
                nmr = (embeds_dists_memory["org_cal_test_mmd"] - iw_cal_test_mmd) / (embeds_dists_memory["org_cal_test_mmd"] - embeds_dists_memory["org_cal_train_mmd"])
                weight_stats["normalized_mmd_reduction"] = nmr.item()
                logger.info(f"Normalized MMD Reduction (NMR): {weight_stats['normalized_mmd_reduction']:.3f}")
                
            update_results(weight_stats)

        else:
            pre_weights = None

        pca_config = None
        if args.latent_pca_n_components > 0:
            pca_config = {
                "sample_original_space": args.sample_original_space,
                "with_dynamic_bandwidth": args.with_dynamic_bandwidth
            }
            pca = PCA(args.latent_pca_n_components)
            pca.fit(cal_outs["x"])
            pca_config["pca"] = pca
            if (not args.sample_original_space):
                cal_outs["x"] = pca.transform(cal_outs["x"])
                test_outs["x"] = pca.transform(test_outs["x"])
                all_outs["x"] = pca.transform(all_outs["x"])
                pca_config["transform_x"] = True

        if args.visualize_data:
            if cal_outs["x"].shape[-1] == 2:
                plot_2d_embedding_scatter(
                    cal_outs=cal_outs,
                    test_outs=test_outs,
                    n_cal_total=n_cal,
                    n_cal_vis=n_cal,
                    n_test_vis=50,
                    bandwidth=args.bandwidth,
                    save_filename=f'2d_data_vis_{i}.pdf'
                )
            elif cal_outs["x"].shape[-1] == 3:
                plot_3d_embedding_scatter(
                    cal_outs=cal_outs,
                    test_outs=test_outs,
                    n_cal_total=n_cal,
                    n_cal_vis=n_cal,
                    n_test_vis=50,
                    bandwidth=args.bandwidth,
                    save_filename=f'3d_data_vis_{i}.pdf'
                )
            else:
                print("Cannot visualize feature above 3 dimensions!")

        conf_extra_args = {}
        if args.method == "split_cp":
            cp_model = SplitCP(
                alpha = args.alpha,
                task = dataset.task,
                device = device,
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
            )
        elif args.method == "cal_lcp":
            cp_model = CalibratedLocalizedCP(
                alpha = args.alpha,
                bandwidth = args.bandwidth,
                batch_size = args.cp_batch_size,
                task = dataset.task,
                device = device,
                weight_analysis_metrics = {
                    "entropy": {},
                    "center_of_mass": {},
                    "cumulative_weight": {"quantile": 0.5}
                },
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
            )
        elif args.method == "cfgnn":
            cp_model = CFGNN(
                alpha = args.alpha,
                batch_size = args.cp_batch_size,
                task = dataset.task,
                device = device,
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
                base_gnn = model,
                confnn_config = Munch({
                    "gnn_type": args.gnn_type,
                    "in_channels": dataset.n_classes if dataset.n_classes != 2 else 1, 
                    "hidden_channels": 64, 
                    "out_channels": dataset.n_classes if dataset.n_classes != 2 else 1, 
                    "gnn_num_hid_layers": 1,
                    "device": device,
                    "pooling": args.gnn_pooling_mode if dataset.level == "graph" else None,
                    "decoder": "linear" if dataset.level == "graph" else "gnn",
                    "with_nonlinearity": True,
                    "with_bn": False,
                    "in_dropout": 0.5,
                    "hid_dropout": 0.0,
                    "out_dropout": 0.0,
                })
            )
            conf_extra_args |= dict(dataset=dataset)
        elif args.method == "rand_lcp":
            cp_model = RandomlyLocalizedCP(
                alpha = args.alpha,
                bandwidth = args.bandwidth,
                batch_size = args.cp_batch_size,
                weight_analysis_metrics = {
                    "entropy": {},
                    "center_of_mass": {},
                    "cumulative_weight": {"quantile": 0.5}
                },
                task = dataset.task,
                device = device,
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
            )
        elif args.method == "graph_rlcp":
            cp_model = GraphRLCP(
                weighting_method = args.weighting_method,
                alpha = args.alpha,
                bandwidth = args.bandwidth,
                batch_size = args.cp_batch_size,
                kernel_type = args.kernel_type, 
                ppr_alpha = args.ppr_alpha,
                ppr_k = args.ppr_k,
                convex_dist_coeff = args.convex_dist_coeff,
                convex_coeff_method = args.convex_coeff_method,
                densification_config = dict(
                    threshold = args.graph_densification_threshold,
                    method = args.graph_densification_method,
                    adaptive = args.adapt_dense_threshold,
                    max_edge_ratio = args.dense_max_edge_ratio
                ),
                weight_analysis_metrics = {
                    "entropy": {},
                    "center_of_mass": {},
                    "cumulative_weight": {"quantile": 0.5}
                },
                task = dataset.task,
                device = device,
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
            )
            conf_extra_args |= dict(dataset = dataset)
        elif args.method == "test_cp":
            cp_model = TestCP(
                weighting_method = args.weighting_method,
                alpha = args.alpha,
                bandwidth = args.bandwidth,
                batch_size = args.cp_batch_size,
                kernel_type = args.kernel_type, 
                ppr_alpha = args.ppr_alpha,
                ppr_k = args.ppr_k,
                convex_dist_coeff = args.convex_dist_coeff,
                convex_coeff_method = args.convex_coeff_method,
                densification_threshold = args.graph_densification_threshold,
                densification_method = args.graph_densification_method,
                weight_analysis_metrics = {
                    "entropy": {},
                    "center_of_mass": {},
                    "cumulative_weight": {"quantile": 0.1}
                },
                task = dataset.task,
                device = device,
                eval_config = dict(
                    method = args.conformal_eval_method,
                    wsc_num_vectors = args.wsc_num_vectors
                ),
            )
        else:
            raise NotImplementedError

        cp_results = cp_model.predict(
            all_outs = all_outs,
            cal_outs = cal_outs,
            test_outs = test_outs,
            pre_weights = pre_weights,
            pca_config = pca_config,
            dataset = dataset
        )
        update_results(cp_results)

    global_logger.info(f"Args seeds: [{', '.join(arg_seeds.astype(str))}]")
    global_logger.info(f"Pretrained models are saved to {models_path} as: [{', '.join([p.replace('/workspace/CertainGNN', '.') for p in pretrained_paths])}]")

    if "predictions" in all_results:
        predictions = all_results.pop("predictions")
    
    for key, value in all_results.items():
        if isinstance(value[0], np.ndarray):
            all_results[key] = stats.describe(np.vstack(value), ddof=0)
        else:
            all_results[key] = stats.describe(np.array(value), ddof=0)

    output_dict = {
        "args": vars(args),
        "results": dict()
    }

    log_content = "\n" + "="*20 + " Final Results " + "="*20 + "\n"
    log_content += "Method: " + args.method.upper() + "\n"
    log_content += "Dataset: " + args.dataset.upper() + "\n"
    log_content += f"Main Config: \n\talpha: {args.alpha}\n\tbandwidth: {args.bandwidth}\n"

    for key, value in all_results.items():
        v_mean = np.atleast_1d(value.mean)
        v_std = np.atleast_1d(np.sqrt(value.variance))

        mean_content = ", ".join([f"{v:.3f}" for v in v_mean])
        std_content = ", ".join([f"{v:.3f}" for v in v_std])

        metric = ' '.join(key.split('_')).upper()
        log_content += f"{metric}: \n\tMean: {mean_content}\n\tStd: {std_content}\n"

        output_dict["results"][key] = {
            "mean": [round(float(v), 3) for v in v_mean],
            "std": [round(float(v), 3) for v in v_std]
        }

    global_logger.info(log_content)

    with open(json_out_path, "w") as f:
        json.dump(output_dict, f, indent=4)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = process_config(args, parser)
    main(args)