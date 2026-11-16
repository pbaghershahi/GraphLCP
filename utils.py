import torch, random, os, logging, shutil
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_dense_adj, subgraph, remove_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from sklearn.preprocessing import StandardScaler
import ipdb



def setup_logger(
    name="global_logger",
    level=logging.DEBUG,
    stream_handler=True,
    file_handler=True,
    formatter=None,
    log_file='default.log'
    ):
    open(log_file, 'w').close()
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    logger.propagate = False
    
    if formatter is None:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
            )

    if stream_handler:
        sth = logging.StreamHandler()
        sth.setLevel(level)
        sth.setFormatter(formatter)
        logger.addHandler(sth)

    if file_handler:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def fix_seed(seed_value, random_lib=True, numpy_lib=True, torch_lib=True):
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)



def normalize_(input_tensor, dim=0, mode="max", normal_params=None):
    input_tensor = input_tensor.float()
    if normal_params is not None:
        if mode == "max":
            input_tensor.div_(normal_params)
        else:
            scalar = normal_params
            input_tensor = scalar.transform(input_tensor.numpy())
    else:
        if mode == "max":
            max_value = input_tensor.max(dim=0).values
            max_value[max_value==0] = 1.
            input_tensor.div_(max_value)
            normal_params = max_value
        else:
            scalar = StandardScaler()
            input_tensor = scalar.fit_transform(input_tensor.numpy())
            normal_params = scalar
    return torch.as_tensor(input_tensor), normal_params



def load_model(cmodel, pmodel=None, read_checkpoint=True, pretrained_path=None):
    if read_checkpoint and pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path, weights_only=True)["model_state_dict"]
    else:
        pretrained_dict = pmodel.state_dict()
    model_dict = cmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    cmodel.load_state_dict(pretrained_dict)



class DummyLogger():
    def __init__(self) -> None:
        pass

    def info(self, log_content):
        print(log_content)
