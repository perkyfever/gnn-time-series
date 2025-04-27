# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import sys
from pathlib import Path
sys.path.append(Path(os.getcwd()).parent.parent.as_posix())

# %%
import dgl
import json
import optuna

import torch
import torch.nn as nn

from functools import partial
from itertools import product
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from dataset import get_datasets, ETTDataset

from utils import seed_everything
from models.gcn import GCNModel

from constructor import construct_ess, construct_vanilla, construct_complete
from graph_features import spectral_features, deepwalk_features

from train import train_step, evaluation_step

import warnings
warnings.simplefilter("ignore")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything()

# %% [markdown]
# # Config

# %%
DATASET_NAME = "ETTh1.csv"

# LSTF setup
LOOKBACK_SIZE = 96
HORIZON_SIZE = 168

# Graphs setup
ALPHA = 0.05
GRAPH_CONSTRUCTION_FN = partial(construct_ess, alpha=ALPHA)
GRAPH_FEATURES_FN = partial(deepwalk_features, epochs=3, batch_size=256, lr=1e-2, weight_decay=1e-5, device=device, embed_size=7)

# Model setup
BATCH_SIZE = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.50
NUM_HEADS = 2
ACTIVATION_FN = nn.ReLU

# Train setup
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 2
LR_FACTOR = 0.5

# %% [markdown]
# # Dataset

# %%
# train_ds, val_ds, test_ds = get_datasets(
#     dataset_name="ETTh1.csv",
#     lookback_size=LOOKBACK_SIZE,
#     horizon_size=HORIZON_SIZE
# )

# %%
class DatasetAdapter(Dataset):
    def __init__(self, dataset: ETTDataset, graph_construction_fn, graph_features_fn=None):
        super().__init__()
        self.graphs: list[dgl.DGLGraph] = []
        self.targets: list[torch.Tensor] = []
        for idx in tqdm(range(len(dataset)), desc="Building graphs"):
            x_data, time_data, y_data = dataset[idx]
            graph = graph_construction_fn(x_data)

            if graph_features_fn:
                graph_features = graph_features_fn(graph)
                graph.ndata["h"] = torch.cat([x_data.T, graph_features], dim=1)
            else:
                graph.ndata["h"] = x_data.T
            
            graph.ndata["h"] = torch.cat([
                graph.ndata["h"],
                time_data.repeat(graph.number_of_nodes(), 1),
            ], dim=1)
            
            self.targets.append(y_data)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> tuple[dgl.DGLGraph, torch.Tensor]:
        return self.graphs[idx], self.targets[idx]

# %%
def graph_collate_fn(batch):
    """
    Custom collate function for batching DGL graphs.
    :param graphs: batch of graphs and targets
    :returns: batched graph, batch of targets
    """
    graphs, targets = zip(*batch)
    targets_tensor = torch.stack(targets, dim=0)
    return dgl.batch(graphs), targets_tensor

# %%
# train_adapter_ds = DatasetAdapter(
#     dataset=train_ds,
#     graph_construction_fn=GRAPH_CONSTRUCTION_FN,
#     graph_features_fn=GRAPH_FEATURES_FN
# )

# val_adapter_ds = DatasetAdapter(
#     dataset=val_ds,
#     graph_construction_fn=GRAPH_CONSTRUCTION_FN,
#     graph_features_fn=GRAPH_FEATURES_FN
# )

# test_adapter_ds = DatasetAdapter(
#     dataset=test_ds,
#     graph_construction_fn=GRAPH_CONSTRUCTION_FN,
#     graph_features_fn=GRAPH_FEATURES_FN
# )

# %%
# train_loader = DataLoader(
#     dataset=train_adapter_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=4,
#     collate_fn=graph_collate_fn
# )

# val_loader = DataLoader(
#     dataset=val_adapter_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=graph_collate_fn
# )

# test_loader = DataLoader(
#     dataset=test_adapter_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=graph_collate_fn
# )

# %%
# INPUT_DIM = train_adapter_ds[0][0].ndata["h"].shape[1]
# INPUT_DIM

# %% [markdown]
# # Model

# %%
class GraphTSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        horizon_size: int,
        activation_fn: nn.Module,
        dropout: float = 0,
    ) -> "GraphTSModel":
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.horizon_size = horizon_size

        self.backbone = GCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation_fn=self.activation_fn,
            dropout=self.dropout
        )

        self.head = nn.Linear(self.hidden_dim, self.horizon_size)
    
    def forward(self, graph, features):
        x = features
        outputs = self.backbone(graph, x)
        tgt_emb = outputs[6::7] # extract OT's embeddings
        outputs = self.head(tgt_emb)
        return outputs

# %% [markdown]
# # Getters

# %%
def get_dataloaders(config) -> dict[str, DataLoader]:
    train_loader = DataLoader(
        dataset=config["train_ds"],
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        collate_fn=graph_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=config["val_ds"],
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=graph_collate_fn,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=config["test_ds"],
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=graph_collate_fn,
        drop_last=True
    )
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}

# %%
def get_model(config) -> GraphTSModel:
    return GraphTSModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        horizon_size=config["horizon_size"],
        activation_fn=ACTIVATION_FN,
        dropout=config["dropout"]
    )

# %%
def get_optimizer(model, config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

# %%
def get_scheduler(optimizer, config) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min",
        factor=config["lr_factor"],
        patience=config["patience"]
    )

# %%
def get_criterion(config) -> nn.Module:
    return nn.MSELoss()

# %% [markdown]
# # Training

# %%
def run_experiment(config):    
    seed_everything()

    model = get_model(config)
    dataloaders = get_dataloaders(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion(config)
    
    report = {
        "train": [], "valid": [], "test": [],
        "score": 1e6, 'epoch': -1
    }
    
    global device
    model.to(device)

    pbar = tqdm(range(config["num_epochs"]))
    pbar.set_description("Training")
    
    for epoch in pbar:
        train_output = train_step(dataloaders["train"], model, optimizer, criterion, device)
        report["train"].append(train_output)

        validation_output = evaluation_step(dataloaders["val"], model, device)
        report["valid"].append(validation_output)

        test_output = evaluation_step(dataloaders["test"], model, device)
        report["test"].append(test_output)
        
        pbar.set_postfix_str(
            # f"[train] mse = {train_output['mse']:.4f} "
            # f"[train] mae = {train_output['mae']:.4f} "
            f"[valid] mse = {validation_output['mse']:.4f} "
            f"[valid] mae = {validation_output['mae']:.4f} "
            f"[test]  mse = {test_output['mse']:.4f} "
            f"[test]  mae = {test_output['mae']:.4f}"
        )

        if validation_output["mse"] < report["score"]:
            report["score"] = validation_output["mse"]
            report["epoch"] = epoch
        
        scheduler.step(validation_output["mse"])

    model.to("cpu")
    
    return report

# %%
def propose_config(config, trial):
    def propose_hparam_value(hparam_name, obj):
        hparam_value = obj
        if isinstance(obj, dict):
            distribution_type = obj["type"]
            distribution_kwargs = dict(filter(lambda p: p[0] != "type", obj.items()))
            suggest_fn = getattr(trial, f"suggest_{distribution_type}")
            hparam_value = suggest_fn(hparam_name, **distribution_kwargs)
        return hparam_value
    
    proposal_config = {}
    for hparam_name, obj in config.items():
        hparam_value = propose_hparam_value(hparam_name, obj)
        proposal_config[hparam_name] = hparam_value

    return proposal_config

def run_tuning(base_config):
    def objective(trial: optuna.Trial):
        proposal_config = propose_config(base_config, trial)
        # print(json.dumps(proposal_config, indent=4))
        experiment_report = run_experiment(proposal_config)
        sys.stdout.flush()
        return experiment_report["score"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=base_config["num_startup_trials"]
        )
    )
    study.optimize(objective, n_trials=base_config["num_trials"])
    return study

# %%
STUDY_RESULTS = {}
HORIZON_SIZES = [24, 48, 168, 336, 720]
GRAPH_CONSTRUCTION_FNS = [
    (construct_complete, "complete"),
    (partial(construct_ess, alpha=ALPHA), "ess"),
    (partial(construct_vanilla, alpha=ALPHA), "vanilla")
]
SETUPS = list(product(HORIZON_SIZES, GRAPH_CONSTRUCTION_FNS))

# %%
pbar = tqdm(SETUPS)
for HORIZON_SIZE, (GRAPH_CONSTRUCTION_FN, NAMING) in pbar:
    pbar.set_description(f"Running tuning for horizon size {HORIZON_SIZE} and {NAMING} graph")
    train_ds, val_ds, test_ds = get_datasets(
        dataset_name=DATASET_NAME,
        lookback_size=LOOKBACK_SIZE,
        horizon_size=HORIZON_SIZE
    )

    train_adapter_ds = DatasetAdapter(
        dataset=train_ds,
        graph_construction_fn=GRAPH_CONSTRUCTION_FN,
        graph_features_fn=GRAPH_FEATURES_FN
    )

    val_adapter_ds = DatasetAdapter(
        dataset=val_ds,
        graph_construction_fn=GRAPH_CONSTRUCTION_FN,
        graph_features_fn=GRAPH_FEATURES_FN
    )

    test_adapter_ds = DatasetAdapter(
        dataset=test_ds,
        graph_construction_fn=GRAPH_CONSTRUCTION_FN,
        graph_features_fn=GRAPH_FEATURES_FN
    )
    
    INPUT_DIM = train_adapter_ds[0][0].ndata["h"].shape[1]
    
    tuning_config = {
        "num_trials": 100,
        "num_startup_trials": 10,
        "train_ds": train_adapter_ds,
        "val_ds": val_adapter_ds,
        "test_ds": test_adapter_ds,
        "model": "GCN",
        "alpha": ALPHA,
        "graph_strategy": "ess",
        "graph_features": "dw",
        "lookback_size": LOOKBACK_SIZE,
        "horizon_size": HORIZON_SIZE,
        "time_usage": "None",
        "input_dim": INPUT_DIM,
        "lr_factor": 0.33,
        "patience": 2,
        "hidden_dim": {
            "type": "int",
            "low": 32,
            "high": 256,
            "step": 1,
            "log": True
        },
        "num_layers": {
            "type": "int",
            "low": 1,
            "high": 4,
            "step": 1,
        },
        "activation_fn": "ReLU",
        "batch_size": {
            "type": "int",
            "low": 32,
            "high": 256,
            "log": True
        },
        "num_epochs": 12,
        "weight_decay": 1e-5,
        "lr": {
            "type": "float",
            "low": 1e-4,
            "high": 1e-2,
            "log": True
        },
        "scheduler": "ReduceONPlateuau",
        "dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.35,
        },
    }
    
    study = run_tuning(tuning_config)
    STUDY_RESULTS[f"{HORIZON_SIZE}_{NAMING}"] = study

# %%
STUDY_RESULTS

# %%
import pickle

with open("dw_gcn_study_results.pkl", "wb") as f:
    pickle.dump(STUDY_RESULTS, f)


