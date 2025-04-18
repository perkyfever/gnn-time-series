# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
import sys
from pathlib import Path
sys.path.append(Path(os.getcwd()).parent.parent.as_posix())

# %%
import math
import random
import pandas as pd

import scipy as sp
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import dgl
import dgl.nn as dglnn

import optuna
import networkx as nx

import sys
import json 

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from pathlib import Path
from collections import OrderedDict

from torch import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from tqdm import tqdm
from dataset import get_datasets, ETTDataset
from graph_features import build_graph, spectral_features, deepwalk_features

import warnings
warnings.simplefilter("ignore")

# %%
import wandb
wandb.login()

# %%
def seed_everything(seed=0xBAD5EED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %%
TOTAL_NUM_NODES = 7
TGT_COMP_IDX = TOTAL_NUM_NODES - 1

ALPHA = 0.05
HORIZON_SIZE = 720
LOOKBACK_SIZE = 96

DATASET_NAME = "ETTh1.csv"
NODES_EMBEDDINGS_FN = partial(spectral_features, embed_size=TOTAL_NUM_NODES)

# %% [markdown]
# # Dataset Setup

# %%
train_ds, val_ds, test_ds = get_datasets(
    dataset_name=DATASET_NAME,
    lookback_size=LOOKBACK_SIZE,
    horizon_size=HORIZON_SIZE
)

# %%
class CustomGraphDataset(Dataset):
    def __init__(self, dataset: ETTDataset, alpha: float = 0.05, graph_features_fn=None):
        super().__init__()
        self.graphs: list[dgl.DGLGraph] = []
        self.targets: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        for idx in tqdm(range(len(dataset)), desc="Building graphs"):
            x_time, x_data, y_data = dataset[idx]
            graph = build_graph(x_data, alpha=alpha)

            if graph_features_fn:
                graph_features = graph_features_fn(graph)
                graph.ndata["h"] = torch.cat([x_data.T, graph_features], dim=1)
            else:
                graph.ndata["h"] = x_data.T
            
            self.targets.append(y_data.T[TGT_COMP_IDX])
            self.times.append(x_time)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> tuple[dgl.DGLGraph, torch.Tensor]:
        return self.times[idx], self.graphs[idx], self.targets[idx]

# %%
def graph_collate_fn(batch):
    """
    Custom collate function for batching DGL graphs.
    :param graphs: batch of graphs and targets
    :returns: batched graph, batch of targets
    """
    batch_size = len(batch)
    times, graphs, targets = zip(*batch)
    horizon_size = targets[0].shape[0]
    lookback_size, n_features = times[0].shape
    times_tensor = torch.zeros((batch_size, lookback_size, n_features))
    targets_tensor = torch.zeros((batch_size, horizon_size))
    for idx in range(batch_size):
        targets_tensor[idx, :] = targets[idx]
        times_tensor[idx, :, :] = times[idx]

    return times_tensor, dgl.batch(graphs), targets_tensor

# %%
train_ds = CustomGraphDataset(train_ds, alpha=ALPHA, graph_features_fn=NODES_EMBEDDINGS_FN)
val_ds = CustomGraphDataset(val_ds, alpha=ALPHA, graph_features_fn=NODES_EMBEDDINGS_FN)
test_ds = CustomGraphDataset(test_ds, alpha=ALPHA, graph_features_fn=NODES_EMBEDDINGS_FN)

# %% [markdown]
# # Model setup

# %%
class GCNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        residual: bool = True,
        dropout: float = 0
    ) -> "GCNBlock":
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn

        self.gcn = dglnn.GraphConv(
            in_feats=self.input_dim,
            out_feats=self.hidden_dim
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = self.activation_fn()
        self.dropout = nn.Dropout(self.dropout)

        if self.residual:
            self.skip = (
                nn.Linear(self.input_dim, self.hidden_dim)
                if self.input_dim != self.hidden_dim
                else nn.Identity()
            )

    def forward(self, graph, features):
        outputs = self.gcn(graph, features)
        outputs = self.norm(outputs)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs)

        if self.residual:
            outputs = outputs + self.skip(features)

        return outputs

# %%
class GCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn: nn.Module,
        residual: bool = True,
        dropout: float = 0
    ) -> "GCNModel":
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            self.blocks.append(GCNBlock(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim,
                activation_fn=self.activation_fn,
                residual=self.residual,
                dropout=self.dropout
            ))
    
    def forward(self, graph, features):
        outputs = features
        for block in self.blocks:
            outputs = block(graph, outputs)
        
        return outputs

# %%
class Encoding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        weight = torch.zeros(self.input_dim, self.output_dim, requires_grad=False).float()
        pos_enc = torch.arange(0, self.input_dim).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.output_dim, 2).float() * -(math.log(10000.0) / self.output_dim))

        weight[:, 0::2] = torch.sin(pos_enc * div_term)
        weight[:, 1::2] = torch.cos(pos_enc * div_term)

        self.embeddings = nn.Embedding(self.input_dim, self.output_dim)
        self.embeddings.weight = nn.Parameter(weight, requires_grad=False) # not learnable

    def forward(self, x):
        return self.embeddings(x).detach()

class TimeEncoding(nn.Module):
    def __init__(self, output_dim: int, learnable: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.learnable = learnable

        Embedding = Encoding if self.learnable else nn.Embedding
        self.month_embed = Embedding(13, self.output_dim)
        self.weekday_embed = Embedding(7, self.output_dim)
        self.day_embed = Embedding(32, self.output_dim)
        self.hour_embed = Embedding(24, self.output_dim)
        self.minute_embed = Embedding(4, self.output_dim)
    
    def forward(self, x):
        x = x.long()
        month = self.month_embed(x[:, 0, 0])
        weekday = self.weekday_embed(x[:, 0, 1])
        day = self.day_embed(x[:, 0, 2])
        hour = self.hour_embed(x[:, 0, 3])
        minute = self.minute_embed(x[:, 0, 4])
        return month + weekday + day + hour + minute

# %%
class GraphTSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        horizon_size: int,
        activation_fn: nn.Module,
        residual: bool = True,
        dropout: float = 0,
        time_learnable: bool = False,
    ) -> "GraphTSModel":
        super().__init__()
        self.residual = residual
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.horizon_size = horizon_size
        self.time_learnable = time_learnable
        
        # self.time_emb = TimeEncoding(
        #     output_dim=self.input_dim,
        #     learnable=self.time_learnable
        # )

        self.backbone = GCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation_fn=self.activation_fn,
            residual=self.residual,
            dropout=self.dropout
        )

        self.head = nn.Linear(self.hidden_dim, self.horizon_size)
    
    def forward(self, graph, features, time_features):
        # time = self.time_emb(time_features)
        # time = time.unsqueeze(1).repeat(1, TOTAL_NUM_NODES, 1)
        x = features # + time.reshape(-1, features.shape[-1])
        outputs = self.backbone(graph, x)
        tgt_emb = outputs[TGT_COMP_IDX::TOTAL_NUM_NODES] # extract OT's embeddings
        outputs = self.head(tgt_emb)
        return outputs

# %% [markdown]
# # Tuning preparation

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def get_dataloaders(config):
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        collate_fn=graph_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=graph_collate_fn,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        collate_fn=graph_collate_fn,
        drop_last=True
    )
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}

# %%
def get_activation_fn(config):
    if config["activation_fn"] == "ReLU":
        return nn.ReLU
    return nn.LeakyReLU

# %%
def get_model(config):
    return GraphTSModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        horizon_size=HORIZON_SIZE,
        activation_fn=get_activation_fn(config),
        residual=config["residual"],
        dropout=config["dropout"],
        time_learnable=False
    )

# %%
def get_optimizer(model, config):
    if config["optimizer"] == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
    if config["optimizer"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

# %%
def get_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["patience"]
    )

# %%
def get_criterion(config):
    return nn.MSELoss()

# %% [markdown]
# # Training Steps

# %%
def train_step(train_loader, model, optimizer, loss_fn):
    model.train()
    loss_acum = 0
    samples_cnt = 0
    scaler = GradScaler()
    for times, graph, targets in train_loader:
        graph = graph.to(device)
        times = times.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(graph, graph.ndata["h"], times)
            loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_acum += targets.shape[0] * loss.item()
        samples_cnt += targets.shape[0]
        wandb.log({"train_mse": loss.item()})
    
    return {"loss": loss_acum / samples_cnt}

# %%
@torch.inference_mode()
def evaluation_step(val_loader, model, loss_fn):
    model.eval()
    loss_acum = 0
    samples_cnt = 0
    for times, graph, targets in val_loader:
        graph = graph.to(device)
        times = times.to(device)
        targets = targets.to(device)
        outputs = model(graph, graph.ndata["h"], times)
        loss = loss_fn(outputs, targets)
        loss_acum += targets.shape[0] * loss.item()
        samples_cnt += targets.shape[0]
        wandb.log({"val_mse": loss.item()})

    return {"loss": loss_acum / samples_cnt}

# %%
@torch.inference_mode()
def test_step(test_loader, model, loss_fn):
    model.eval()
    loss_acum = 0
    samples_cnt = 0
    for times, graph, targets in test_loader:
        graph = graph.to(device)
        times = times.to(device)
        targets = targets.to(device)
        outputs = model(graph, graph.ndata["h"], times)
        loss = loss_fn(outputs, targets)
        loss_acum += targets.shape[0] * loss.item()
        samples_cnt += targets.shape[0]
        wandb.log({"test_mse": loss.item()})

    return {"loss": loss_acum / samples_cnt}

# %%
def run_experiment(config):
    wandb.init(
        project="gnn-ts",
        group="tuning_runs",
        name="tuning",
        config=config
    )
    
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
        train_output = train_step(dataloaders["train"], model, optimizer, criterion)
        report["train"].append(train_output)

        validation_output = evaluation_step(dataloaders["val"], model, criterion)
        report["valid"].append(validation_output)

        test_output = test_step(dataloaders["test"], model, criterion)
        report["test"].append(test_output)
        
        pbar.set_postfix_str(
            f"[train] loss = {train_output['loss']:.4f}\t"
            f"[valid] mse = {validation_output['loss']:.4f}\t"
            f"[test]  mse = {test_output['loss']:.4f}"
        )

        if validation_output["loss"] < report["score"]:
            report["score"] = validation_output["loss"]
            report["epoch"] = epoch
        
        scheduler.step(validation_output["loss"])

    model.to("cpu")
    wandb.finish()
    
    return report

# %% [markdown]
# # Hyperparameters Tuning

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
        print(json.dumps(proposal_config, indent=4))
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
tuning_config = {
    "num_trials": 120,
    "num_startup_trials": 15,
    "model": "GCN",
    "alpha": ALPHA,
    "graph_strategy": "local",
    "lookback_size": LOOKBACK_SIZE,
    "horizon_size": HORIZON_SIZE,
    "time_usage": "None",
    "input_dim": LOOKBACK_SIZE + TOTAL_NUM_NODES,
    "lr_factor": 0.50,
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
        "high": 8,
        "step": 1,
    },
    "activation_fn": "ReLU",
    "residual": True,
    "batch_size": {
        "type": "int",
        "low": 32,
        "high": 128,
        "step": 1,
        "log": True
    },
    "num_epochs": 8,
    "weight_decay": 1e-5,
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log": True
    },
    "scheduler": "None",
    "optimizer": {
        "type": "categorical",
        "choices": ["Adam", "SGD"]
    },
    "dropout": {
        "type": "float",
        "low": 0.0,
        "high": 0.35,
    },
}

# %%
run_tuning(tuning_config)

# %%


# %%


# %%



