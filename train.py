import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import GradScaler, autocast

def train_step(train_loader, model, optimizer, loss_fn, device):
    model.train()
    mse_acum = 0
    mae_acum = 0
    samples_cnt = 0
    scaler = GradScaler()

    for graph, targets in train_loader:
        graph = graph.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(graph, graph.ndata["h"])
            loss = loss_fn(outputs, targets)
            mae = F.l1_loss(outputs, targets, reduction="sum")
            mse = F.mse_loss(outputs, targets, reduction="sum")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        mse_acum += mse.item()
        mae_acum += mae.item()
        samples_cnt += batch_size

    return {
        "mse": mse_acum / samples_cnt,
        "mae": mae_acum / samples_cnt
    }

@torch.inference_mode()
def evaluation_step(loader, model, device):
    model.eval()
    mse_acum = 0
    mae_acum = 0
    samples_cnt = 0

    for graph, targets in loader:
        graph = graph.to(device)
        targets = targets.to(device)

        outputs = model(graph, graph.ndata["h"])
        mae = F.l1_loss(outputs, targets, reduction="sum")
        mse = F.mse_loss(outputs, targets, reduction="sum")

        batch_size = targets.shape[0]
        mse_acum += mse.item()
        mae_acum += mae.item()
        samples_cnt += batch_size

    return {
        "mse": mse_acum / samples_cnt,
        "mae": mae_acum / samples_cnt
    }
