import dgl
import dgl.nn as dglnn

import torch
from torch import GradScaler, autocast

from tqdm import tqdm
from utils import seed_everything

def spectral_features(graph: dgl.graph, embed_size: int) -> torch.Tensor:
    """
    Computes Laplacian eigenmaps.
    :param graph: graph to process
    :param embed_size: size of embeddings (must be <= num_nodes)
    :returns: spectral node embeddings of shape (num_nodes, embed_size)
    """
    A = graph.adj().to_dense()
    D = torch.diag(A.sum(dim=1))
    L = D - A
    eig_vecs = torch.linalg.eigh(L)[1]
    return eig_vecs[:, :embed_size]
    
def deepwalk_features(
    graph: dgl.graph, device: torch.device,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    embed_size: int, walk_length=60, window_size=7, negative_size=3
) -> torch.Tensor:
    """
    Computes DeepWalk features.
    :param graph: graph to process
    :param device: device for training
    :param epochs: number of epochs to train
    :param batch_size: nodes batch size
    :param lr: learing rate
    :param weight_decay: optimizer weight decay
    :param embed_size: size of embeddings
    :param walk_length: deepwalk length
    :param window_size: size of context window
    :param negative_size: negative samples factor
    :returns: deepwalk node embeddings of shape (num_nodes, embed_size)
    """
    deepwalk = dglnn.DeepWalk(
        g=graph.cpu(),
        emb_dim=embed_size,
        walk_length=walk_length,
        window_size=window_size,
        negative_size=negative_size,
        fast_neg=False,
        sparse=False
    )

    loader = torch.utils.data.DataLoader(
        dataset=torch.arange(graph.num_nodes()),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=deepwalk.sample,
        drop_last=True,
    )
    
    deepwalk.train()
    deepwalk = deepwalk.to(device)
    optimizer = torch.optim.AdamW(deepwalk.parameters(), lr=lr, weight_decay=weight_decay)
    
    seed_everything()
    scaler = GradScaler()
    for epoch in range(epochs):
        pbar = tqdm(loader, leave=False)
        pbar.set_description(f'epoch = {epoch}')
        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)

            with autocast(device_type="cuda"):
                loss = deepwalk(batch)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix_str(f'loss = {loss.item():.4f}')
    
    node_embeds = deepwalk.node_embed.weight.detach().cpu()
    return node_embeds
    