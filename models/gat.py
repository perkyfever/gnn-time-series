import torch.nn as nn
import dgl.nn as dglnn

class GATv2Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        activation_fn: nn.Module,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation_fn
        self.dropout = dropout

        assert self.hidden_dim % self.num_heads == 0
        output_dim = self.hidden_dim // self.num_heads
        
        self.norm = nn.LayerNorm(input_dim)
        self.gat = dglnn.GATv2Conv(
            in_feats=self.input_dim,
            out_feats=output_dim,
            num_heads=self.num_heads,
            activation=self.activation(),
            feat_drop=dropout,
            residual=True
        )
        self.Wo = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, graph, features):
        outputs = self.norm(features)
        outputs = self.gat(graph, outputs).reshape(graph.num_nodes(), -1)
        outputs = self.Wo(outputs)
        return outputs

class GATv2Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        activation_fn: nn.Module,
        dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(GATv2Block(
                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn,
                dropout=dropout
            ))
    
    def forward(self, graph, features):
        outputs = features
        for block in self.blocks:
            outputs = block(graph, outputs)
        
        return outputs
