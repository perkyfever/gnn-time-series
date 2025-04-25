import torch.nn as nn
import dgl.nn as dglnn

class GCNBlock(nn.Module):
    def __init__(
        self, input_dim: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        dropout: float = 0
    ):
        super().__init__()
        self.dropout = dropout
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
        outputs = outputs + self.skip(features)
        return outputs

class GCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn: nn.Module,
        dropout: float = 0
    ):
        super().__init__()
        self.dropout = dropout
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
                dropout=self.dropout
            ))
    
    def forward(self, graph, features):
        outputs = features
        for block in self.blocks:
            outputs = block(graph, outputs)

        return outputs
