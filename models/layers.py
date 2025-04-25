import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)
        
    def forward(self, X):
        return self.linear(X)


class MLPLayer(nn.Module):
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)
        
    def forward(self, X):
        X = F.relu(self.linear_in(X))
        X = self.linear_out(X)
        return X

class LinearLayerNorm(nn.Module): # TODO: what for?
    def __init__(self, d_input, d_output):
        super().__init__()
        self.ln = nn.LayerNorm(d_input)
        self.linear = LinearLayer(d_input, d_output)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, X):
        X = self.ln(X)
        X = F.relu(self.linear(X))
        X = self.dropout(X)
        return X

class MLPLayerNorm(nn.Module):
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(d_input)
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, X):
        X = self.ln(X)
        X = F.relu(self.linear_in(X))
        X = self.linear_out(X)
        # X = self.dropout(X)
        return X

class MLP2Layer(nn.Module):
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_hidden = LinearLayer(d_hidden, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)
        
    def forward(self, X):
        X = F.gelu(self.linear_in(X))
        X = F.gelu(self.linear_hidden(X))
        X = self.linear_out(X)
        return X

class MLP2LayerBatchNorm(nn.Module):
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d_input)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_hidden = LinearLayer(d_hidden, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)
        
    def forward(self, X):
        X = F.gelu(self.linear_in(self.bn1(X)))
        X = F.gelu(self.linear_hidden(self.bn2(X)))
        X = self.linear_out(X)
        return X
    
class MLP3Layer(nn.Module): # TODO: what for?
    def __init__(self, d_input, d_output, d_hidden):
        super().__init__()
        # self.bn1 = nn.BatchNorm1d(d_input)
        # self.bn2 = nn.BatchNorm1d(d_hidden)
        #self.bn3 = nn.BatchNorm1d(d_hidden)
        self.linear_in = LinearLayer(d_input, d_hidden)
        self.linear_hidden1 = LinearLayer(d_hidden, d_hidden)
        self.linear_hidden2 = LinearLayer(d_hidden, d_hidden)
        self.linear_out = LinearLayer(d_hidden, d_output)
        
    def forward(self, X):
        X = F.gelu(self.linear_in(X))
        X = F.gelu(self.linear_hidden1(X))
        X = F.gelu(self.linear_hidden2(X))
        X = self.linear_out(X)
        return X
    
class HyperLinearLayer(nn.Module):
    def __init__(self, d_input, d_output, d_hyper_hidden, n_input=1, d_embedding=8, embedding=None):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        if embedding is not None:
            self.embedding = nn.Parameter(embedding, requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([n_input, d_embedding], requires_grad=True))
        self.hyperweights = MLP2Layer(d_embedding, d_input*d_output+d_output, d_hyper_hidden)

    def forward(self, X):
        weights_bias = self.hyperweights(self.embedding)
        W = weights_bias[:,:self.d_input*self.d_output].reshape(-1, self.d_output, self.d_input)
        b = weights_bias[:,self.d_input*self.d_output:]
        X = torch.einsum("bct,cht->bch", X, W) + b
        return X

class HyperLinearLayer3(nn.Module):
    def __init__(self, d_input, d_output, d_hyper_hidden, n_input=1, d_embedding=8, embedding=None):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        if embedding is not None:
            self.embedding = nn.Parameter(embedding, requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([n_input, d_embedding], requires_grad=True))
        self.hyperweights = MLP3Layer(d_embedding, d_input*d_output+d_output, d_hyper_hidden)

    def forward(self, X):
        weights_bias = self.hyperweights(self.embedding)
        W = weights_bias[:,:self.d_input*self.d_output].reshape(-1, self.d_output, self.d_input)
        b = weights_bias[:,self.d_input*self.d_output:]
        X = torch.einsum("bct,cht->bch", X, W) + b
        return X

class HyperMLPLayer(nn.Module): # TODO: what for?
    def __init__(self, d_input, d_output, d_hidden, d_hyper_hidden, n_input=1, d_embedding=8, embedding=None):
        super().__init__()
        self.shape = (d_output, d_input, d_hidden)
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        
        if embedding!=None:
            self.condition_embeddings = nn.Parameter(embedding, requires_grad=True)
            d_embedding = embedding.shape[1]
        else:
            U, _, V = torch.svd(torch.randn([n_input, d_embedding]))
            W = U @ V.T
            self.condition_embeddings = nn.Parameter(W, requires_grad=False)

        self.hypernet = MLPLayer(d_embedding, d_hidden*d_output+d_output, d_hyper_hidden)
        self.mainnet = MLPLayer(d_input, d_output, d_hidden)
        
    def forward(self, x, condition_id=0):
        weights_hat = self.hypernet(self.condition_embeddings)[condition_id]
        del self.mainnet.linear_out.linear.weight
        del self.mainnet.linear_out.linear.bias
        self.mainnet.linear_out.linear.weight = weights_hat[:self.d_hidden*self.d_output].reshape(self.d_output, self.d_hidden)
        self.mainnet.linear_out.linear.bias = weights_hat[self.d_hidden*self.d_output:]
        return self.mainnet(x)

class HyperMLP2Layer(nn.Module): # TODO: what for?
    def __init__(self, d_input, d_output, d_hidden, d_hyper_hidden, n_input=1, d_embedding=8, embedding=None):
        super().__init__()
        self.shape = (d_output, d_input, d_hidden)
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        
        if embedding!=None:
            self.condition_embeddings = embedding
        else:
            self.condition_embeddings = nn.Parameter(torch.randn([n_input, d_embedding], requires_grad=True))

        self.hypernet = MLPLayer(d_embedding, d_hidden*d_output+d_output, d_hyper_hidden)
        self.mainnet = MLP2Layer(d_input, d_output, d_hidden)
        
    def forward(self, x, condition_id=0):
        weights_hat = self.hypernet(self.condition_embeddings)[condition_id]
        del self.mainnet.linear_out.linear.weight
        del self.mainnet.linear_out.linear.bias
        self.mainnet.linear_out.linear.weight = weights_hat[:self.d_hidden*self.d_output].reshape(self.d_output, self.d_hidden)
        self.mainnet.linear_out.linear.bias = weights_hat[self.d_hidden*self.d_output:]
        return self.mainnet(x)

class ReversibleInstanceNormalization(nn.Module):
    def __init__(self, n_channels: int, affine=True, eps=1e-5):
        """
        :param n_channels: the number of features or channels
        :param affine: if True, RevIN has learnable affine parameters
        :param eps: a value added for numerical stability
        """
        super().__init__()
        self.n_channels = n_channels
        self.affine = affine
        self.eps = eps
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode=="norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode=="denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.n_channels))
        self.affine_bias = nn.Parameter(torch.zeros(self.n_channels))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.std
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.std
        x = x + self.mean
        return x

class MovingAverageLayer(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, X):
        X_front = X[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        X_end = X[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        X = torch.cat([X_front, X, X_end], dim=1)
        
        X = self.avg(X.permute(0, 2, 1))
        X = X.permute(0, 2, 1)

        return X

class SeasonalTrendDecompositionLayer(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_average = MovingAverageLayer(kernel_size)

    def forward(self, X):
        X = X.transpose(1, 2)
        X_trend = self.moving_average(X)
        X_season = X - X_trend
        return X_season.transpose(1, 2), X_trend.transpose(1, 2)
    
class EmbeddingInverted(nn.Module):
    def __init__(self, c_in, d_model, p_dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        # x: [Batch Variate Time]
        X = self.value_embedding(X)
        # x: [Batch Variate d_model]
        return self.dropout(X)
    
class PatchingLayer(nn.Module):
    def __init__(self, d_patch, stride):
        super().__init__()
        self.d_patch = d_patch
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, stride))
    
    def forward(self, X):
        X = self.padding(X)
        return X.unfold(dimension=-1, size=self.d_patch, step=self.stride)
    