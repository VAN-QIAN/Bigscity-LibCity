import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def calculate_normalized_laplacian(adj):
    """
    A = A + I
    L = D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


class TGCNCell(nn.Module):
    def __init__(self, num_units, afc_mx,adj_mx,adj_mx1, num_nodes,coarse_nodes, device, input_dim=1):
        # ----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.coarse_nodes = coarse_nodes
        self.input_dim = input_dim
        self._device = device
        self.act = torch.tanh

        # 这里提前构建好拉普拉斯
        support = calculate_normalized_laplacian(adj_mx)
        support1 = calculate_normalized_laplacian(adj_mx1)
        self.adj_mx = adj_mx
        self.adj_mx1 = adj_mx1
        self.afc_mx = torch.tensor(afc_mx.T,device=self._device)
        self.afc_mxt = torch.tensor(afc_mx,device=self._device)
        self.normalized_adj = self._build_sparse_matrix(support, self._device)
        self.normalized_adj1 = self._build_sparse_matrix(support1, self._device)
        self.init_params()

    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_0 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

        weight_01 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_01 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_11 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_11 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_01)
        torch.nn.init.xavier_normal_(weight_11)
        torch.nn.init.constant_(bias_01, bias_start)
        torch.nn.init.constant_(bias_11, bias_start)

        self.register_parameter(name='weights_01', param=weight_01)
        self.register_parameter(name='weights_11', param=weight_11)
        self.register_parameter(name='bias_01', param=bias_01)
        self.register_parameter(name='bias_11', param=bias_11)

        self.weigts1 = {weight_01.shape: weight_01, weight_11.shape: weight_11}
        self.biases1 = {bias_01.shape: bias_01, bias_11.shape: bias_11}

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, state):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)

        Returns:
            torch.tensor: shape (B, num_nodes * gru_units)
        """
        output_size = 2 * self.num_units
        x1,x1fc = self._gc(inputs, state, output_size, bias_start=1.0)
        value= torch.sigmoid(x1)  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))
        c0,c1=self._gc(inputs, r * state, self.num_units)
        c = self.act(c0)
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c

        # value1= torch.sigmoid(x1fc)  # (batch_size, self.num_nodes, output_size)
        # r, u = torch.split(tensor=value1, split_size_or_sections=self.num_units, dim=-1)
        # r = torch.reshape(r, (-1, self.coarse_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        # u = torch.reshape(u, (-1, self.coarse_nodes * self.num_units))
        # c = self.act(c1)
        # c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        # new_state1 = u * state + (1.0 - u) * c
        return new_state
    
    def _gc(self, inputs, state, output_size, bias_start=0.0):
        """
        GCN

        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            output_size:
            bias_start:

        Returns:
            torch.tensor: (B, num_nodes , output_size)
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.dim)
        # coarse = torch.
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        # state1 =  torch.reshape(state, (batch_size, self.coarse_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2] #input_size=inputs.shape[2] 

        x = inputs_and_state #inputs #
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1)) #(num_nodes, batch*dim)
        # d = np.array(self.afc_mx.sum(1))
        x0fc = torch.matmul(self.afc_mx.float(), x0.float()) # (coarse_nodes, batch*dim)
        # x0fc = x0fc/d
        

        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())  # A * X H0
        x1fc = torch.sparse.mm(self.normalized_adj1.float(), x0fc.float())  # A * Xc HC0
        x1 = x1 + torch.sigmoid(torch.matmul(self.afc_mxt.float(), x0fc.float())) #H1
        x1fc = x1fc + torch.sigmoid(torch.matmul(self.afc_mx.float(),x1.float())) #H1C
        x1 = x1.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1fc = x1fc.reshape(shape=(self.coarse_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.permute(2, 0, 1)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))

        weights = self.weigts[(input_size, output_size)]
        weights1 = self.weigts1[(input_size, output_size)]
        x1 = torch.matmul(x1, weights)#+torch.sigmoid(torch.matmul(x1fc,weights1)) #torch.matmul(x1, weights)  # (batch_size * self.num_nodes, output_size)
        x1fc = torch.matmul(x1fc,weights1)#+torch.sigmoid(torch.matmul(x1,weights)) #torch.matmul(
        biases = self.biases[(output_size,)]
        biases1 = self.biases1[(output_size,)]
        x1 += biases
        x1fc += biases1

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        x1fc = x1fc.reshape(shape=(batch_size, self.coarse_nodes, output_size))
        return x1,x1fc


class TGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.adj_mx1 = data_feature.get('adj_mx1')
        self.coarse_nodes = data_feature.get('coarse_nodes')
        self.afc_mx = data_feature.get('afc_mx')#Add the afc matrix in the data feature
        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda', 0.0015)

        super().__init__(config, data_feature)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # -------------------构造模型-----------------------------
        self.tgcn_model = TGCNCell(self.gru_units,self.afc_mx ,self.adj_mx,self.adj_mx1 ,self.num_nodes,self.coarse_nodes ,self.device, self.input_dim)
        self.output_model = nn.Linear(self.gru_units, self.output_window * self.output_dim)

    def forward(self, batch):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        # labels = batch['y']
        

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        # coarse_inputs = self.afc_mx.T * inputs.reshape(shape=(num_nodes, -1))
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        # coarse_inputs = coarse_inputs.reshape(num_nodes,batch_size, input_window, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)

        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        for t in range(input_window):
            state = self.tgcn_model(inputs[t], state)

        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        output = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        output = output.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        # output1 = output.permute(1,2,3,0)
        # output1 = self.afc_mx.T * output1
        output = output.permute(0, 2, 1, 3)
        return output

    def calculate_loss(self, batch):
        lam = self.lam
        lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())

        labels = batch['y']
        y_predicted = self.predict(batch)

        y_true = self._scaler.inverse_transform(labels[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        loss = torch.mean(torch.norm(y_true - y_predicted) ** 2 / 2) + lam * lreg
        loss /= y_predicted.numel()
        # return loss.masked_mae_torch(y_predicted, y_true, 0)
        return loss

    def predict(self, batch):
        return self.forward(batch)


class CTGCNCell(nn.Module):
    def __init__(self, num_units, afc_mx,adj_mx,adj_mx1, num_nodes,coarse_nodes, device, input_dim=1):
        # ----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.coarse_nodes = coarse_nodes
        self.input_dim = input_dim
        self._device = device
        self.act = torch.tanh

        # 这里提前构建好拉普拉斯
        support = calculate_normalized_laplacian(adj_mx)
        support1 = calculate_normalized_laplacian(adj_mx1)
        self.adj_mx = adj_mx
        self.adj_mx1 = adj_mx1
        self.afc_mx = torch.tensor(afc_mx.T,device=self._device)
        self.afc_mxt = torch.tensor(afc_mx,device=self._device)
        self.normalized_adj = self._build_sparse_matrix(support, self._device)
        self.normalized_adj1 = self._build_sparse_matrix(support1, self._device)
        self.init_params()

    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_0 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

        weight_01 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_01 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_11 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_11 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_01)
        torch.nn.init.xavier_normal_(weight_11)
        torch.nn.init.constant_(bias_01, bias_start)
        torch.nn.init.constant_(bias_11, bias_start)

        self.register_parameter(name='weights_01', param=weight_01)
        self.register_parameter(name='weights_11', param=weight_11)
        self.register_parameter(name='bias_01', param=bias_01)
        self.register_parameter(name='bias_11', param=bias_11)

        self.weigts1 = {weight_01.shape: weight_01, weight_11.shape: weight_11}
        self.biases1 = {bias_01.shape: bias_01, bias_11.shape: bias_11}

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, state):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)

        Returns:
            torch.tensor: shape (B, num_nodes * gru_units)
        """
        output_size = 2 * self.num_units
        x1,x1fc = self._gc(inputs, state, output_size, bias_start=1.0)
        value= torch.sigmoid(x1)  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))
        c0,c1=self._gc(inputs, r * state, self.num_units)
        c = self.act(c0)
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c

        # value1= torch.sigmoid(x1fc)  # (batch_size, self.num_nodes, output_size)
        # r, u = torch.split(tensor=value1, split_size_or_sections=self.num_units, dim=-1)
        # r = torch.reshape(r, (-1, self.coarse_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        # u = torch.reshape(u, (-1, self.coarse_nodes * self.num_units))
        # c = self.act(c1)
        # c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        # new_state1 = u * state + (1.0 - u) * c
        return new_state
    
    def _gc(self, inputs, state, output_size, bias_start=0.0):
        """
        GCN

        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            output_size:
            bias_start:

        Returns:
            torch.tensor: (B, num_nodes , output_size)
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.dim)
        # coarse = torch.
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        # state1 =  torch.reshape(state, (batch_size, self.coarse_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2] #input_size=inputs.shape[2] 

        x = inputs_and_state #inputs #
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1)) #(num_nodes, batch*dim)
        # d = np.array(self.afc_mx.sum(1))
        x0fc = torch.matmul(self.afc_mx.float(), x0.float()) # (coarse_nodes, batch*dim)
        # x0fc = x0fc/d
        

        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())  # A * X
        x1fc = torch.sparse.mm(self.normalized_adj1.float(), x0fc.float())  # A * Xc
        x1 = x1 + torch.sigmoid(torch.matmul(self.afc_mxt.float(), x0fc.float()))
        x1fc = x1fc + torch.sigmoid(torch.matmul(self.afc_mx.float(),x1.float()))
        x1 = x1.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1fc = x1fc.reshape(shape=(self.coarse_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.permute(2, 0, 1)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))

        weights = self.weigts[(input_size, output_size)]
        weights1 = self.weigts1[(input_size, output_size)]
        x1 = torch.matmul(x1, weights)#+torch.sigmoid(torch.matmul(x1fc,weights1)) #torch.matmul(x1, weights)  # (batch_size * self.num_nodes, output_size)
        x1fc = torch.matmul(x1fc,weights1)#+torch.sigmoid(torch.matmul(x1,weights)) #torch.matmul(
        biases = self.biases[(output_size,)]
        biases1 = self.biases1[(output_size,)]
        x1 += biases
        x1fc += biases1

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        x1fc = x1fc.reshape(shape=(batch_size, self.coarse_nodes, output_size))
        return x1,x1fc


class CTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.adj_mx1 = data_feature.get('adj_mx1')
        self.coarse_nodes = data_feature.get('coarse_nodes')
        self.afc_mx = data_feature.get('afc_mx')#Add the afc matrix in the data feature
        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda', 0.0015)

        super().__init__(config, data_feature)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # -------------------构造模型-----------------------------
        self.tgcn_model = TGCNCell(self.gru_units,self.afc_mx ,self.adj_mx,self.adj_mx1 ,self.num_nodes,self.coarse_nodes ,self.device, self.input_dim)
        self.output_model = nn.Linear(self.gru_units, self.output_window * self.output_dim)

    def forward(self, batch):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        # labels = batch['y']
        

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        # coarse_inputs = self.afc_mx.T * inputs.reshape(shape=(num_nodes, -1))
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        # coarse_inputs = coarse_inputs.reshape(num_nodes,batch_size, input_window, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)

        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        for t in range(input_window):
            state = self.tgcn_model(inputs[t], state)

        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        output = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        output = output.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        # output1 = output.permute(1,2,3,0)
        # output1 = self.afc_mx.T * output1
        output = output.permute(0, 2, 1, 3)
        return output

    def calculate_loss(self, batch):
        lam = self.lam
        lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())

        labels = batch['y']
        y_predicted = self.predict(batch)

        y_true = self._scaler.inverse_transform(labels[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        loss = torch.mean(torch.norm(y_true - y_predicted) ** 2 / 2) + lam * lreg
        loss /= y_predicted.numel()
        # return loss.masked_mae_torch(y_predicted, y_true, 0)
        return loss

    def predict(self, batch):
        return self.forward(batch)