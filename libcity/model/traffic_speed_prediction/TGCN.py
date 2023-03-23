import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from torch import Tensor


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
    def __init__(self, num_units, afc_mx, adj_mx, adj_mx1, num_nodes, coarse_nodes, super_nodes, device, n1, n2,
                 input_dim=1):
        # ----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.coarse_nodes = coarse_nodes
        self.super_nodes = super_nodes
        self.input_dim = input_dim
        self._device = device
        self.n1 = n1
        self.n2 = n2
        self.n3 = n2
        self.n4 = n2
        self.act = torch.tanh

        # 这里提前构建好拉普拉斯
        support = calculate_normalized_laplacian(adj_mx)
        support1 = calculate_normalized_laplacian(adj_mx1)
        self.adj_mx = adj_mx
        self.adj_mx1 = adj_mx1
        self.afc_mx = torch.tensor(afc_mx.T, device=self._device)
        self.afc_mxt = torch.tensor(afc_mx, device=self._device)
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
        # Second layer
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

        # Third layer
        weight_02 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_02 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_12 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_12 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_02)
        torch.nn.init.xavier_normal_(weight_12)
        torch.nn.init.constant_(bias_02, bias_start)
        torch.nn.init.constant_(bias_12, bias_start)

        self.register_parameter(name='weights_02', param=weight_02)
        self.register_parameter(name='weights_12', param=weight_12)
        self.register_parameter(name='bias_02', param=bias_02)
        self.register_parameter(name='bias_12', param=bias_12)

        self.weigts2 = {weight_02.shape: weight_02, weight_12.shape: weight_12}
        self.biases2 = {bias_02.shape: bias_02, bias_12.shape: bias_12}

        #     Parameter for the trainable assign matrix
        assMatrix = torch.nn.Parameter(torch.ones((self.coarse_nodes, self.super_nodes), device=self._device))
        self.register_parameter(name='assMatrix', param=assMatrix)
        self.assMatrix = assMatrix

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def dense_diff_pool(
            self,
            x: Tensor,
            adj: Tensor,
            s: Tensor,
            # mask: Optional[Tensor] = None,
            normalize: bool = True,
    ):
        r"""The differentiable pooling operator from the `"Hierarchical Graph
        Representation Learning with Differentiable Pooling"
        <https://arxiv.org/abs/1806.08804>`_ paper

        .. math::
            \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
            \mathbf{X}

            \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
            \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

        based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
        \times N \times C}`.
        Returns the pooled node feature matrix, the coarsened adjacency matrix and
        two auxiliary objectives: (1) The link prediction loss

        .. math::
            \mathcal{L}_{LP} = {\| \mathbf{A} -
            \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
            \|}_F,

        and (2) the entropy regularization

        .. math::
            \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            s (torch.Tensor): Assignment tensor
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
                with number of clusters :math:`C`.
                The softmax does not have to be applied before-hand, since it is
                executed within this method.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            normalize (bool, optional): If set to :obj:`False`, the link
                prediction loss is not divided by :obj:`adj.numel()`.
                (default: :obj:`True`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = torch.tensor(adj).float()
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()

        s = torch.softmax(s, dim=-1)

        # if mask is not None:
        #     mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        #     x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        if normalize is True:
            link_loss = link_loss / adj.numel()

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        # out_adj = out_adj.numpy()

        return out, out_adj, link_loss, ent_loss

    def forward(self, inputs, state, state1, state2):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)

        Returns:
            torch.tensor: shape (B, num_nodes * gru_units)
        """
        output_size = 2 * self.num_units
        x1, x1fc, x1cs, adj2 = self._gc(inputs, state, state1, state2, output_size, bias_start=1.0)

        value = torch.sigmoid(x1)  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        value1 = torch.sigmoid(x1fc)  # (batch_size, self.coarse_nodes, output_size)
        r1, u1 = torch.split(tensor=value1, split_size_or_sections=self.num_units, dim=-1)
        value2 = torch.sigmoid(x1cs)  # (batch_size, self.super_nodes, output_size)
        r2, u2 = torch.split(tensor=value2, split_size_or_sections=self.num_units, dim=-1)

        r2 = torch.reshape(r2, (-1, self.super_nodes * self.num_units))
        u2 = torch.reshape(u2, (-1, self.super_nodes * self.num_units))

        r1 = torch.reshape(r1,
                           (-1, self.coarse_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u1 = torch.reshape(u1, (-1, self.coarse_nodes * self.num_units))

        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))

        c0 = self._ggc(inputs, r * state, r1 * state1, r2*state2,self.num_units,adj2)
        c = self.act(c0)
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c

        c1 = self._g2c(inputs, r1 * state1, new_state, self.num_units)  # or new_state?
        c1 = self.act(c1)
        c1 = c1.reshape(shape=(-1, self.coarse_nodes * self.num_units))
        new_state1 = u1 * state1 + (1.0 - u1) * c1

        c2 = self._gcsc(inputs, r2 * state2, new_state1,adj2 ,self.num_units)  # or new_state?
        c2 = self.act(c2)
        c2 = c2.reshape(shape=(-1, self.super_nodes * self.num_units))
        new_state2 = u2 * state2 + (1.0 - u2) * c2

        return new_state, new_state1, new_state2,adj2,torch.softmax(self.assMatrix, dim=-1)

    def _ggc(self, inputs, state, state1,state2,output_size,adj_mx2 ,bias_start=0.0):
        """
        2nd layer GCN for fine nodes

        n1: from coarse_nodes to the fine_nodes

        n3: from super nodes to the fine_nodes

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
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        state1 = torch.reshape(state1, (batch_size, self.coarse_nodes, -1))
        state2 = torch.reshape(state2, (batch_size, self.super_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        coarse_input = inputs.permute(1, 2, 0)
        coarse_input = coarse_input.reshape(self.num_nodes, -1)
        coarse_input = torch.mm(self.afc_mx.float(), coarse_input.float())

        # super_input
        acs_mx = torch.softmax(self.assMatrix, dim=-1)
        acs_mxt = torch.transpose(acs_mx, 0, 1)
        super_input = torch.mm(acs_mxt.float(), coarse_input.float())
        # super_input = torch.mm(self.afc_mx.float(), super_input.float())
        # print("super_input.shape is{}".format(super_input.shape))

        coarse_input = torch.reshape(coarse_input, (batch_size, self.coarse_nodes, -1))
        super_input = torch.reshape(super_input, (batch_size, self.super_nodes, -1))

        # print("super_input.shape is{}".format(super_input.shape))

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim+gru_units, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1))

        x0fc = torch.cat([coarse_input, state1], dim=2)
        x0fc = x0fc.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0fc = x0fc.reshape(shape=(self.coarse_nodes, -1))  # (coarse_nodes, batch*dim)

        x0cs = torch.cat([super_input, state2], dim=2)
        # print("x0cs {}: ".format(x0cs.shape))
        # print("state2 {}: ".format(state2.shape))
        x0cs = x0cs.permute(1, 2, 0)  # (num_nodes, dim, batch)
        # print("x0cs {}: ".format(x0cs.shape))
        x0cs = x0cs.reshape(shape=(self.super_nodes, -1))  # (super_nodes, batch*dim)
        # print("x0cs {}: ".format(x0cs.shape))

        # adj_mx2 = np.transpose(adj_mx2.detach().numpy(), (1, 2, 0))
        # adj_mx2 = adj_mx2.reshape(self.super_nodes, self.super_nodes)
        # print(adj_mx2.shape)
        # adj_mx2 = adj_mx2.cpu().numpy()
        #
        #
        # support = calculate_normalized_laplacian(adj_mx2)
        # adj2 = self._build_sparse_matrix(support,device=self._device)

        adj2 = adj_mx2


        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())  # A * X
        x1fc = torch.mm(self.afc_mxt.float(), x0fc.float())
        x1cs = torch.mm(self.afc_mxt.float(),(torch.mm(acs_mx.float(), x0cs.float())))

        x1 = x1.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        x1fc = x1fc.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1fc = x1fc.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)
        # print("x1fc {}: ".format(x1fc.shape))


        x1cs = x1cs.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1cs = x1cs.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1cs = x1cs.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)
        # print("x1cs {}: ".format(x1cs.shape))

        weights = self.weigts[(input_size, output_size)]
        weights1 = self.weigts1[(input_size, output_size)]
        weights2 = self.weigts2[(input_size, output_size)]
        # print("weigts2 {}: ".format(weights2.shape))

        x1 = torch.matmul(x1, weights) + self.n1 * torch.sigmoid(
            torch.matmul(x1fc, weights1)) + self.n3*torch.sigmoid(
            torch.matmul(x1cs, weights2)) # (batch_size * self.num_nodes, output_size)

        biases = self.biases[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        return x1

    def _gcsc(self, inputs, state, state1,adj2 ,output_size,bias_start=0.0):
        """
        2nd layer GCN for super nodes

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

        state = torch.reshape(state, (batch_size, self.super_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        state1 = torch.reshape(state1, (batch_size, self.coarse_nodes, -1))
        # inputs_and_state = torch.cat([inputs, state], dim=2)


        coarse_input = inputs.permute(1, 2, 0)
        coarse_input = coarse_input.reshape(self.num_nodes, -1)
        coarse_input = torch.mm(self.afc_mx.float(), coarse_input.float())

        #super_input
        acs_mx = torch.softmax(self.assMatrix, dim=-1)
        acs_mxt = torch.transpose(acs_mx,0,1)
        super_input = torch.mm(acs_mxt.float(),coarse_input.float())
        # coarse_input = torch.mm(acs_mxt.float(), coarse_input.float())


        coarse_input = torch.reshape(coarse_input, (batch_size, self.coarse_nodes, -1))
        super_input = torch.reshape(super_input, (batch_size, self.super_nodes, -1))
        # print(coarse_input.shape)

        x = torch.cat([super_input, state], dim=2)
        input_size = x.shape[2]
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim+gru_units, batch)

        x0 = x0.reshape(shape=(self.super_nodes, -1))

        x0fc = torch.cat([coarse_input, state1], dim=2)

        x0fc = x0fc.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0fc = x0fc.reshape(shape=(self.coarse_nodes, -1))  # (coarse_nodes, batch*dim)
        x0fc = torch.mm(acs_mxt.float(), x0fc.float())

        x1fc = x0fc
        # adj2
        x1 = torch.mm(adj2.float(),x0.float())  # A * X

        x1 = x1.reshape(shape=(self.super_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        x1fc = x1fc.reshape(shape=(self.super_nodes, input_size, batch_size))
        x1fc = x1fc.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        weights = self.weigts2[(input_size, output_size)]
        weights1 = self.weigts1[(input_size, output_size)]
        x1 = torch.matmul(x1, weights) + self.n1 * torch.sigmoid(
            torch.matmul(x1fc, weights1))  # (batch_size * self.super_nodes, output_size)

        biases = self.biases2[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.super_nodes, output_size))
        return x1

    def _g2c(self, inputs, state, state1,output_size, bias_start=0.0):
        """
        2nd layer GCN for the coarse nodes

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
        inputs = inputs.permute(1, 2, 0)
        fine_input = inputs

        inputs = inputs.reshape(self.num_nodes, -1)
        inputs = torch.mm(self.afc_mx.float(), inputs.float())
        inputs = torch.reshape(inputs, (batch_size, self.coarse_nodes, -1))

        state = torch.reshape(state, (batch_size, self.coarse_nodes, -1))  # (batch, self.num_nodes, self.gru_units)

        state1 = torch.reshape(state1, (batch_size, self.num_nodes, -1))

        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        # fine_input = fine_input.reshape(self.coarse_nodes, -1)
        # fine_input = torch.mm(self.afc_mx.float(), fine_input.float())
        fine_input = torch.reshape(fine_input, (batch_size, self.num_nodes, -1))
        # print(coarse_input.shape)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim+gru_units, batch)
        x0 = x0.reshape(shape=(self.coarse_nodes, -1))

        x0fc = torch.cat([fine_input, state1], dim=2)
        x0fc = x0fc.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0fc = x0fc.reshape(shape=(self.num_nodes, -1))  # (coarse_nodes, batch*dim)
        x1fc = torch.mm(self.afc_mx.float(), x0fc)

        x1 = torch.sparse.mm(self.normalized_adj1.float(), x0.float())  # A * X

        x1 = x1.reshape(shape=(self.coarse_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        x1fc = x1fc.reshape(shape=(self.coarse_nodes, input_size, batch_size))
        x1fc = x1fc.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        weights = self.weigts1[(input_size, output_size)]
        weights1 = self.weigts[(input_size, output_size)]
        x1 = torch.matmul(x1, weights) + self.n4 * torch.sigmoid(
            torch.matmul(x1fc, weights1))  # (batch_size * self.coarse_nodes, output_size)

        biases = self.biases1[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.coarse_nodes, output_size))
        return x1

    def _gc(self, inputs, state, state1, state2, output_size, bias_start=0.0):
        """
        1st layer GCN for both coarse fine and super nodes

        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            state1: (batch, self.coarse_nodes * self.gru_units)
            state2: (batch, self.super_nodes * self.gru_units)
            output_size:
            bias_start:

        Returns:
            torch.tensor: (B, num_nodes , output_size)
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.dim)
        # coarse = torch.
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        state1 = torch.reshape(state1, (batch_size, self.coarse_nodes, -1))
        state2 = torch.reshape(state2, (batch_size, self.super_nodes, -1))
        # print("state.shape is {}".format(state.shape))
        # print("state1.shape is {}".format(state1.shape))
        # print("state2.shape is {}".format(state2.shape))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]  # input_size=inputs.shape[2]

        x = inputs_and_state  # inputs #
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1))  # (num_nodes, batch*dim)
        # d = np.array(self.afc_mx.sum(1))
        input = inputs.permute(1, 2, 0)
        # print("input.shape IS {}".format(input.shape))
        coarse_input = input.reshape(self.num_nodes, -1)
        coarse_input = torch.mm(self.afc_mx.float(), coarse_input.float())

        # print("COARSE input.shape IS {}".format(coarse_input.shape))

        # super_input, adj_mx2, loss1, loss2 = self.dense_diff_pool(x=coarse_input, adj=torch.tensor(self.adj_mx1,device=self._device),
                                                            #  s=self.assMatrix.to(device=self._device))
        # acs_mx_t=torch.transpose(self.assMatrix, 0, 1)
        # print("acs_mt shape is {}".format(acs_mx_t.shape))
        # super_input = acs_mx_t@coarse_input #torch.mm(acs_mx_t.float(),coarse_input.float())
        # print("super input shape: {}".format(super_input.shape))
        # print("diff pool done")
        # print(super_input1.shape)
        # super_input1 = torch.squeeze(super_input1)
        # print(super_input1.shape)
        # # print("squeeze")
        # # print(super_input.shape)

        coarse_input = torch.reshape(coarse_input, (batch_size, self.coarse_nodes, -1))
        # super_input = torch.reshape(super_input, (batch_size, self.super_nodes, -1))
        # print("super input shape: {}".format(super_input.shape))
        x0fc = torch.cat([coarse_input, state1], dim=2)

        x0cs, adj_mx2, loss1, loss2 = self.dense_diff_pool(x=x0fc, adj=torch.tensor(self.adj_mx1,device=self._device),
                                                             s=self.assMatrix.to(device=self._device))
 
        x0fc = x0fc.permute(1, 2, 0)  # (num_nodes, dim, batch)
        x0fc = x0fc.reshape(shape=(self.coarse_nodes, -1))  # (coarse_nodes, batch*dim)
        # x0fc = torch.matmul(self.afc_mx.float(), x0fc.float()) # (coarse_nodes, batch*dim)
        # x0fc = x0fc/d

        # x0cs = torch.cat([super_input, state2], dim=2)
        # print("x0cs shape: {}".format(x0cs.shape))
        x0cs = x0cs.permute(1, 2, 0)  # (num_nodes, dim, batch)
        # print("x0cs shape: {}".format(x0cs.shape))
        x0cs = x0cs.reshape(shape=(self.super_nodes, -1))  # (super_nodes, batch*dim)
        # print(x0cs.shape)
        adj_mx2 = np.transpose(adj_mx2.cpu().detach().numpy(), (1, 2, 0))
        adj_mx2 = adj_mx2.reshape(self.super_nodes,self.super_nodes)
        # print(adj_mx2.shape)
        support = calculate_normalized_laplacian(adj_mx2)
        adj2 = self._build_sparse_matrix(support, device=self._device)

        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())  # A * X H0
        x1fc = torch.sparse.mm(self.normalized_adj1.float(), x0fc.float())  # A * Xc HC0
        x1cs = torch.mm(adj2.float(), x0cs.float())
        # print(x1cs.shape)
        # x1 = x1 + self.n1*torch.sigmoid(torch.matmul(self.afc_mxt.float(), x0fc.float())) #H1
        # x1fc = x1fc + self.n2*torch.sigmoid(torch.matmul(self.afc_mx.float(),x1.float())) #H1C
        x1 = x1.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1fc = x1fc.reshape(shape=(self.coarse_nodes, input_size, batch_size))
        x1cs = x1cs.reshape(shape=(self.super_nodes, input_size, batch_size))
        # print(x1cs.shape)
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1fc = x1fc.permute(2, 0, 1)
        x1cs = x1cs.permute(2, 0, 1)

        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)
        x1fc = x1fc.reshape(shape=(-1, input_size))
        x1cs = x1cs.reshape(shape=(-1, input_size))

        weights = self.weigts[(input_size, output_size)]
        weights1 = self.weigts1[(input_size, output_size)]
        weights2 = self.weigts2[(input_size, output_size)]

        x1 = torch.matmul(x1,
                          weights)  # +self.n1*torch.sigmoid(torch.matmul(torch.matmul(self.afc_mxt.float(), x1fc.float()),weights1)) #H1
        x1fc = torch.matmul(x1fc,
                            weights1)  # +self.n2*torch.sigmoid(torch.matmul(torch.matmul(self.afc_mx.float(),x1.float()),weights)) #H1C
        x1cs = torch.matmul(x1cs, weights2)

        biases = self.biases[(output_size,)]
        biases1 = self.biases1[(output_size,)]
        biases2 = self.biases2[(output_size,)]

        x1 += biases
        x1fc += biases1
        x1cs += biases2

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        x1fc = x1fc.reshape(shape=(batch_size, self.coarse_nodes, output_size))
        x1cs = x1cs.reshape(shape=(batch_size, self.super_nodes, output_size))
        return x1, x1fc, x1cs, adj2


class TGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.adj_mx1 = data_feature.get('adj_mx1')
        self.coarse_nodes = data_feature.get('coarse_nodes')
        self.super_nodes = data_feature.get('super_nodes',10)
        self.afc_mx = data_feature.get('afc_mx')  # Add the afc matrix in the data feature
        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda', 0.01)
        self.n1 = config.get('n1', 0.2)
        self.n2 = config.get('n2', 0.8)

        super().__init__(config, data_feature)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # -------------------构造模型-----------------------------
        self.tgcn_model = TGCNCell(self.gru_units, self.afc_mx, self.adj_mx, self.adj_mx1, self.num_nodes,
                                   self.coarse_nodes,10 ,self.device, self.n1, self.n2, self.input_dim)
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
        state1 = torch.zeros(batch_size, self.coarse_nodes * self.gru_units).to(self.device)
        state2 = torch.zeros(batch_size, self.super_nodes * self.gru_units).to(self.device)
        for t in range(input_window):
            state, state1,state2,adj2,acs = self.tgcn_model(inputs[t], state, state1, state2)

        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        output = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        output = output.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        output = output.permute(0, 2, 1, 3)

        state1 = state1.view(batch_size, self.coarse_nodes, self.gru_units)
        output1 = self.output_model(state1)
        output1 = output1.view(batch_size, self.coarse_nodes, self.output_window, self.output_dim)
        output1 = output1.permute(0, 2, 1, 3)

        state2 = state2.view(batch_size, self.super_nodes, self.gru_units)
        output2 = self.output_model(state2)
        output2 = output2.view(batch_size, self.super_nodes, self.output_window, self.output_dim)
        output2 = output2.permute(0, 2, 1, 3)
        return output, output1, output2,adj2,acs

    def calculate_loss(self, batch):
        lam = self.lam
        lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())

        labels = batch['y']
        y_predicted, cy_predicted,sy_predicted,adj2,acs= self.predict(batch)

        y_true = self._scaler.inverse_transform(labels[..., :self.output_dim])
        # print(y_true.shape)
        batch_size, input_window, num_nodes, input_dim = y_true.shape
        cy_true = torch.reshape(y_true, (batch_size, self.num_nodes, -1))
        cy_true = cy_true.permute(1, 2, 0)
        cy_true = cy_true.reshape(num_nodes, -1)
        # sy_true = cy_true
        afc_mx = torch.tensor(self.afc_mx.T, device=self.device)
        acs_mx = torch.transpose(acs,0,1)
        cy_true = torch.mm(afc_mx.float(), cy_true.float())
        sy_true = torch.mm(acs_mx.float(), cy_true.float())
        cy_true = cy_true.reshape(batch_size, input_window, self.coarse_nodes, input_dim)
        sy_true = sy_true.reshape(batch_size, input_window, self.super_nodes, input_dim)

        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        cy_predicted = self._scaler.inverse_transform(cy_predicted[..., :self.output_dim])
        sy_predicted = self._scaler.inverse_transform(sy_predicted[..., :self.output_dim])

        batch_size, input_window, num_nodes, input_dim = y_predicted.shape
        y_predicted_c = torch.reshape(y_true, (batch_size, self.num_nodes, -1))
        y_predicted_c = y_predicted_c.permute(1, 2, 0)
        y_predicted_c = y_predicted_c.reshape(num_nodes, -1)
        y_predicted_c = torch.mm(afc_mx.float(), y_predicted_c.float())
        y_predicted_c = y_predicted_c.reshape(batch_size, input_window, self.coarse_nodes, input_dim)

        loss1,loss2 = self.assign_loss(torch.tensor(self.adj_mx1,device=self.device),acs)

        self._logger.info('link_loss: {0} ent_loss:{1}'.format(loss1,loss2))

        a = torch.mean(torch.norm(y_true - y_predicted) ** 2 / 2)/y_predicted.numel()
        b = torch.mean(torch.norm(cy_true - cy_predicted) ** 2 / 2)/y_predicted.numel()
        c = torch.mean(torch.norm(sy_true - sy_predicted) ** 2 / 2)/y_predicted.numel()
        d = torch.mean(torch.norm(y_predicted_c - cy_predicted) ** 2 / 2)/y_predicted.numel()

        loss = 2 * a + b + c + lam * d
        # loss /= /y_predicted.numel()
        loss = loss + loss1 + loss2

        self._logger.info('fine_loss: {0} coarse_loss:{1} superloss:{2} align_loss:{3}'.format(a,b,c,d))

        # return loss.masked_mae_torch(y_predicted, y_true, 0)
        return loss

    def assign_loss(self,adj,s):
        s = s.unsqueeze(0)
        adj = torch.tensor(adj)
        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / adj.numel()

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        return link_loss,ent_loss

    def predict(self, batch):
        return self.forward(batch)
