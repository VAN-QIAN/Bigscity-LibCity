import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class HGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout,afc,acs,super_nodes,coarse_nodes,device,n1=0.8,n2=0.2,n3=0.2,n4=0.2,n5=0.2 ,support_len=3, order=2):
        super(HGCN, self).__init__()
        # c_in = (order * support_len + 1) * c_in
        self.fgcn = GCN(c_in, c_out, dropout, support_len)
        self.cgcn = GCN(c_in, c_out, dropout, support_len)
        # self.sgcn = GCN(c_in, c_out, dropout, support_len)
        
        self.coarse_nodes = coarse_nodes
        self.super_nodes = super_nodes
        self._device = device
        self.afc = torch.from_numpy(afc).detach().to(device=self._device)

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5

        # self.nconv = NConv()
        # self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
        # self.init_params()
        # acs = torch.softmax(acs,dim=-1)

    def init_params(self):
        assMatrix = torch.nn.Parameter(torch.ones((self.coarse_nodes, self.super_nodes), device=self._device))
        self.register_parameter(name='assMatrix', param=assMatrix)
        self.acs =  torch.softmax(assMatrix, dim=-1)

    def assLoss(self,adj,s):
        adj = torch.from_numpy(adj)
        adj = adj.to(device=self._device)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        # out_adj = out_adj.numpy()

        return link_loss, ent_loss

    def forward(self, x, support,acs):
        out = [x]
        cout = [self.afc.detach().t().float() @ x]
        # sout = [acs.t().float() @ self.afc.detach().t().float() @ x]
        # h0 = self.fgcn(x,support)
        # h0c = self.cgcn(x, support)
        # h0s = self.sgcn(x, support)
        for a in support:
            # fine-grained
            # ac = self.afc.trans @ a @ self.afc
            # xc = self.afc @ x
            # print(a.shape)
            # print(x.shape)
            x1 = self.fgcn.nconv(x,a) # +self.n1 * self.cgcn(xc,ac)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.fgcn.nconv(x1, a)
                out.append(x2)
                x1 = x2
        hf = torch.cat(out, dim=1)
        hf = self.fgcn.mlp(hf)
        hf = F.dropout(hf, self.dropout, training=self.training)

        for a in support:
            # coarse-grained
            # print(a.shape)
            ac = self.afc.detach().t().float() @ a @ self.afc.detach().float()
            # print('ac '+str(ac.shape))
            # print('afc_t '+str(self.afc.t().shape))
            # print(x.shape)
            xc = self.afc.detach().t().float() @ x
            # print('xc '+str(xc.shape))
            x1 = self.cgcn.nconv(xc,ac)
            # print('x1 '+str(x1.shape))
            cout.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.cgcn.nconv(x1, ac)
                # print('x2 '+str(x2.shape))
                cout.append(x2)
                x1 = x2
        hc = torch.cat(cout, dim=1)
        hc= self.cgcn.mlp(hc)
        hc = F.dropout(hc, self.dropout, training=self.training)

        # for a in support:
        #     # super-grained
        #     # ac = self.afc.t() @ a @ self.afc
        #     asc = acs.t().float() @ self.afc.detach().t().float() @ a @ self.afc.detach().float() @ acs.float()
        #     xs = acs.t().float() @ self.afc.detach().t().float() @ x
        #     x1 = self.fgcn.nconv(xs,asc)
        #     sout.append(x1)
        #     for k in range(2, self.order + 1):
        #         x2 = self.fgcn.nconv(x1, asc)
        #         sout.append(x2)
        #         x1 = x2
        #     # link_loss,ent_loss = self.assLoss(ac,acs)
        #     # self._logger.info('link_loss: %.4f'%link_loss)
        #     # self._logger.info('link_loss: %.4f'%ent_loss)
        # hs = torch.cat(sout, dim=1)
        # hs = self.sgcn.mlp(hs)
        # hs = F.dropout(hs, self.dropout, training=self.training)

        # print('hf '+str(hf.shape))
        # print('hc '+str(hc.shape))
        # print('hs '+str(hs.shape))

        hf = hf+self.n1*torch.sigmoid(self.afc.detach().float()@hc) #+self.n2*torch.sigmoid(self.afc.float()@acs.float()@hs)
        hc = hc+self.n3*torch.sigmoid(self.afc.detach().t().float()@hf)
        # hs = hs+self.n5*torch.sigmoid(acs.t().float()@hc) #+self.n4*torch.sigmoid(acs.t().float()@self.afc.t().float()@hf)

        return hf,hc#,hs#,acs


class GWNET(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.afc = data_feature.get('afc_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.coarse_nodes = len(self.afc[0])
        self.super_nodes = data_feature.get('super_nodes', 10)
        self.feature_dim = data_feature.get('feature_dim', 2)
        super().__init__(config, data_feature)

        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.gcn_bool = config.get('gcn_bool', True)
        self.addaptadj = config.get('addaptadj', True)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.randomadj = config.get('randomadj', True)
        self.aptonly = config.get('aptonly', True)
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.residual_channels = config.get('residual_channels', self.nhid)
        self.dilation_channels = config.get('dilation_channels', self.nhid)
        self.skip_channels = config.get('skip_channels', self.nhid * 8)
        self.end_channels = config.get('end_channels', self.nhid * 16)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.afc_mx = torch.from_numpy(self.afc).to(device=self.device)

        self.n1 = config.get('n1',0.8)
        self.n2 = config.get('n2',0.1)
        self.n3 = config.get('n3',0.8)
        self.n4 = config.get('n4',0.1)
        self.n5 = config.get('n5',0.8)

        self.apt_layer = config.get('apt_layer', True)
        if self.apt_layer:
            self.layers = np.int(
                np.round(np.log((((self.input_window - 1) / (self.blocks * (self.kernel_size - 1))) + 1)) / np.log(2)))
            print('# of layers change to %s' % self.layers)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.bnc = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.acs = torch.nn.Parameter(torch.ones((self.coarse_nodes, self.super_nodes), device=self.device),requires_grad=True)
        # self.register_parameter(name='assMatrix', param=assMatrix)
        # self.acs = assMatrix #torch.softmax(assMatrix.float(), dim=-1)
        # self.acs = self.assMatrix.cpu().detach().numpy()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.cal_adj(self.adjtype)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly:
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1
        # self.supports_len = 1 
        print('supports_len '+str(self.supports_len))
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                self.bnc.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(HGCN(self.dilation_channels, self.residual_channels,self.dropout,self.afc,self.acs,self.super_nodes,self.coarse_nodes,device=self.device
                                          , support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, batch):
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)
        cinputs = self.afc_mx.detach().t().float() @ inputs
        sinputs = self.acs.detach().t().float() @ cinputs

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
            xc = nn.functional.pad(cinputs, (self.receptive_field - in_len, 0, 0, 0))
            xs = nn.functional.pad(sinputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
            xc = cinputs
            xs = sinputs

        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        xc = self.start_conv(xc)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip_c = 0

        xs = self.start_conv(xs)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip_c = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        learned_acsmx = []
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            residual_c = xc
            residual_s = xs
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)

            # course-grained inputs
            filter_c = self.filter_convs[i](residual_c)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter_c = torch.tanh(filter_c)
            gate_c = self.gate_convs[i](residual_c)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate_c = torch.sigmoid(gate_c)
            xc = filter_c * gate_c
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            sc = xc
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            sc = self.skip_convs[i](sc)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip_c = skip_c[:, :, :, -sc.size(3):]
            except(Exception):
                skip_c = 0
            skip_c = sc + skip_c

            # # super-grained inputs
            # filter_s = self.filter_convs[i](residual_s)
            # # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # filter_s = torch.tanh(filter_s)
            # gate_s = self.gate_convs[i](residual_s)
            # # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # gate_s = torch.sigmoid(gate_s)
            # xs = filter_s * gate_s
            # # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # # parametrized skip connection
            # ss = xs
            # # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # ss = self.skip_convs[i](ss)
            # # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            # try:
            #     skip_s = skip_s[:, :, :, -ss.size(3):]
            # except(Exception):
            #     skip_s = 0
            # skip_s = ss + skip_s

            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    self._logger.info('adapt')
                    x = self.gconv[i](x, new_supports)
                else:
                    # self._logger.info('gconv') ,xs
                    x,xc = self.gconv[i](x, self.supports ,torch.softmax(self.acs,dim=-1))
                    # learned_acsmx.append(learned_acs)
                    
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                self._logger.info('res conv')
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            xc = xc + residual_c[:, :, :, -xc.size(3):]
            # xs = xs + residual_s[:, :, :, -xs.size(3):]
            
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
            xc = self.bnc[i](xc)
            # xs = self.bn[i](xs)
        x = F.relu(skip)
        xc = F.relu(skip_c)
        # xs = F.relu(skip_s)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        xc = F.relu(self.end_conv_1(xc))
        # xs = F.relu(self.end_conv_1(xs))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        xc = self.end_conv_2(xc)
        # xs = self.end_conv_2(xs)
        # link_loss, ent_loss = self.assLoss(self.afc.T @ self.adj_mx @ self.afc,learned_acsmx[-1])
        # self._logger.info('link_loss: %.4f'%link_loss)
        # self._logger.info('ent_loss: %.4f'%(ent_loss))
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x,xc#,xs #,learned_acsmx[-1]
    
    def assLoss(self,adj,s):
        adj = torch.from_numpy(adj)
        adj = adj.to(device=self.device)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s,dim=-1)
        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        # out_adj = out_adj.numpy()

        return link_loss, ent_loss

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted,cy_predicted = self.predict(batch)
        # y_predicted,cy_predicted,sy_predicted = self.predict(batch)
        # print('y_true', y_true.shape) ,cy_predicted,sy_predicted,acs
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        cy_true = self.afc_mx.detach().T.float() @ y_true
        # sy_true = torch.softmax(self.acs.detach(),dim=-1).t().float() @ cy_true


        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        cy_predicted = self._scaler.inverse_transform(cy_predicted[..., :self.output_dim])
        # sy_predicted = self._scaler.inverse_transform(sy_predicted[..., :self.output_dim])
        link_loss, ent_loss = self.assLoss(self.afc.T @ self.adj_mx @ self.afc,torch.softmax(self.acs,dim=-1))
        loss_f = loss.masked_mae_torch(y_predicted, y_true, 0)
        loss_c = loss.masked_mae_torch(cy_predicted, cy_true, 0)
        # loss_s = loss.masked_mae_torch(sy_predicted, sy_true, 0)
        # train_loss = loss_f + loss_c + loss_s
        self._logger.info('link_loss: {0} ent_loss:{1}'.format(link_loss,ent_loss))
        self._logger.info('fine_loss: {0} coarse_loss:{1} '.format(loss_f,loss_c))
        # self._logger.info('fine_loss: {0} coarse_loss:{1} super_loss:{2}'.format(loss_f,loss_c,loss_s))
        return loss_f + 0.001*loss_c #+ loss_s

    def predict(self, batch):
        return self.forward(batch)
