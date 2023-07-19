import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

# torch.autograd.set_detect_anomaly(True)
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
    def __init__(self, c_in, c_out, dropout, support_len=3, order=1):
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
            if self.order >= 2:
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class HGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout,afc,super_nodes,coarse_nodes,device,n1=0.8,n2=0.2,n3=0.2,n4=0.2,n5=0.2 ,support_len=3, order=1):
        super(HGCN, self).__init__()
        self.gcn = GCN(c_in, c_out, dropout, support_len,order)
        
        self.coarse_nodes = coarse_nodes
        self.super_nodes = super_nodes
        self._device = device
        self.afc = afc.to(device=self._device) 

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5

        # self.nconv = NConv()
        # self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support,support_c,acs): #,support_s
        # out = [x]
        # cout = [self.afc.detach().t().float() @ x]
        # sout = [acs.t().float() @ self.afc.detach().t().float() @ x]
        hf = self.gcn(x,support)
        hc = self.gcn(self.afc.t().float() @ x, support_c)
        sxg = acs.t().float() @ self.afc.t().float() @ x
        # print(sxg.size())
        hs = sxg.permute(2,1,0,3)
        hs = torch.reshape(hs,(self.super_nodes,-1))
        # print(hs.size())
        hst = sxg.permute(3,0,1,2)
        hst = torch.reshape(hst,(-1,self.super_nodes))
        # print(hst.size())
        as_mat = (F.relu(hs@hst - 0.5)).detach().cpu().numpy()
        # as_mat = (torch.sigmoid(hs@hst)).detach().cpu().numpy() #change to sigmoid
        
        adj_mxs = [asym_adj(as_mat), asym_adj(np.transpose(as_mat))]
        # print("as_mat")
        # print(adj_mxs[0])
        supports_s = [torch.tensor(i).to(self._device) for i in adj_mxs]
        supports_s = [F.softmax(i,dim=-1).to(self._device) for i in supports_s]
        hs = self.gcn(sxg, supports_s)
        hc = hc + self.n1*F.relu(acs.float()@hs)
        hf = hf+self.n2*F.relu(self.afc.float()@hc) #+self.n2*F.relu(self.afc.float()@acs.float()@hs)
        hc = hc+self.n3*F.relu(self.afc.t().float()@hf)
        hs = hs+self.n4*F.relu(acs.t().float()@hc)

        return hf,hc,hs#,ac_hat#,as_hat


class HIESTBase(AbstractTrafficStateModel):
    def __init__(self, config, data_feature,source):
        if source == True:
            self.adj_mx = data_feature.get('source_adj_mx')
            self.afc = data_feature.get('source_afc_mx')
            self.num_nodes = data_feature.get('source_num_nodes', 1)
            self.coarse_nodes = len(self.afc[0])
            self.super_nodes = config.get('super_nodes', 10)
            print('super_nodes '+str(self.super_nodes))
        else:
            self.adj_mx = data_feature.get('target_adj_mx')
            self.afc = data_feature.get('target_afc_mx')
            self.num_nodes = data_feature.get('target_num_nodes', 1)
            self.coarse_nodes = len(self.afc[0])
            self.super_nodes = config.get('super_nodes', 10)
            print('super_nodes '+str(self.super_nodes))
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
        self.order = config.get('order', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.afc_mx = torch.from_numpy(self.afc).detach().to(device=self.device)
        self.af = torch.from_numpy(self.adj_mx).detach().to(device=self.device)
        self.adj_mx1 = torch.from_numpy(self.afc.T @ self.adj_mx @ self.afc).detach().to(device=self.device)
        self.memory_dim = config.get('memory_dim',12)

        self.n1 = config.get('n1',1)
        self.n2 = config.get('n2',1)
        self.n3 = config.get('n3',1)
        self.n4 = config.get('n4',1)
        self.n5 = config.get('n5',1)

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
        self.bns = nn.ModuleList()

        self.gconv = nn.ModuleList()
        self.acs = torch.nn.init.kaiming_uniform_(torch.nn.Parameter(torch.empty((self.coarse_nodes, self.super_nodes),requires_grad=True).to(device=self.device)))

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        

        self.cal_adj(self.adjtype)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        self.supports_c = [(self.afc_mx.t().float() @ i.clone().detach() @ self.afc_mx.float()).to(self.device) for i in self.supports]
        self.supports_c = [F.softmax(i,dim=-1).to(self.device) for i in self.supports_c]
        

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

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
                self.bns.append(nn.BatchNorm2d(self.residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(HGCN(self.dilation_channels, self.residual_channels,self.dropout,self.afc_mx,self.super_nodes,self.coarse_nodes,device=self.device
                                          ,n1=self.n1,n2=self.n2,n3=self.n3,n4=self.n4,n5=self.n5 ,support_len=self.supports_len,order=self.order))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,#+self.memory_dim,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

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

        


class HIESTDistill(HIESTBase):
    def __init__(self, config, data_feature,source):
        if source == False:
            super().__init__(config, data_feature,source=False)
            self.teacher = HIESTBase(config, data_feature,source=False)
            self.student = HIESTBase(config, data_feature,source=False)


    def forward(self, batch):
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)
        acs = F.softmax(F.relu(self.acs), dim=1) #-0.5

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs

        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        tx = x
        skip = 0
        t_skip = 0
        stu_skip_r = 0
        stu_skip_g = 0
        tea_skip_r = 0
        tea_skip_g = 0

        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ---S-|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|      |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # residual_c = sxr
            # residual_s = sxg
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

            t_residual = tx
            # residual_c = sxr
            # residual_s = sxg
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            t_filter = self.filter_convs[i](t_residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            t_filter = torch.tanh(t_filter)
            t_gate = self.gate_convs[i](t_residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            t_gate = torch.sigmoid(t_gate)
            tx = t_filter * t_gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            t_s = tx
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            t_s = self.skip_convs[i](t_s)

            # Student
            x,sxr,sxg = self.student.gconv[i](x,self.supports,self.supports_c,acs)
            s_xr = self.skip_convs[i](sxr)
            s_xg = self.skip_convs[i](sxg)
            # print('x.size()')
            # print(x.size())
            # print(sxr.size())
            # print(sxg.size())

            try:
                skip = skip[:, :, :, -s.size(3):]
                stu_skip_r = stu_skip_r[:, :, :, -s_xr.size(3):]
                stu_skip_g = stu_skip_g[:, :, :, -s_xg.size(3):]
                
            except(Exception):
                skip = 0
                stu_skip_r = 0
                stu_skip_g = 0

            skip = s + skip
            # print('skip')
            # print(skip.size())

            stu_skip_r = s_xr + stu_skip_r

            # print('stu_skip_r')
            # print(stu_skip_r.size())
            stu_skip_g = s_xg + stu_skip_g
            
            x = x + residual[:, :, :, -x.size(3):]
            # print('before bc')
            # print(x.size())
            x = self.bn[i](x)
            # print('after bc')
            # print(x.size())
            # sxr = self.bn[i](sxr)
            # print('before bnc')
            # print(sxr.size())
            sxr = self.bnc[i](sxr) #student.
            sxg = self.bns[i](sxg)

            # Teacher
            tx,txr,txg = self.student.gconv[i](tx,self.supports,self.supports_c,acs)
            t_xr = self.skip_convs[i](txr)
            t_xg = self.skip_convs[i](txg)
            

            try:
                t_skip = t_skip[:, :, :, -t_s.size(3):]
                tea_skip_r = tea_skip_r[:, :, :, -t_xr.size(3):]
                tea_skip_g = tea_skip_g[:, :, :, -t_xg.size(3):]
                
            except(Exception):
                t_skip = 0
                tea_skip_r = 0
                tea_skip_g = 0

            t_skip = t_s + t_skip
            tea_skip_r = txr + tea_skip_r
            tea_skip_g = txg + tea_skip_g
            
            tx = tx + t_residual[:, :, :, -tx.size(3):]
            tx = self.bn[i](tx) #teacher.
            # sxr = self.bn[i](sxr)
            txr = self.bnc[i](txr)
            txg = self.bns[i](txg)

        x = F.relu(skip)
        # print(x.size())
        # print('t_skip')
        # print(t_skip.size())
        tx = F.relu(t_skip)
        # print('stu_skip_r')
        # print(stu_skip_r.size())
        sxr = F.relu(stu_skip_r)
        txr = F.relu(tea_skip_r)
        fc = sxr
        sxg = F.relu(stu_skip_g)
        txg = F.relu(tea_skip_g)
        fs = sxg
        # Feature representation is ready

        # (batch_size, skip_channels, num_nodes, self.output_dim)
        # xm = x.permute(0,3,2,1)
        # (batch_size, self.output_dim, num_nodes,skip_channels )
        # xm = torch.reshape(xm,(self.super_nodes,-1))
        # attention_vals = self.attention(xm, self.memory_weights)
        # attention_vals (batch_size, self.output_dim, num_nodes,super_nodes )
        # attentive_cluster_reps = ((attention_vals@self.memory_weights['M'])@self.memory_weights['fc']).permute(0,3,2,1)
        # (batch_size, self.output_dim, num_nodes,memory_dim ) .permute(0,3,2,1)

        # x = torch.cat((x, attentive_cluster_reps), dim=1)
        # Go through the end_cov to get the predict result
        x = F.relu(self.end_conv_1(x))
        sxr = F.relu(self.end_conv_1(sxr)) 
        sxg = F.relu(self.end_conv_1(sxg))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        sxr = self.end_conv_2(sxr)
        sxg = self.end_conv_2(sxg)
        # link_loss, ent_loss = self.assLoss(self.afc.T @ self.adj_mx @ self.afc,learned_acsmx[-1])
        # self._logger.info('link_loss: %.4f'%link_loss)
        # self._logger.info('ent_loss: %.4f'%(ent_loss))
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x,fc,fs,sxr,sxg#,ac_hat,as_hat #,learned_acsmx[-1]
    
    def assLoss(self,adj,s):
        # adj = torch.from_numpy(adj)
        adj = adj.to(device=self.device)
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = F.softmax(s,dim=-1).detach()
        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        # out_adj = out_adj.numpy()

        return link_loss, ent_loss
    
    def othLoss(self,hc):
        # hc = (batch_size, end_channels, num_nodes, self.output_dim)
        # hc = hc.squeeze(3)
        hc = hc.permute(2,0,1,3)
        hc = torch.reshape(hc,(self.super_nodes,-1))
        # print('hc.size')
        # print(hc.size())
        tmpLoss = 0
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(self.super_nodes-1):
            for j in range(i+1,self.super_nodes):
                # print(hc[i].size())
                tmpLoss += cos(hc[i],hc[j])

        tmpLoss = tmpLoss/(self.super_nodes*(self.super_nodes-1)/2)

        return tmpLoss
    
    def construct_memory(self):
        memory = {}
        # memory_weights['M'] = tf.get_variable('mem', [self.cluster_num, self.memory_dim],
        #                                       initializer=init, dtype=dtype)
        # memory_weights['Wa'] = tf.get_variable('att', [self.dim_lstm_hidden, self.memory_dim],
        #                                        initializer=init, dtype=dtype)
        # memory_weights['fc'] = tf.get_variable('mem_fc', [self.memory_dim, self.memory_dim],
        #                                        initializer=init, dtype=dtype)
        memory['M'] = torch.nn.init.kaiming_uniform_(torch.nn.Parameter(torch.empty((self.super_nodes, self.memory_dim),requires_grad=True).to(device=self.device)))
        memory['Wa'] = torch.nn.init.kaiming_uniform_(torch.nn.Parameter(torch.empty((self.skip_channels, self.memory_dim),requires_grad=True).to(device=self.device)))
        memory['fc'] = torch.nn.init.kaiming_uniform_(torch.nn.Parameter(torch.empty((self.memory_dim, self.memory_dim),requires_grad=True).to(device=self.device)))
        return memory
    
    def attention(self, inp, weights):
        # inp (batch_size, self.output_dim, num_nodes,skip_channels )
        # print(inp.size())
        score =torch.einsum('bons,sk,km->bonm', (inp, weights['Wa'],weights['M'].transpose(0,1)))
        # score = ((inp@ weights['Wa'])@(weights['M']))
        # score (batch_size, self.output_dim, num_nodes,super_nodes )
        return F.softmax(score,dim=-1)



    

    def calculate_loss(self, batch):
        y_true = batch['y']
        # y_predicted,cy_predicted = self.predict(batch)
        y_predicted,cy_predicted,hs,cy,sy = self.predict(batch)
        # print('y_true', y_true.shape) ,cy_predicted,sy_predicted,acs
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # ac = self.adj_mx1
        acs = F.softmax(F.relu(self.acs), dim=1) #-0.5
        # print("acs")
        # print(acs)
        nf = self.afc_mx.float() @ cy_predicted
        hf = nf.permute(2,1,0,3)
        hf = torch.reshape(hf,(self.num_nodes,-1))
        # print(hc.size())
        hft = nf.permute(3,0,1,2)
        hft = torch.reshape(hft,(-1,self.num_nodes))
        # print(hct.size())
        af_hat = torch.sigmoid(hf.float() @ hft.float())

        nc = acs.float()@hs
        # print(nc.size())
        hc = nc.permute(2,1,0,3)
        hc = torch.reshape(hc,(self.coarse_nodes,-1))
        # print(hc.size())
        hct = nc.permute(3,0,1,2)
        hct = torch.reshape(hct,(-1,self.coarse_nodes))
        # print(hct.size())
        ac_hat = torch.sigmoid(hc.float() @ hct.float())
        # print(ac_hat.size())
        # print(self.adj_mx1.size())
        
        cy_true = self.afc_mx.detach().T.float() @ y_true
        sy_true = self.acs.detach().t().float() @ cy_true #F.softmax(self.acs.detach(),dim=-1)


        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        cy = self._scaler.inverse_transform(cy[..., :self.output_dim])
        sy = self._scaler.inverse_transform(sy[..., :self.output_dim])
        # link_loss, ent_loss = self.assLoss(self.supports_c[0],self.acs.clone())#F.softmax(self.acs,dim=-1)
        loss_f = loss.masked_mae_torch(y_predicted, y_true, 0)
        # loss_f1 = loss.masked_rmse_torch(y_predicted, y_true, 0)
        # x1 = torch.reshape(ac_hat, (-1,))
        # x2 = torch.reshape(self.adj_mx1.long(), (-1,))
        loss_bce0 = F.binary_cross_entropy(af_hat,self.af,reduction='mean')
        loss_bce = F.binary_cross_entropy(ac_hat,self.adj_mx1.float(),reduction='mean')
        othloss = self.othLoss(hs)
        # othloss = torch.abs(self.othLoss(hs))
        loss_c = loss.masked_mae_torch(cy, cy_true, 0)
        loss_s = loss.masked_mae_torch(sy, sy_true, 0)
        loss_or = loss.masked_mae_torch(self.afc_mx.detach().T.float() @ y_predicted,cy,0)
        loss_rg = loss.masked_mae_torch(self.acs.detach().t().float() @ cy,sy,0)
        # train_loss = loss_f + loss_c + 0.01*loss_s
        self._logger.info('loss_or: {0} loss_rg:{1}'.format(loss_or,loss_rg))
        # self._logger.info('fine_loss: {0} bce_loss:{1} oth_loss:{2} '.format(loss_f,loss_bce,othloss))
        # self._logger.info('fine_loss: {0} bce_loss:{1} bce_loss0:{2} othLoss:{3}'.format(loss_f,loss_bce,loss_bce0,othloss))
        self._logger.info('fine_loss: {0} regional_loss:{1} global_loss:{2}'.format(loss_f,loss_c,loss_s))
        return loss_f + 0.01*loss_bce + 0.01*loss_bce0 + 0.1*othloss + 0.1*loss_c  + 0.01*loss_or + 0.01*loss_rg #+ loss_f1 # + 0.01*loss_bce0+ loss_c #+ 0.001*(loss_s)#+link_loss+ent_loss)++ 0.01*loss_s

    def predict(self, batch):
        return self.forward(batch)
