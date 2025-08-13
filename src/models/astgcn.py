# src/models/astgcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporal_Attention(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps):
        super(SpatioTemporal_Attention, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_timesteps))
        self.W2 = nn.Parameter(torch.randn(num_features, num_timesteps))
        self.W3 = nn.Parameter(torch.randn(num_features))
        self.Vs = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.bs = nn.Parameter(torch.randn(1, num_nodes, num_nodes))
        
        self.U1 = nn.Parameter(torch.randn(num_nodes))
        self.U2 = nn.Parameter(torch.randn(num_features, num_nodes))
        self.U3 = nn.Parameter(torch.randn(num_features))
        self.Ve = nn.Parameter(torch.randn(num_timesteps, num_timesteps))
        self.be = nn.Parameter(torch.randn(1, num_timesteps, num_timesteps))

    def forward(self, x):
        # x shape: (B, T, N, F)
        lhs = torch.einsum('btf,f->bt', x, self.W3) @ self.W1 # (B, N)
        rhs = torch.einsum('bf,f->b', self.W2, self.W3)
        S = torch.einsum('bn, b, nm->bnm', lhs, rhs, self.Vs) + self.bs
        S_prime = F.softmax(F.relu(S), dim=-1)

        lhs = torch.einsum('bnf,f->bn', x, self.U3) @ self.U1
        rhs = torch.einsum('bf,f->b', self.U2, self.U3)
        E = torch.einsum('bt, b, ts->bts', lhs, rhs, self.Ve) + self.be
        E_prime = F.softmax(F.relu(E), dim=-1)

        x_weighted_by_time = torch.einsum('btnf,bts->bsnf', x, E_prime)
        return x_weighted_by_time, S_prime

class SpatioTemporal_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, K, num_nodes):
        super(SpatioTemporal_Convolution, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.num_nodes = num_nodes
    
    def forward(self, x, adj, spatial_attention):
        # x shape: (B, T, N, C_in)
        batch_size = x.shape[0]
        L_tilde = 2 * adj / torch.lambda_max(adj) - torch.eye(self.num_nodes, device=x.device)
        cheb_polynomials = [torch.eye(self.num_nodes, device=x.device), L_tilde]
        for i in range(2, self.K):
            cheb_polynomials.append(2 * L_tilde @ cheb_polynomials[-1] - cheb_polynomials[-2])
        
        gcn_output = torch.zeros(batch_size, x.shape[1], self.num_nodes, self.out_channels, device=x.device)
        for t in range(x.shape[1]):
            x_t = x[:, t, :, :]
            graph_conv_sum = torch.zeros(batch_size, self.num_nodes, self.out_channels, device=x.device)
            for k in range(self.K):
                T_k = cheb_polynomials[k]
                T_k_with_at = T_k * spatial_attention
                rhs = T_k_with_at @ x_t
                graph_conv_sum += rhs @ self.theta[k]
            gcn_output[:,t,:,:] = graph_conv_sum
        
        x_graph_convolved = F.relu(gcn_output)
        x_time_convolved = self.temporal_conv(x_graph_convolved.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return F.relu(x_time_convolved)

class ST_Block(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps, K):
        super(ST_Block, self).__init__()
        self.attention = SpatioTemporal_Attention(num_nodes, num_features, num_timesteps)
        self.convolution = SpatioTemporal_Convolution(num_features, num_features, K, num_nodes)
        self.residual_conv = nn.Conv2d(num_features, num_features, kernel_size=(1, 1))

    def forward(self, x, adj):
        residual = self.residual_conv(x.permute(0,3,2,1)).permute(0,3,2,1)
        x_att, spatial_att = self.attention(x)
        x_conv = self.convolution(x_att, adj, spatial_att)
        return F.relu(x_conv + residual)

class ASTGCN_Sub_Component(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, num_st_blocks, K):
        super(ASTGCN_Sub_Component, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.blocks.append(ST_Block(num_nodes, num_features, num_timesteps_input, K))
        self.output_layer = nn.Linear(num_timesteps_input * num_features, num_timesteps_output)

    def forward(self, x, adj):
        for block in self.blocks:
            x = block(x, adj)
        B, T, N, F = x.shape
        x_reshaped = x.reshape(B, N, T * F)
        y_hat = self.output_layer(x_reshaped)
        return y_hat

class ASTGCN(nn.Module):
    def __init__(self, config):
        super(ASTGCN, self).__init__()
        self.num_nodes = config['data']['num_nodes']
        num_features = config['data']['num_features']
        self.Tp = config['data']['Tp']
        num_st_blocks = config['model']['num_st_blocks']
        K = config['model']['K']

        self.recent_comp = ASTGCN_Sub_Component(self.num_nodes, num_features, config['data']['Th'], self.Tp, num_st_blocks, K)
        self.daily_comp = ASTGCN_Sub_Component(self.num_nodes, num_features, config['data']['Td'], self.Tp, num_st_blocks, K)
        self.weekly_comp = ASTGCN_Sub_Component(self.num_nodes, num_features, config['data']['Tw'], self.Tp, num_st_blocks, K)

        self.W_h = nn.Parameter(torch.randn(1, self.num_nodes, self.Tp))
        self.W_d = nn.Parameter(torch.randn(1, self.num_nodes, self.Tp))
        self.W_w = nn.Parameter(torch.randn(1, self.num_nodes, self.Tp))
        
    def forward(self, x_h, x_d, x_w, adj):
        y_h = self.recent_comp(x_h, adj)
        y_d = self.daily_comp(x_d, adj)
        y_w = self.weekly_comp(x_w, adj)
        
        y_fused = self.W_h * y_h + self.W_d * y_d + self.W_w * y_w
        return y_fused