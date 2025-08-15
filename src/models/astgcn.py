# src/models/astgcn.py
# --- FINAL CORRECTED VERSION ---

import torch
import torch.nn as nn
import torch.nn.functional as F  # Keep this import for clarity

class SpatioTemporal_Attention(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps, hidden_dim=64):
        super(SpatioTemporal_Attention, self).__init__()
        self.query_proj_s = nn.Linear(num_timesteps * num_features, hidden_dim)
        self.key_proj_s = nn.Linear(num_timesteps * num_features, hidden_dim)
        self.query_proj_t = nn.Linear(num_nodes * num_features, hidden_dim)
        self.key_proj_t = nn.Linear(num_nodes * num_features, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, T, N, C = x.shape
        x_spatial = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        query_s = self.query_proj_s(x_spatial)
        key_s = self.key_proj_s(x_spatial)
        S = torch.matmul(query_s, key_s.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
        # Use explicit functional call to avoid conflicts
        S_prime = torch.nn.functional.softmax(S, dim=-1)

        x_temporal = x.reshape(B, T, N * C)
        query_t = self.query_proj_t(x_temporal)
        key_t = self.key_proj_t(x_temporal)
        E = torch.matmul(query_t, key_t.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
        # Use explicit functional call to avoid conflicts
        E_prime = torch.nn.functional.softmax(E, dim=-1)
        
        x_weighted_by_time = torch.matmul(E_prime, x_temporal).reshape(B, T, N, C)
        return x_weighted_by_time, S_prime

class SpatioTemporal_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, K, num_nodes):
        super(SpatioTemporal_Convolution, self).__init__()
        self.K = K
        self.theta = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.num_nodes = num_nodes
    
    def forward(self, x, adj, spatial_attention):
        B, T, N, C_in = x.shape
        
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj
        lambda_max = torch.linalg.eigvalsh(L).max()
        L_tilde = (2 * L / lambda_max) - torch.eye(self.num_nodes, device=x.device)

        cheb_polynomials = [torch.eye(self.num_nodes, device=x.device), L_tilde]
        for i in range(2, self.K):
            cheb_polynomials.append(2 * L_tilde @ cheb_polynomials[-1] - cheb_polynomials[-2])
        
        gcn_output = torch.zeros(B, T, N, self.theta.shape[2], device=x.device)
        for t in range(T):
            x_t = x[:, t, :, :]
            graph_conv_sum = torch.zeros(B, N, self.theta.shape[2], device=x.device)
            for k in range(self.K):
                T_k = cheb_polynomials[k]
                T_k_with_at = T_k * spatial_attention
                rhs = T_k_with_at @ x_t
                graph_conv_sum += rhs @ self.theta[k]
            gcn_output[:, t, :, :] = graph_conv_sum
        
        # Use explicit functional call to avoid conflicts
        x_graph_convolved = torch.nn.functional.relu(gcn_output)
        
        x_time_convolved = self.temporal_conv(x_graph_convolved.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # Use explicit functional call to avoid conflicts
        return torch.nn.functional.relu(x_time_convolved)

class ST_Block(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps, K):
        super(ST_Block, self).__init__()
        self.attention = SpatioTemporal_Attention(num_nodes, num_features, num_timesteps)
        self.convolution = SpatioTemporal_Convolution(num_features, num_features, K, num_nodes)
        self.residual_conv = nn.Conv2d(num_features, num_features, kernel_size=(1, 1))

    def forward(self, x, adj):
        residual = self.residual_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_att, spatial_att = self.attention(x)
        x_conv = self.convolution(x_att, adj, spatial_att)
        # Use explicit functional call to avoid conflicts
        return torch.nn.functional.relu(x_conv + residual)

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
        
        # *** FIX: Changed F to C to avoid variable shadowing ***
        B, T, N, C = x.shape
        x_reshaped = x.reshape(B, N, T * C)
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