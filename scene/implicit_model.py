# 新增模块：隐式场定义 (scene/implicit_model.py)
# 我们需要一个支持二阶导数（用于计算法线和 Eikonal Loss）的 SDF 网络和一个纹理场。这里使用 PyTorch 原生实现以保证兼容性，通过 HashGrid (Multi-resolution Feature Grid) 保证速度。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder(nn.Module):
    def __init__(self, input_dims, multires, log_sampling=True):
        super().__init__()
        self.max_freq = multires - 1
        self.N_freqs = multires
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2.**torch.linspace(0., self.max_freq, steps=multires) if log_sampling else torch.linspace(1, 2.**self.max_freq, steps=multires)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, -1)

class NeuralSDF(nn.Module):
    """
    神经符号距离场 (Neural SDF)
    输入: xyz (3D 坐标)
    输出: sdf 值 (标量)
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=4, multires=6):
        super().__init__()
        self.embedder = Embedder(input_dim, multires)
        input_ch = input_dim + input_dim * 2 * multires
        
        layers = []
        for i in range(num_layers):
            in_dim = input_ch if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.Softplus(beta=100)) # 使用 Softplus 保证导数连续
                # layers.append(nn.LayerNorm(hidden_dim)) # Geometric initialization usually prefers no norm
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 归一化输入到 [-1, 1] 或 [0, 1] 范围通常有助于训练，这里假设输入已归一化
        h = self.embedder(x)
        sdf = self.net(h)
        return sdf

    def gradient(self, x):
        """计算 SDF 的梯度 (即法线方向)"""
        x.requires_grad_(True)
        y = self.forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

class TextureField(nn.Module):
    """
    纹理场 (Texture Field)
    输入: xyz (几何表面点), view_dir (可选，用于高光)
    输出: RGB 颜色
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=4, multires=4):
        super().__init__()
        self.embedder = Embedder(input_dim, multires)
        input_ch = input_dim + input_dim * 2 * multires
        
        layers = []
        for i in range(num_layers):
            in_dim = input_ch if i == 0 else hidden_dim
            out_dim = 3 if i == num_layers - 1 else hidden_dim # 输出 RGB
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.embedder(x)
        rgb = self.net(h)
        return self.sigmoid(rgb)