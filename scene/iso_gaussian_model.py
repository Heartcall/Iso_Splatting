#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, build_scaling_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.implicit_model import NeuralSDF, TextureField

# --- 辅助函数：旋转矩阵转四元数 (替代 pytorch3d) ---
def matrix_to_quaternion(M):
    """
    M: [B, 3, 3] rotation matrices
    Returns: [B, 4] quaternions (w, x, y, z)
    """
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    q = torch.zeros((M.shape[0], 4), device=M.device)
    
    mask_tr_pos = tr > 0
    
    # Case 1: Trace > 0
    S = torch.sqrt(tr[mask_tr_pos] + 1.0) * 2
    q[mask_tr_pos, 0] = 0.25 * S
    q[mask_tr_pos, 1] = (M[mask_tr_pos, 2, 1] - M[mask_tr_pos, 1, 2]) / S
    q[mask_tr_pos, 2] = (M[mask_tr_pos, 0, 2] - M[mask_tr_pos, 2, 0]) / S
    q[mask_tr_pos, 3] = (M[mask_tr_pos, 1, 0] - M[mask_tr_pos, 0, 1]) / S

    # Case 2: Trace <= 0 (处理数值稳定性)
    mask_tr_neg = ~mask_tr_pos
    if mask_tr_neg.any():
        # Find major diagonal element
        max_diag_idx = torch.argmax(torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]], dim=1), dim=1)
        
        # Column 0
        mask_c0 = mask_tr_neg & (max_diag_idx == 0)
        if mask_c0.any():
            S0 = torch.sqrt(1.0 + M[mask_c0, 0, 0] - M[mask_c0, 1, 1] - M[mask_c0, 2, 2]) * 2
            q[mask_c0, 0] = (M[mask_c0, 2, 1] - M[mask_c0, 1, 2]) / S0
            q[mask_c0, 1] = 0.25 * S0
            q[mask_c0, 2] = (M[mask_c0, 0, 1] + M[mask_c0, 1, 0]) / S0
            q[mask_c0, 3] = (M[mask_c0, 0, 2] + M[mask_c0, 2, 0]) / S0
        
        # Column 1
        mask_c1 = mask_tr_neg & (max_diag_idx == 1)
        if mask_c1.any():
            S1 = torch.sqrt(1.0 + M[mask_c1, 1, 1] - M[mask_c1, 0, 0] - M[mask_c1, 2, 2]) * 2
            q[mask_c1, 0] = (M[mask_c1, 0, 2] - M[mask_c1, 2, 0]) / S1
            q[mask_c1, 1] = (M[mask_c1, 0, 1] + M[mask_c1, 1, 0]) / S1
            q[mask_c1, 2] = 0.25 * S1
            q[mask_c1, 3] = (M[mask_c1, 1, 2] + M[mask_c1, 2, 1]) / S1
            
        # Column 2
        mask_c2 = mask_tr_neg & (max_diag_idx == 2)
        if mask_c2.any():
            S2 = torch.sqrt(1.0 + M[mask_c2, 2, 2] - M[mask_c2, 0, 0] - M[mask_c2, 1, 1]) * 2
            q[mask_c2, 0] = (M[mask_c2, 1, 0] - M[mask_c2, 0, 1]) / S2
            q[mask_c2, 1] = (M[mask_c2, 0, 2] + M[mask_c2, 2, 0]) / S2
            q[mask_c2, 2] = (M[mask_c2, 1, 2] + M[mask_c2, 2, 1]) / S2
            q[mask_c2, 3] = 0.25 * S2

    return q

class IsoGaussianModel:
    def __init__(self, sh_degree : int):
        # 显式参数
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        # 锚点
        self._anchor_points = torch.empty(0)
        
        # 隐式场 (不作为 Parameter 列表直接暴露给 get_current_geometry 外部，而是内部管理)
        self.sdf_network = NeuralSDF().cuda()
        self.texture_network = TextureField().cuda()
        
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0  
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        # 1. 初始化 Anchors
        self._anchor_points = nn.Parameter(points.requires_grad_(True))
        
        # 2. 初始化 Scaling
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        # 3. 初始化 Opacity
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # 梯度累积变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_xyz(self):
        return self._anchor_points

    def get_current_geometry(self):
        """
        每一帧渲染时动态计算：
        1. 法线 (来自 SDF)
        2. 旋转 (来自 法线)
        3. 颜色 (来自 TextureField) -- 在外部调用 get_colors
        """
        gradients = self.sdf_network.gradient(self._anchor_points)
        normals = torch.nn.functional.normalize(gradients, dim=1)
        
        # 构建切平面坐标系
        ref_vec = torch.tensor([0.0, 1.0, 0.0], device=normals.device).expand_as(normals)
        mask = torch.abs(normals[:, 1]) > 0.9
        ref_vec[mask] = torch.tensor([1.0, 0.0, 0.0], device=normals.device)
        
        tangent_u = torch.cross(normals, ref_vec)
        tangent_u = torch.nn.functional.normalize(tangent_u, dim=1)
        tangent_v = torch.cross(normals, tangent_u)
        
        # R = [u, v, n]
        R_mat = torch.stack([tangent_u, tangent_v, normals], dim=2)
        rotations = matrix_to_quaternion(R_mat)
        
        return self._anchor_points, rotations, normals

    def get_colors(self):
        # 纹理场查询
        return self.texture_network(self._anchor_points)

    def projection_step(self):
        """ 几何吸附：Newton-Raphson Projection """
        with torch.no_grad():
            sdf_val = self.sdf_network(self._anchor_points)
            grad = self.sdf_network.gradient(self._anchor_points)
            grad_norm = torch.nn.functional.normalize(grad, dim=1)
            # p = p - sdf * normal
            projection_vector = sdf_val * grad_norm
            self._anchor_points.data -= projection_vector

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 包含 显式参数 和 隐式网络参数
        l = [
            {'params': [self._anchor_points], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': self.sdf_network.parameters(), 'lr': 1e-4, "name": "sdf"},
            {'params': self.texture_network.parameters(), 'lr': 1e-3, "name": "texture"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # SDF 和 Texture 参数是全局网络，不需要 prune
            if group["name"] in ["sdf", "texture"]:
                continue
            
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._anchor_points = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["sdf", "texture"]:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling):
        # Iso-Splatting 只需要致密化 xyz, opacity, scaling
        # Rotation 和 Color 是隐式推导的，不需要显式存储
        d = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling": new_scaling
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._anchor_points = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 采样新的 Anchors
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        
        # 计算旋转以进行采样偏移 (注意：这里需要临时计算一次旋转)
        _, current_rots, _ = self.get_current_geometry()
        rots = build_rotation(current_rots[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._anchor_points[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities, new_scaling)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # 烘焙几何和纹理用于保存
        with torch.no_grad():
            xyz, rotation, _ = self.get_current_geometry()
            rgb = self.get_colors()
            
            xyz = xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz) # 占位
            # 将 RGB 转换为 SH (dc only) 格式以便兼容 viewers
            f_dc = RGB2SH(rgb).unsqueeze(1).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            # f_rest 置零
            f_rest = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1), dtype=torch.float).flatten(start_dim=1).contiguous().cpu().numpy()
            
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3): l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]): l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]): l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]): l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
