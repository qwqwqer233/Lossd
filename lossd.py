import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import os
matplotlib.use('Agg')  # 使用非图形界面后端（不弹窗，可保存）
import matplotlib.pyplot as plt
import math

# 二值化图百分比
def print_mask_distribution(pred_bin_mask):


    print(f"🎯 pred_bin_mask 中像素值分布：")
    print(f"  1（前景）: {num_ones} 像素，占比 {pct_ones:.2f}%")
    print(f"  0（背景）: {num_zeros} 像素，占比 {pct_zeros:.2f}%")

# 二值化图可视化
def save_binary_mask(pred_bin_mask, save_path="pred_bin_mask.png"):

    print(f"✅ 已保存为黑白图：{save_path}")

# 二值化检查
def assert_binary_mask(mask: torch.Tensor, label=None):
    """
    检查 mask 是否严格为二值（0 或 1），否则抛出异常。
    Args:
        mask: Tensor，形状为 [B, 1, H, W]
        label: 可选，报错时输出是哪一类
    """

    if not torch.all((unique_vals == 0) | (unique_vals == 1)):
        raise ValueError(f"❌ pred_bin_mask 中存在非 0/1 值: {unique_vals.tolist()} (label={label})")


def manhattan_distance_soft_limited(mask: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
    """
    对 binary mask 传播每点到最近背景点（值为0）的曼哈顿距离，限制传播步数。
    如果传播到 max_iter 步还没遇到背景，距离就保持为 max_iter。

    Args:
        mask: [B, 1, H, W]，binary mask，前景=1，背景=0
        max_iter: 最大传播步数

    Returns:
        dist: [B, 1, H, W]，每个前景像素点到最近背景的曼哈顿距离
    """


    # Step 2: 曼哈顿传播


    return dist

# 生成正负方向性距离图
def directional_distance_maps(gt_bin_mask: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
    """
    构造方向性距离图（正值：掩码内部距离，负值：掩码外部越界距离）
    Args:
        gt_bin_mask: [B, 1, H, W]，真实标签二值掩码    1 1 1 1 0 0 0 0
        max_iter: 最大传播步数
    Returns:
        full_dist: [B, 1, H, W]，带符号距离图
    """

    return gt_dist_pos, full_dist

# 生成伪one - hot 编码
def ste_one_hot_from_logits(logits: torch.Tensor):
    """
    使用 Straight-Through Estimator 实现的可导 one-hot 预测。

    Args:
        logits: [B, C, H, W] 原始网络输出
    Returns:
        pred_class: [B, H, W] 非可导预测类别索引（for visualization）
        ste_mask:   [B, C, H, W] 可导的 one-hot 近似（for training）
    """
    # Step 1: softmax 得到概率

    return pred_class, ste_mask

# 归一化（两个参数控制）
def normalize_log_triangular_lossvorg(diff: torch.Tensor, max_n: int = 10, clip_n: int = 20, eps: float = 1e-6):
    """
    使用 log(1 + loss) 对三角形+线性增长损失进行归一化压缩，返回范围在 [0, 1]
    """


    return norm_loss

# 归一化，自动设置截取为2倍最大值 针对L1
def normalize_log_triangular_loss_smart(diff: torch.Tensor, max_n: int = 20, eps: float = 1e-6):
    """
    三角+线性增长形式的损失函数的 log 归一化版本
    Args:
        diff: 预测与目标差值 |pred - target|，Tensor
        max_n: 拐点，三角增长 → 线性增长的临界点
        eps: 避免除以 0 的微小数

    Returns:
        norm_loss: [B, 1, H, W]，归一化后的 loss 值，范围 [0, 1]
    """


    return norm_loss


def normalize_log(diff, max_n=10, eps=1e-6):
    """
    norm(d) = log(1 + |d|) / log(1 + max_n)
    """
    safe_diff = diff.abs() + eps
    return torch.log1p(safe_diff) / math.log1p(max_n)  # 输出形状 仍是 [B, 1, H, W]


class DistanceLoss(nn.Module):
    def __init__(self, label_list, label_weight=None, max_iter=50, p=1, reduction='mean'):
        """
        Args:
            label_list: 参与计算的类别列表，如 [1, 2, 3, 4]
            label_weight: 每个类别的损失权重，list 类型，与 label_list 对应
            max_iter: 最大传播步数
            p: L1（1）或 L2（2）距离
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.label_list = label_list
        self.label_weight = label_weight if label_weight is not None else [1.0] * len(label_list)
        assert len(self.label_list) == len(self.label_weight), "label_list 和 label_weight 长度不一致！"

        self.max_iter = max_iter
        self.p = p
        self.reduction = reduction
        self.loss_fn = nn.L1Loss() if p == 1 else nn.MSELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_class, ste_mask = ste_one_hot_from_logits(logits)

        total_loss = 0.0
        valid_class_count = 0

        # 筛选出现的类别 & 属于 label_list 的
        pred_labels = pred_class.unique()  # 预测中出现的类别
        true_labels = target.unique()      # 标签中出现的类别
        all_labels = torch.unique(torch.cat([pred_labels, true_labels], dim=0))  # 合并后去重复

        allowed_labels = torch.tensor(self.label_list, device=logits.device)
        all_labels = all_labels[torch.isin(all_labels, allowed_labels)]

        for label in all_labels:
            label = int(label.item())
            label_idx = self.label_list.index(label)
            weight = self.label_weight[label_idx]

            pred_bin_mask = ste_mask[:, label:label+1, :, :]   #  从 ste_mask 中提取指定 label 的通道，作为该类的预测二值图
            assert_binary_mask(pred_bin_mask, label)
            # print("pred_bin_mask unique values:", torch.unique(pred_bin_mask))  # 打印 看里面出现的是不是1/0
            # print_mask_distribution(pred_bin_mask)  # 计算1/0百分比
            # save_binary_mask(pred_bin_mask, save_path="pred_bin_mask.png")  # 可视化

            target_bin_mask = (target == label).float().unsqueeze(1)

            if pred_bin_mask.sum() < 1e-6 and target_bin_mask.sum() < 1e-6:
                continue

            gt_dist_pos, full_dist = directional_distance_maps(target_bin_mask, self.max_iter)
            pred_dist = pred_bin_mask * full_dist  # 仅提取预测区域的距离值（为正或负） # 4 3 2 1 -1 -2 0 0

            # loss = self.loss_fn(pred_dist, gt_dist_pos)
            diff = torch.abs(pred_dist - gt_dist_pos)  # 0 0 0 0 -1 -2

            # 对这个差值图做归一化
            norm_diff = normalize_log(diff, max_n=self.max_iter)  # 输出形状 仍是 [B, 1, H, W]

            # 然后再聚合（整张图上的像素求平均）
            loss = norm_diff.mean()

            # # 预测前景求平均，有问题，可能预测前景全为0
            # nonzero_mask = (pred_bin_mask == 1).float()
            # loss = (norm_diff * nonzero_mask).sum() / (nonzero_mask.sum() + 1e-6)

            # 真实前景求平均  有问题，可能真实前景为全0
            # target_mask = (target_bin_mask == 1).float()
            # loss = (norm_diff * pred_bin_mask).sum() / (target_mask.sum() + 1e-6)

            total_loss += weight * loss
            valid_class_count += 1

        if valid_class_count == 0:
            return logits.sum() * 0  # 保证梯度存在

        return total_loss / valid_class_count if self.reduction == 'mean' else total_loss