"""
style.py — 计算机顶会统一图表风格（NeurIPS/ICLR 双栏兼容）

所有画图脚本统一从此模块导入，确保：
  - 色弱友好配色（Seaborn colorblind 调色板）
  - 无衬线字体（Arial）
  - 适度字号（双栏可读）
  - 浅灰虚线网格
  - 输出为矢量 PDF

用法:
  from style import set_style, C, set_ax_grid, save_fig
  set_style()
  fig, ax = plt.subplots(figsize=(3.25, 2.5))
  ...
  save_fig(fig, "fig1_phenomenon")
"""

import os, sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# ── 颜色调色板（Seaborn colorblind, 10 色）────────────────────────────────
# 按索引使用: C[0]~C[9]；或按别名使用: C["TENT"], C["SHOT"], C["BN-adapt"]
_PALETTE = sns.color_palette("colorblind", 10).as_hex()
C = {
    # 通用索引
    0: _PALETTE[0], 1: _PALETTE[1], 2: _PALETTE[2], 3: _PALETTE[3],
    4: _PALETTE[4], 5: _PALETTE[5], 6: _PALETTE[6], 7: _PALETTE[7],
    8: _PALETTE[8], 9: _PALETTE[9],
    # 方法别名
    "TENT":     _PALETTE[0],   # 蓝色
    "SHOT":     _PALETTE[3],   # 绿色
    "BN-adapt": _PALETTE[2],   # 橙色
    # 数据分层
    "low":      _PALETTE[0],   # 蓝色
    "mid":      _PALETTE[2],   # 橙色
    "high":     _PALETTE[3],   # 绿色
    "as_0.3":   _PALETTE[0],
    "as_0.5":   _PALETTE[2],
    "as_0.8":   _PALETTE[3],
    # 数据集
    "cora":     _PALETTE[0],
    "questions": _PALETTE[2],
    "reddit":   _PALETTE[3],
    "tolokers": _PALETTE[4],
    "weibo":    _PALETTE[1],
    # 曲线
    "iso":      _PALETTE[5],   # 灰色
    "aniso":    _PALETTE[3],   # 绿色
    "real":     _PALETTE[0],   # 蓝色
}


def set_style():
    """应用顶会统一风格"""
    # 使用 seaborn 的白色背景 + 刻度样式
    sns.set_style("ticks", {
        "axes.facecolor": "white",
        "axes.edgecolor": ".15",
        "axes.grid":      True,
        "grid.color":     ".85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "xtick.bottom":   True,
        "ytick.left":     True,
        "xtick.color":    ".15",
        "ytick.color":    ".15",
    })

    rcParams.update({
        # 字体
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
        "font.size":         8,
        "mathtext.fontset": "dejavusans",

        # 标题 & 标签
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "axes.labelpad":    3,
        "axes.titlepad":    4,
        "axes.linewidth":   0.6,

        # 刻度
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,

        # 图例
        "legend.fontsize":   7,
        "legend.frameon":    True,
        "legend.framealpha": 0.85,
        "legend.fancybox":   False,
        "legend.edgecolor":  ".6",
        "legend.borderpad":  0.3,
        "legend.handlelength": 1.2,

        # 线条
        "lines.linewidth":  1.5,
        "lines.markersize": 4,

        # 图形
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.format":   "pdf",
    })


def set_ax_grid(ax, axis="both"):
    """设置浅灰虚线网格"""
    ax.grid(True, axis=axis, color=".85", linestyle="--", linewidth=0.4,
            alpha=0.8)
    # 确保网格在数据下方
    ax.set_axisbelow(True)


def save_fig(fig, name, dpi=300, bbox_inches="tight", pdf=True, png=False):
    """
    保存图形为 PDF（默认）和/或 PNG。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str — 文件名（不含扩展名）
    pdf : bool — 是否保存 PDF（矢量图，默认 True）
    png : bool — 是否保存 PNG（光栅图，默认 False）
    """
    from config import figure
    if pdf:
        fig.savefig(figure(f"{name}.pdf"), dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved {name}.pdf")
    if png:
        fig.savefig(figure(f"{name}.png"), dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved {name}.png")
    plt.close(fig)