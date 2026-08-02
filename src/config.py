"""
config.py — 项目路径配置（所有脚本统一通过此模块定位文件）

使用方法：
    from config import ROOT, DATA_DIR, RESULTS_CSV_DIR, RESULTS_TXT_DIR, FIGURES_DIR

所有路径都基于 ROOT（项目根目录），脚本在 scripts/ 子目录中运行时自动推导。
"""

import os

# ── 项目根目录 ──────────────────────────────────────────────────────────────
# 当前文件在 scripts/ 下，所以 ROOT = 当前文件目录的父目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── 数据目录 ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data")

# ── 结果输出目录 ────────────────────────────────────────────────────────────
RESULTS_CSV_DIR = os.path.join(ROOT, "results", "csv")
RESULTS_TXT_DIR = os.path.join(ROOT, "results", "txt")

# ── 图表输出目录 ────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(ROOT, "figures")

# ── 辅助函数 ────────────────────────────────────────────────────────────────
def results_csv(path: str) -> str:
    """返回 results/csv/ 下文件的完整路径"""
    return os.path.join(RESULTS_CSV_DIR, path)

def results_txt(path: str) -> str:
    """返回 results/txt/ 下文件的完整路径"""
    return os.path.join(RESULTS_TXT_DIR, path)

def figure(path: str) -> str:
    """返回 figures/ 下文件的完整路径"""
    return os.path.join(FIGURES_DIR, path)