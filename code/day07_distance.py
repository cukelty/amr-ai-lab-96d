# ============================================================
# amr-ai-lab / Day 07 — 距离：欧氏距离 ≈ 菌株之间差了多少
# ------------------------------------------------------------
# 对应文章：articles/day07_distance.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：把 day05 手算的欧氏距离升级成可复用函数，对接 day06 产出的特征表 CSV，
#             算出三条序列的两两距离，并拆开 S1-S2 逐格看差多少、换曼哈顿尺子读数怎么变
# 运行命令：conda activate amr_ai && python code/day07_distance.py
# 预期输出：终端打印三条 16 维向量（与 day05 逐位一致）+ 两两欧氏距离
#           （S1-S2 3.162 / S1-S3 7.211 / S2-S3 8.246，与 day05 一致：True）
#           + S1-S2 逐维差拆解（L1=10，平均每格差 0.625）+ 欧氏/曼哈顿两把尺子对照行；不产出任何文件
# 依赖：python 3.11+ / numpy 1.26.4（csv / sys / pathlib 为标准库；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 距离 ≈ 一把尺子：把「两条序列差多少」量成一个数；
#           欧氏 = 各维差平方、加总、开根（勾股定理的多维版），平方放大了大差距、开根又压回来，
#           差得多的格子话语权更大；但「直线距离」不是唯一合理的尺子——曼哈顿
#           （像在棋盘街区间走路：不平方、直接加绝对值）读数就不同，尺子要跟问题配。
#           这个脚本**不**演示：余弦相似度（后面讲聚类/相似检索时登场）、降维可视化、sklearn。
# ============================================================

import sys                                  # 标准库：CSV 缺失时带非零状态码退出，让上游知道没跑成
import csv                                  # 标准库：读 day06 产出的特征表（保持「标准库 + numpy」的克制）
from pathlib import Path                    # 标准库：跨平台路径，目录用 Path(__file__) 拼（规范 §5）
import numpy as np                          # day05 的向量库：求「差的长度」它最顺手

BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根：在哪个目录运行都定位正确
CSV_PATH = BASE_DIR / "outputs" / "day06_kmer_features.csv"   # day06 流水线的产出，今天的主角输入

def euclid(a, b):
    # 欧氏距离：d = sqrt(sum((a-b)^2))——day05 手算正式升级为可复用函数
    # 勾股定理的多维版：各维差平方、加总、开根
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def manhattan(a, b):
    # 曼哈顿距离（对照尺子）：不平方、直接加绝对值——像在棋盘街区间走路
    return float(np.abs(np.array(a) - np.array(b)).sum())

def load_vectors(path):
    # 对接 day06：读特征表 CSV，跳过 name 列取 16 个二核苷酸计数，回读成 3 个 16 维向量
    if not path.exists():
        print("找不到 outputs/day06_kmer_features.csv——先跑一遍 python code/day06_seq_pipeline.py 生成特征表")
        sys.exit(1)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return [(row[0], np.array([int(x) for x in row[1:]])) for row in rows[1:]]   # 表头 17 列 = name + 16 词

vecs = load_vectors(CSV_PATH)               # 读回 S1/S2/S3 三行向量
for name, vec in vecs:
    print(f"{name} 的 16 维向量: {vec}")    # 应与 day05 打印值逐位一致

pairs = [("S1-S2", 0, 1), ("S1-S3", 0, 2), ("S2-S3", 1, 2)]
eu = [(tag, euclid(vecs[i][1], vecs[j][1])) for tag, i, j in pairs]
print("\n两两欧氏距离: " + " / ".join(f"{tag} {d:.3f}" for tag, d in eu))
day05 = [3.162, 7.211, 8.246]               # day05 手算的三个数：今天换 CSV 来路重算，看是否对得上
print(f"与 day05 一致: {all(abs(d - ref) < 0.001 for (_, d), ref in zip(eu, day05))}")

a, b = vecs[0][1], vecs[1][1]               # 本篇的魂：拿 S1-S2 拆开看，别只盯着一个总数
diff = a - b                                # 16 个格子逐个问：这里差多少？
l1 = manhattan(a, b)
print("\n=== 怎么读距离：把 S1-S2 拆开看 ===")
print(f"逐维差 (S1-S2): {diff.tolist()}")
print(f"绝对值总和（曼哈顿 L1）= {l1:.0f}，平均每格差 = L1/16 = {l1 / 16:.3f}")
print(f"欧氏 L2 = sqrt(sum(差^2)) = {euclid(a, b):.3f}，注意它不是每格平均差 {l1 / 16:.3f}——")
print("平方放大了大差距，开根又压回来：差得多的格子话语权更大")

print("\n=== 换把尺子：欧氏 L2 vs 曼哈顿 L1（同一对序列，数字不同）===")
for tag, i, j in pairs:
    print(f"  {tag}: 欧氏 L2 {euclid(vecs[i][1], vecs[j][1]):.3f} | 曼哈顿 L1 {manhattan(vecs[i][1], vecs[j][1]):.0f}")

print("\n今天我学会了：距离 = 把「差多少」变成一个数；欧氏 = 各维差平方和开根，平方放大了大差距；换尺子数字会变——尺子要跟问题配。")
