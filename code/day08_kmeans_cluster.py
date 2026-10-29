# ============================================================
# amr-ai-lab / Day 08 — 聚类：一行代码的 AI（KMeans 首次登场）
# ------------------------------------------------------------
# 对应文章：articles/day08_kmeans_cluster.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：对 8 条教学序列的 16 维二核苷酸向量跑 KMeans（k=2 与 k=3 对比），
#             终端打印分堆结果与「是否按族分开」判定，并画出 PCA 降维散点图
# 运行命令：conda activate amr_ai && python code/day08_kmeans_cluster.py
# 预期输出：终端打印 8 条序列（标注派生来源）+ k=2/k=3 簇标签 + 两条解读；
#           outputs/day08_kmeans_scatter.png（左 k=2 右 k=3，点旁标序列名）
# 依赖：python 3.11+ / numpy 1.26.4 / matplotlib 3.8.4 / scikit-learn 1.5.0
#       ⏰ scikit-learn 为 day08 首次引入的第三方依赖，实测版本 1.5.0（见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 聚类 ≈ 微生物自动鉴定仪把未知菌按生化反应谱归堆：
#           不需要「谁是谁」的标签，只用「谁跟谁像」。KMeans ≈ 先随机立 k 个「标准菌」，
#           反复「归堆 -> 重立标准」，直到不再变动；k 是人给的参数，不是机器算出来的。
#           这个脚本**不**演示：KMeans 数学推导（只讲循环直觉）、层次聚类 / DBSCAN
#           （后面需要再讲）、肘部法则选 k（Day 14 后）、pandas。
# ============================================================

import random                                  # 标准库：生成派生序列（固定种子，任何人重跑结果一致）
from pathlib import Path                       # 标准库：跨平台路径，outputs/ 用 Path(__file__) 拼（规范 §5）
import numpy as np                             # 向量库：16 维向量与 PCA 降维都靠它
import matplotlib.pyplot as plt                # 绘图库：Figure ≈ 一块画布（day01 同款）
from sklearn.cluster import KMeans             # sklearn 首次登场：聚类本体就这一行 KMeans(...).fit(...)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42; rng = random.Random(SEED); print(f"随机种子 = {SEED}（固定：重跑得到同一批派生序列）")   # 可复现红线：种子固定并打印

S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"          # 沿用 day05 主序列（GC 43.3%）
S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"          # 沿用 day05：S1 的近亲，3 个点突变
S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"          # 沿用 day05：AT 富集陌生菌（GC 26.7%）

def mutate(seq, rng):                          # 点突变派生：随机取 2-4 个位置，各换成另一个碱基
    chars = list(seq)
    for p in rng.sample(range(len(seq)), rng.randint(2, 4)): chars[p] = rng.choice([b for b in "ACGT" if b != chars[p]])
    return "".join(chars)

seqs = [("S1", S1, "day05 原序列"), ("S2", S2, "day05 原序列"), ("S3", S3, "day05 原序列")]   # 教学构造序列，非真实数据
seqs += [("S" + str(4 + i), mutate(S1, rng), "由 S1 点突变派生") for i in range(3)]           # S4-S6：S1 族
seqs += [("S" + str(7 + i), mutate(S3, rng), "由 S3 点突变派生") for i in range(2)]           # S7/S8：S3 族
for t, s, n in seqs: print(f"{t}: {s}（{len(s)} bp，GC {(s.count('G') + s.count('C')) / len(s):.1%}，{n}）")

DINUCS = [a + b for a in "ACGT" for b in "ACGT"]   # 16 词固定词表，口径与 day05/06/07 完全一致
vecs = np.array([[[s[i:i + 2] for i in range(len(s) - 1)].count(d) for d in DINUCS] for _, s, _ in seqs])

km2 = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(vecs)   # 今天的主角就这一行
km3 = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(vecs)
print(f"\nKMeans(n_clusters=k, random_state={SEED}, n_init=10)（序列顺序 = 上面打印的 S1-S8）")
print("k=2 簇标签:", km2.labels_.tolist()); print("k=3 簇标签:", km3.labels_.tolist())

S1_FAM, S3_FAM = {0, 1, 3, 4, 5}, {2, 6, 7}    # S1 族 = S1/S2/S4-S6；S3 族 = S3/S7/S8
ok = len({km2.labels_[i] for i in S1_FAM}) == 1 and len({km2.labels_[i] for i in S3_FAM}) == 1 and km2.labels_[0] != km2.labels_[2]
print(f"解读 1  k=2 恰好按「S1 族 vs S3 族」分开: {ok}")
n1 = len({km3.labels_[i] for i in S1_FAM}); n3 = len({km3.labels_[i] for i in S3_FAM})
print("解读 2  k=3 的变化: " + ("两族都被拆开" if n1 > 1 and n3 > 1 else "S1 族被再拆开" if n1 > 1 else "S3 族被再拆开") + "（k 是人给的，机器只会照办）")

X = vecs - vecs.mean(axis=0)                   # 画图用降维（聚类在 16 维里做的）；自写 PCA，3 行
_, _, vt = np.linalg.svd(X, full_matrices=False)   # 奇异值分解：找出方差最大的两个方向
XY = X @ vt[:2].T                              # 8 条序列的前两个主成分坐标

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))   # 两个子图并排：左 k=2，右 k=3
for ax, labels, k in ((axes[0], km2.labels_, 2), (axes[1], km3.labels_, 3)):
    for c in sorted(set(labels)):
        ax.scatter(XY[labels == c, 0], XY[labels == c, 1], s=70, label=f"cluster {c}")   # 图上全英文（规范 §3）
    for i, (tag, _, _) in enumerate(seqs): ax.annotate(tag, XY[i], xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"KMeans on 8 seqs (k={k})"); ax.legend(fontsize=8, loc="lower right")   # 图例放右下：右上会挡住 S8 的名字
out = OUTPUT_DIR / "day08_kmeans_scatter.png"; plt.tight_layout(); plt.savefig(out, dpi=200); print(f"\n图已保存：{out}")

print("今天我学会了：聚类 = 没有标签也能分堆；KMeans = 先立 k 个「标准」反复「归堆->重立标准」；k 是人给的，结果好不好要人看。")
