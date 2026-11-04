# ============================================================
# amr-ai-lab / Day 12 — 端到端流水线：从 fasta 到分群（第一周主线收官）
# ------------------------------------------------------------
# 对应文章：articles/day12_embedding_pipeline.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：把「切碎 -> 数字化 -> 算远近 -> 降维画图 -> 分群」串成一条链：
#             幂等生成 32 条教学 fasta，读回后向量化、自写 PCA、KMeans 分群，落盘散点图与坐标 CSV
# 运行命令：conda activate amr_ai && python code/day12_embedding_pipeline.py
# 预期输出：终端打印五段小结 + 两簇构成 + X 的两种裁决对照 + 程序化解读（含表型验证警句）
#           + CSV 前 5 行预览；outputs/day12_pipeline_scatter.png 与 outputs/day12_coordinates.csv
# 依赖：python 3.11+ / numpy 1.26.4 / pandas 2.2.2 / matplotlib 3.8.4 / scikit-learn 1.5.0（见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 前五天学的都是单件工具，今天是把它们接成一条产线：
#           文件进 -> 坐标表和图出，这正是真实分析的最小骨架。KMeans 按 16 维全体距离分堆，
#           day11 的 2 维影子投票只看投影后的邻居，X 在两把尺子下答案不同——分歧本身是教学点。
#           这个脚本**不**演示：pipeline 工程化（Snakemake/参数化）、聚类质量指标（轮廓系数）、
#           k 的选择、真实数据清洗（表头/缺失值），这些后面讲。
# ============================================================

import random                                  # 标准库：固定种子生成派生序列（可复现红线，规范 §7）
from collections import Counter                # 标准库：簇构成统计与 2 维投票计票
from pathlib import Path                       # 标准库：data/outputs 路径用 Path(__file__) 拼（规范 §5）
import numpy as np                             # 向量库：16 维向量、SVD 降维、欧氏距离
import pandas as pd                            # 表格库：坐标 CSV 落盘与预览
import matplotlib.pyplot as plt                # 绘图库：散点图与标注（day01/08/11 同款）
from sklearn.cluster import KMeans             # day08 登场过的聚类器：今天接入流水线

DATA_DIR = Path(__file__).resolve().parent.parent / "data"; OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED_A, SEED_B, SEED_KM = 42, 44, 42; rng_a = random.Random(SEED_A); rng_b = random.Random(SEED_B)   # 42 沿用 day08/10/11 口径，44 造 A/B 群，KMeans random_state 固定
print(f"随机种子 = {SEED_A}（S4-S8，与 day08/10/11 逐字符同口径）/ {SEED_B}（A1-A11、B1-B12）/ KMeans random_state = {SEED_KM}——固定：重跑结果不变")

S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"; S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"; S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"   # 沿用 day05-11：S1 主串、S1 近亲、AT 富集陌生菌

def mutate(seq, rng):                          # 与 day08/10/11 逐字符相同：随机 2-4 个位置点突变
    chars = list(seq)
    for p in rng.sample(range(len(seq)), rng.randint(2, 4)): chars[p] = rng.choice([b for b in "ACGT" if b != chars[p]])
    return "".join(chars)

seqs = [("S1", S1), ("S2", S2), ("S3", S3)]    # 教学构造序列，非真实数据（规范 §6）
seqs += [("S" + str(4 + i), mutate(S1, rng_a)) for i in range(3)] + [("S" + str(7 + i), mutate(S3, rng_a)) for i in range(2)] + [("A" + str(i), mutate(S1, rng_b)) for i in range(1, 12)] + [("B" + str(i), mutate(S3, rng_b)) for i in range(1, 13)]   # 先 S 后 A 后 B，rng 调用顺序与 day11 逐字符一致
seqs += [("X", "ACGTAATTGCACGTGATGCATCGATTAATA")]   # day11 冻结的硬编码边界嵌合体石串（教学构造，非真实数据），勿改勿随机生成
fam = ["S1族", "S1族", "S3族", "S1族", "S1族", "S1族", "S3族", "S3族"] + ["S1族"] * 11 + ["S3族"] * 12 + ["S1族"]   # 构造真族 = 流水线全程不读的「答案」

fasta = DATA_DIR / "day12_teaching.fasta"      # 独立文件，不碰 day06 的 sample_teaching.fasta
fasta.write_text("".join(f">{n}\n{s}\n" for n, s in seqs), encoding="utf-8")   # 幂等生成：内容由固定种子唯一决定，每次重写结果相同
records, name = [], None
for line in fasta.read_text(encoding="utf-8").splitlines():    # 第 1 段：把文件读回来——流水线从这里才真正开始
    if line.startswith(">"): name = line[1:]
    elif name: records.append((name, line)); name = None
print(f"\n=== 第 1 段 读 fasta === {fasta.name} 已生成并读回 {len(records)} 条（教学构造，非真实数据），首条 {records[0][0]}，末条 {records[-1][0]}")

DINUCS = [a + b for a in "ACGT" for b in "ACGT"]   # 16 词固定词表，口径与 day05-11 完全一致
vecs = np.array([[[s[i:i + 2] for i in range(len(s) - 1)].count(d) for d in DINUCS] for _, s in records], dtype=float)   # 第 2 段：序列 -> 32x16 数字矩阵
print(f"\n=== 第 2 段 向量化 === {len(records)} 条 -> {vecs.shape[0]}x{vecs.shape[1]} 矩阵（16 种二核苷酸滑窗计数，embedding 雏形）")

Xc = vecs - vecs.mean(axis=0); _, _, vt = np.linalg.svd(Xc, full_matrices=False); XY = Xc @ vt[:2].T   # 第 3 段：自写 PCA（day08/11 同款三行）
print("\n=== 第 3 段 降维 === 自写 PCA：16 维 -> 2 维坐标（XY）")

fa = np.array(fam); iX = [n for n, _ in records].index("X")    # 第 4 段：KMeans 分群（fa 全程不被算法读取，仅供事后对答案）
km = KMeans(n_clusters=2, random_state=SEED_KM, n_init=10).fit(vecs)   # 在 16 维全体距离上分堆，random_state 固定保证可复现
sides = {c: Counter(fa[km.labels_ == c]).most_common(1)[0][0] for c in (0, 1)}   # 每个簇按成员真族多数起名
print("\n=== 第 4 段 分群（KMeans k=2）=== 簇标签:", km.labels_.tolist())
for c in (0, 1):
    mem = [records[i][0] for i in range(32) if km.labels_[i] == c]
    print(f"簇 {c}（{sides[c]}侧）：{len(mem)} 条，构成 {dict(Counter(fa[km.labels_ == c]))}，成员 {' '.join(mem)}")
xk = int(km.labels_[iX]); print(f"X 被分到簇 {xk}（{sides[xk]}侧）——这是 16 维全体距离的裁决")
dd = sorted((float(np.linalg.norm(XY[iX] - XY[j])), j) for j in range(32) if j != iX); pile2d = Counter(fa[j] for _, j in dd[:3]).most_common(1)[0][0]   # day11 的尺子：2 维影子 k=3 近邻投票
print(f"day11 的 2 维影子投票曾把 X 判进 {pile2d} 堆")
if sides[xk] != pile2d:
    print("解读  两把尺子量出两个答案：聚类按 16 维全体距离，与 2 维影子近邻投票是两把尺子——影子与聚类的分歧不是谁错了，而是投影丢掉的方向在提醒：边界样本要回高维看。")

print("\n=== 第 5 段 落盘 ===")
EN = {"S1族": "S1-like", "S3族": "S3-like"}     # 图上全英文（规范 §3）
fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
for c, col in ((0, "#1f77b4"), (1, "#d62728")):
    ax.scatter(XY[km.labels_ == c, 0], XY[km.labels_ == c, 1], s=70, c=col, label=f"cluster {c} ({EN[sides[c]]})")
for i, (n, _) in enumerate(records): ax.annotate(n, XY[i], xytext=(3, 3), textcoords="offset points", fontsize=8)
ax.scatter(XY[iX, 0], XY[iX, 1], s=1100, facecolors="none", edgecolors="red", linewidths=1.8); ax.annotate("X: border strain", XY[iX], xytext=(12, -16), textcoords="offset points", fontsize=9, color="red")   # 红圈标记边界序列
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("32 seqs: k-mer embedding -> PCA -> KMeans (k=2)"); ax.legend(fontsize=8, loc="lower right")   # 图例放右下，避开左右两堆（day08 的坑）
out_png = OUT_DIR / "day12_pipeline_scatter.png"; fig.tight_layout(); fig.savefig(out_png); print(f"图已保存：{out_png}")
df = pd.DataFrame({"name": [n for n, _ in records], "seq": [s for _, s in records],
                   "gc": [f"{(s.count('G') + s.count('C')) / len(s):.4f}" for _, s in records],
                   "cluster": km.labels_.tolist(), "pc1": XY[:, 0].round(4), "pc2": XY[:, 1].round(4)})   # 第 5 段：坐标表落盘
out_csv = OUT_DIR / "day12_coordinates.csv"; df.to_csv(out_csv, index=False); print(f"坐标表已保存：{out_csv}（{len(df)} 行 x {len(df.columns)} 列）")
print("\nCSV 前 5 行预览："); print(df.head().to_string(index=False))

D2 = ((vecs[:, None, :] - vecs[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D2, np.inf)   # 程序化解读：全部两两 16 维距离
pi, pj = np.unravel_index(np.argmin(np.where(km.labels_[:, None] == km.labels_[None, :], D2, np.inf)), D2.shape)   # 同簇最近的一对
cents = {c: vecs[km.labels_ == c].mean(axis=0) for c in (0, 1)}   # 各簇 16 维质心
d_own = [float(np.linalg.norm(vecs[i] - cents[km.labels_[i]])) for i in range(32)]   # 每点到本簇质心的距离
oi = int(np.argmax(d_own))                     # 离群点候选：距离最大者
print(f"""
=== 程序化解读 ===
为什么 {records[pi][0]} 和 {records[pj][0]} 聚在一起：二者 16 维 k-mer 频谱距离全簇最近（{np.sqrt(D2[pi, pj]):.3f}），二核苷酸使用偏好几乎相同——向量化把「谁跟谁像」变成了可计算的数字。
哪一株最像离群点：{records[oi][0]} 到本簇质心的距离 {d_own[oi]:.3f}，全表最大——它是刻意构造的边界嵌合体，前一半像一族、后一半像另一族，两头都不完全归属。
聚类给出的是「按目前的数据怎么分组」的假设，不是身份结论；分组要能用于工作，必须先回到实验室确认。
AI 只能给假设，聚类结果必须回实验室做表型验证。""")

print("\n今天我学会了：一条完整流水线 = 读文件 -> 向量化 -> 降维 -> 分群 -> 落盘；聚类给的是假设不是结论；16 维聚类与 2 维影子是两把尺子，分歧要回高维看。")
