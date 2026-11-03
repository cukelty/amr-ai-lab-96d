# ============================================================
# amr-ai-lab / Day 11 — embedding 雏形：16 维真相 vs 2 维影子（PCA 散点 + 落错堆检测）
# ------------------------------------------------------------
# 对应文章：articles/day11_embedding_scatter.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：把 32 条序列的 16 维二核苷酸向量（embedding 雏形）自写 PCA 降到 2 维画散点图，
#             程序化找出「16 维真族 vs 2 维图上堆」不一致的落错堆点，打印完整证据链
# 运行命令：conda activate amr_ai && python code/day11_embedding_scatter.py
# 预期输出：终端打印种子声明 + 32 条一览 + 落错堆检测报告（X：16 维最近质心 S1 族 / 2 维 3 近邻全 S3 族）
#           + 解读两段；outputs/day11_embedding_scatter.png（双族两色散点，X 红圈标注）
# 依赖：python 3.11+ / numpy 1.26.4 / matplotlib 3.8.4（random / collections 为标准库；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 16 维 k-mer 向量 ≈ 一条菌的「鉴定条」，PCA ≈ 挑两个最有代表性的
#           反应把整条鉴定条画成一张平面图：图能看，但只是影子。X 是刻意构造的边界序列：
#           16 维里离 S1 族质心更近（真相），2 维投影却落进 S3 族堆（影子）——把 day08 的
#           「影子 != 真相」推进一层：降维丢掉的那几维，恰恰可能让边界样本被搬错家。
#           embedding 的价值：把序列变成可计算、可聚类、可分类的数字；
#           风险：任何压缩表示都会丢细节，结论要以高维数据为准。
#           这个脚本**不**演示：真正的语言模型 embedding（ESM/DNABERT，阶段 2 讲）、
#           t-SNE/UMAP 等非线性降维、PCA 方差解释率的定量讨论、监督降维。
# ============================================================

import random                                  # 标准库：固定种子生成派生序列（可复现红线，规范 §7）
from collections import Counter                # 标准库：2 维 3 近邻多数票的计票器
from pathlib import Path                       # 标准库：outputs/ 用 Path(__file__) 拼（规范 §5）
import numpy as np                             # 向量库：16 维向量、SVD 降维、欧氏距离
import matplotlib.pyplot as plt                # 绘图库：散点图与标注（day01/08 同款）

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED_A, SEED_B = 42, 44; rng_a = random.Random(SEED_A); rng_b = random.Random(SEED_B)   # 42 = day08/10 已知菌库同口径，44 = 本篇 A/B 群
print(f"随机种子 = {SEED_A}（S4-S8，与 day08/10 逐字符同口径）/ {SEED_B}（A1-A11、B1-B12）——固定：重跑结果不变")

S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"; S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"; S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"   # 沿用 day05/08/10：S1 主串、S1 近亲、AT 富集陌生菌

def mutate(seq, rng):                          # 与 day08/10 逐字符相同：随机 2-4 个位置点突变
    chars = list(seq)
    for p in rng.sample(range(len(seq)), rng.randint(2, 4)): chars[p] = rng.choice([b for b in "ACGT" if b != chars[p]])
    return "".join(chars)

seqs = [("S1", S1), ("S2", S2), ("S3", S3)]    # 教学构造序列，非真实数据（规范 §6）
seqs += [("S" + str(4 + i), mutate(S1, rng_a)) for i in range(3)] + [("S" + str(7 + i), mutate(S3, rng_a)) for i in range(2)] + [("A" + str(i), mutate(S1, rng_b)) for i in range(1, 12)] + [("B" + str(i), mutate(S3, rng_b)) for i in range(1, 13)]   # 先 S 后 A 后 B，rng 调用顺序固定
seqs += [("X", "ACGTAATTGCACGTGATGCATCGATTAATA")]   # 边界序列：前 14 bp = S1 原样 + 后 16 bp AT 富集；离线调参后冻结的硬编码石串（教学构造，非真实数据），勿改勿随机生成
fam = ["S1族", "S1族", "S3族", "S1族", "S1族", "S1族", "S3族", "S3族"] + ["S1族"] * 11 + ["S3族"] * 12 + ["S1族"]   # 真族 = 构造来源；X 挂 S1 族，下面由程序验证

print("\n=== 32 条序列一览（名字 / 序列 / GC / 真族 = 构造来源）===")
for (n, s), f in zip(seqs, fam): print(f"{n:4}: {s}  GC {(s.count('G') + s.count('C')) / len(s):.1%}  真族 = {f}")

DINUCS = [a + b for a in "ACGT" for b in "ACGT"]   # 16 词固定词表，口径与 day05-10 完全一致
vecs = np.array([[[s[i:i + 2] for i in range(len(s) - 1)].count(d) for d in DINUCS] for _, s in seqs], dtype=float)   # 32x16：embedding 雏形——序列变成可计算的数字
fam_arr = np.array(fam); iX = [n for n, _ in seqs].index("X"); piles, votes_all, nn_all = [], [], []
cent = {f: vecs[(fam_arr == f) & (np.arange(32) != iX)].mean(axis=0) for f in ("S1族", "S3族")}   # 两族 16 维质心（不含 X：用 31 条派生菌算，避免循环论证）
Xc = vecs - vecs.mean(axis=0); _, _, vt = np.linalg.svd(Xc, full_matrices=False); XY = Xc @ vt[:2].T   # 自写 PCA（day08 同款）：中心化 + SVD，取前两主成分 = 2 维影子
for i in range(32):                            # 图上堆检测：2 维空间 k=3 最近邻多数票（不含自己）
    nn = [j for _, j in sorted((float(np.linalg.norm(XY[i] - XY[j])), j) for j in range(32) if j != i)[:3]]
    votes = Counter(fam[j] for j in nn); nn_all.append(nn); votes_all.append(votes); piles.append(votes.most_common(1)[0][0])

print("\n=== 落错堆检测：16 维真族 vs 2 维图上堆（k=3 最近邻多数票，不含自己）===")
for i in [i for i in range(32) if piles[i] != fam[i]]:
    d1 = float(np.linalg.norm(vecs[i] - cent["S1族"])); d3 = float(np.linalg.norm(vecs[i] - cent["S3族"]))
    print(f"{seqs[i][0]}: 真族 {fam[i]} -> 图上堆 {piles[i]}（票型 {dict(votes_all[i])}，3 近邻 {[seqs[j][0] for j in nn_all[i]]}）\n    16 维证据：d(S1族质心) = {d1:.3f}  d(S3族质心) = {d3:.3f}  ->  16 维最近质心 = {'S1族' if d1 < d3 else 'S3族'}（真相）\n    2 维证据：投影坐标 ({XY[i][0]:.2f}, {XY[i][1]:.2f}) 落进 {piles[i]} 堆（影子）")
print("\n解读 1  PCA ≈ 用少数关键生化反应代表整条鉴定条：32 条的 16 维向量被压成 2 维图，肉眼能看，但只是影子。")
print("解读 2  X 的教训：16 维里它离 S1 族质心更近（真相），2 维投影却落进 S3 族堆（影子）——降维会丢细节，边界样本最先被搬错家；embedding 的价值是把序列变成可计算的数字，风险是丢掉的那几维可能恰好关键，结论以高维数据为准。")

for f, c, lab in (("S1族", "#1f77b4", "S1 family (16D nearest)"), ("S3族", "#d62728", "S3 family")):   # 图上全英文（规范 §3）；蓝 = S1 族，红 = S3 族
    plt.scatter(XY[fam_arr == f, 0], XY[fam_arr == f, 1], s=70, c=c, label=lab)
for i, (n, _) in enumerate(seqs): plt.annotate(n, XY[i], xytext=(3, 3), textcoords="offset points", fontsize=8)
plt.scatter(XY[iX, 0], XY[iX, 1], s=1100, facecolors="none", edgecolors="red", linewidths=1.8); plt.annotate("X: mis-placed in 2D", XY[iX], xytext=(12, -16), textcoords="offset points", fontsize=9, color="red")   # X 本体按真族画蓝点，红圈 = 2 维影子搬错家
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("32 seqs: 16-dim k-mer vectors -> 2-dim PCA (X mis-placed)"); plt.legend(fontsize=8, loc="lower right")   # 图例放右下：左下有 A11、右上有 B12（day08 的坑）
out = OUTPUT_DIR / "day11_embedding_scatter.png"; plt.tight_layout(); plt.savefig(out, dpi=200); print(f"\n图已保存：{out}")

print("\n今天我学会了：16 维 k-mer 向量是序列的第一版 embedding；PCA 把它压成 2 维影子；影子会骗人——X 在 16 维里属于 S1 族，图上却落进 S3 族堆；看图下结论前，先回高维核对。")
