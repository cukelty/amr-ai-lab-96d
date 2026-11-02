# ============================================================
# amr-ai-lab / Day 10 — 最近邻分类：按最像的已知菌来判（kNN 首次登场）
# ------------------------------------------------------------
# 对应文章：articles/day10_knn_classify.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：对 2 条未知菌的 16 维向量，用 day07 的欧氏距离手写 kNN（k=1 与 k=3），
#             对照 day08 的 8 条带标签已知菌判归属，终端打印距离表、判定与真实标签对照
# 运行命令：conda activate amr_ai && python code/day10_knn_classify.py
# 预期输出：终端打印带标签已知菌库一览（名字/GC/标签）+ U1/U2 距离升序前 3 名
#           + k=1/k=3 判定与真实标签对照（应全 True）+ k 值解读一句话；不产出任何文件（本篇不画图）
# 依赖：python 3.11+ / numpy 1.26.4（random / collections 为标准库；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— kNN ≈ MALDI-TOF 鉴定：拿未知菌的「谱」去库里找最像的
#           已知菌，看它最像谁就算它是谁；k=1 信最像的 1 株，k=3 让最像的 3 株投票，
#           投票能抵掉一个恰好很近的噪声邻居。带标签才叫「分类」——上次聚类没有标签，
#           这正是两者的分界线。这个脚本**不**演示：sklearn 的 KNeighborsClassifier
#           （手写才见本质，sklearn 版后面讲）、距离加权投票、train/test 切分与
#           交叉验证（后面讲）、画图。
# ============================================================

import random                                  # 标准库：固定种子生成派生序列（可复现红线，规范 §7）
from collections import Counter                # 标准库：k 票投票的计票器
import numpy as np                             # 向量库：16 维向量与欧氏距离都靠它

SEED = 42; SEED_U = 43; rng = random.Random(SEED); rng_u = random.Random(SEED_U)   # 42 造已知菌库（与 day08 一致），43 造未知菌
print(f"随机种子 = {SEED}（已知菌库，与 day08 一致）/ {SEED_U}（未知菌 U1/U2）——固定：重跑结果不变")

S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"          # 沿用 day05/08 主序列（GC 43.3%）
S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"          # 沿用 day05/08：S1 的近亲，3 个点突变
S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"          # 沿用 day05/08：AT 富集陌生菌（GC 26.7%）

def mutate(seq, rng):                          # 与 day08 逐字符相同：随机 2-4 个位置点突变
    chars = list(seq)
    for p in rng.sample(range(len(seq)), rng.randint(2, 4)): chars[p] = rng.choice([b for b in "ACGT" if b != chars[p]])
    return "".join(chars)

seqs = [("S1", S1), ("S2", S2), ("S3", S3)]    # 教学构造序列，非真实数据（规范 §6）
seqs += [("S" + str(4 + i), mutate(S1, rng)) for i in range(3)]   # S4-S6：S1 族，派生口径与 day08 一致
seqs += [("S" + str(7 + i), mutate(S3, rng)) for i in range(2)]   # S7/S8：S3 族
labels = ["S1族", "S1族", "S3族", "S1族", "S1族", "S1族", "S3族", "S3族"]   # 标签 = 构造来源族：这就是「答案」

print("\n=== 已知菌库（每条菌都有已确认归属——带标签，正是上次聚类没有的东西）===")
for (name, seq), lab in zip(seqs, labels): print(f"{name}: GC {(seq.count('G') + seq.count('C')) / len(seq):.1%}  标签 = {lab}")

def mutate_n(seq, rng, n):                     # 未知菌派生：固定 n 个点突变（幅度可控，保证结果可解读）
    chars = list(seq)
    for p in rng.sample(range(len(seq)), n): chars[p] = rng.choice([b for b in "ACGT" if b != chars[p]])
    return "".join(chars)

# 真实标签只放在这一行里供最后对答案——下面的算法全程不读它，这就是「分类」与「聚类」的分界线
unknowns = [("U1", mutate_n(S1, rng_u, 3), "S1族"), ("U2", mutate_n(S3, rng_u, 3), "S3族")]   # 教学构造，非真实数据

DINUCS = [a + b for a in "ACGT" for b in "ACGT"]   # 16 词固定词表，口径与 day05-08 完全一致
vecs = np.array([[[s[i:i + 2] for i in range(len(s) - 1)].count(d) for d in DINUCS] for _, s in seqs])
uvecs = [np.array([[s[i:i + 2] for i in range(len(s) - 1)].count(d) for d in DINUCS]) for _, s, _ in unknowns]

def euclid(a, b):                              # day07 的欧氏公式原样复用：各维差平方、加总、开根
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def knn(uvec, k):                              # 手写 kNN 三步：算距离 -> 排序 -> 投票（不用 sklearn，手写才见本质）
    dists = sorted((euclid(uvec, vecs[i]), labels[i], seqs[i][0]) for i in range(len(seqs)))   # MALDI-TOF 鉴定 = 拿质谱峰去库里找最像的已知菌
    votes = Counter(lab for _, lab, _ in dists[:k])   # 数 k 票里谁多；3 票 2 类必有一方 >=2，不会平票
    return dists[:k], votes, votes.most_common(1)[0][0]

print("\n=== 未知菌 U1/U2：与库内 8 条的距离升序前 3 名 + k=1/k=3 判定 ===")
for (uname, useq, truth), uvec in zip(unknowns, uvecs):
    print(f"\n{uname}: {useq}（GC {(useq.count('G') + useq.count('C')) / len(useq):.1%}，真实标签 {truth}——仅供对答案，算法没读过）")
    top3, votes3, pred3 = knn(uvec, 3); _, _, pred1 = knn(uvec, 1)   # 距离表取 k=3 调用的前 3 名
    print("  距离前 3: " + "  ".join(f"{nm}({lab}) {d:.3f}" for d, lab, nm in top3) + f"\n  k=1 判定 = {pred1} | k=3 判定 = {pred3}（票型 {dict(votes3)}）| 判定 = 真实: {pred1 == truth and pred3 == truth}")
    print(f"  最近距离 = {top3[0][0]:.3f}——距离太远就该判「库里没有」：可信度门槛后面讲（埋个点）")

print("\n解读  k=1 vs k=3: k=1 信最像的 1 株，k=3 让最像的 3 株投票——投票能抵掉一个「恰好很近的噪声邻居」；本例数据干净、没有噪声邻居，所以两个 k 答案相同（如实打印：这正是好数据集的样子）。")
print("今天我学会了：kNN = 看最像的谁就算谁；带标签才叫分类；k 是人给的，投票比独信一个稳；MALDI-TOF 鉴定就是 kNN 思路。")
