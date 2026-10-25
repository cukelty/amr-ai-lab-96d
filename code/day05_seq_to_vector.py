# ============================================================
# amr-ai-lab / Day 05 — 向量化第一步：把一条序列变成一串数字
# ------------------------------------------------------------
# 对应文章：articles/day05_seq_to_vector.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：算出三条教学序列的碱基组成 4 维向量与二核苷酸 16 维向量，
#             并打印两两欧氏距离，验证「近亲序列距离 < 陌生序列距离」
# 运行命令：conda activate amr_ai && python code/day05_seq_to_vector.py
# 预期输出：终端打印三条序列的 4 维 / 16 维向量 + 两两欧氏距离矩阵（4 维与 16 维各一组）
#           + 验证结论（16 维下 S1-S2 < S1-S3 为 True，4 维下同样成立）；不产出任何文件
# 依赖：python 3.11+ / numpy 1.26.4（连载第一次用 numpy；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 让电脑比较两条序列的前提，是先把它们各自变成一排数字（向量）：
#           向量 ≈ API 鉴定条上 20 管反应结果排成一行，或 MALDI-TOF 的峰图；
#           坐标相近 = 对象相近（生化反应谱相近的菌，在鉴定库里就归到一起 ≈ 向量空间里靠近）。
#           但要亲手认清边界：今天的坐标是人按规则【手造】的（k-mer 计数），
#           不是模型【学】出来的 embedding（embedding 由训练学出，上模型是后面的事）。
#           这个脚本**不**演示：可视化（散点图后面用得上再说）、sklearn / torch、任何模型训练。
# ============================================================

import numpy as np                           # 向量登场，numpy 登场：一排数字的减法与求「差的长度」（求距离），它最顺手

# 教学构造序列，非真实测序数据
S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"        # 与 day03/04 同一条主序列（连载连续性）；A9 C6 G7 T8，GC 13/30 ≈ 43.3%
S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"        # S1 的近亲：第 7/14/25 位各 1 个点突变（T->C、T->A、C->G）≈ 同种菌的不同株；GC 14/30 ≈ 46.7%
S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"        # 陌生菌：AT 富集设计序列，GC 8/30 ≈ 26.7%，与 S1 拉开差距

BASES = "ACGT"                               # 碱基字母表：4 维向量的每一维对应一种碱基（鉴定条上 4 管反应各占一格）
DINUCS = [a + b for a in BASES for b in BASES]   # 二核苷酸固定词表：按 A/C/G/T 字典序 16 格排好（day04 数 top10，今天全数齐、排固定顺序）

def base_vector(seq):
    # 第一级：碱基组成 4 维向量 [count_A, count_C, count_G, count_T]（API 鉴定条每管一个反应，4 管读数排成一行）
    return np.array([seq.count(b) for b in BASES], dtype=int)

def dinuc_vector(seq):
    # 第二级（今天的主角）：2-mer 频次 16 维向量——滑窗切出 29 个窗口，再按固定词表逐格计数
    tokens = [seq[i:i + 2] for i in range(len(seq) - 1)]   # day03 的同一把刀，k=2
    return np.array([tokens.count(d) for d in DINUCS], dtype=int)

def euclid(a, b):
    # 欧氏距离：d = sqrt(sum((a-b)^2))，手算可跟；np.linalg.norm(a-b) 就是「两坐标相减后的长度」
    return float(np.linalg.norm(a - b))

names = [("S1", S1), ("S2", S2), ("S3", S3)]
for tag, seq in names:
    gc = seq.count("G") + seq.count("C")
    print(f"{tag}: {seq}（{len(seq)} bp，GC {gc}/{len(seq)} = {gc / len(seq):.1%}）")

for tag, seq in names:
    print(f"{tag} 的 4 维向量 [A,C,G,T]: {base_vector(seq)}")
for tag, seq in names:
    print(f"{tag} 的 16 维向量: {dinuc_vector(seq)}")   # 维度顺序 = 词表 AA AC AG AT CA CC CG CT GA GC GG GT TA TC TG TT

for label, make in [("4 维（碱基组成）", base_vector), ("16 维（二核苷酸）", dinuc_vector)]:
    vecs = [make(seq) for _, seq in names]
    print(f"\n=== {label}：两两欧氏距离 d = sqrt(sum((a-b)^2)) ===")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  {names[i][0]}-{names[j][0]}: {euclid(vecs[i], vecs[j]):.3f}")

d4_12, d4_13 = euclid(base_vector(S1), base_vector(S2)), euclid(base_vector(S1), base_vector(S3))
d16_12, d16_13 = euclid(dinuc_vector(S1), dinuc_vector(S2)), euclid(dinuc_vector(S1), dinuc_vector(S3))
print(f"\n验证（4 维）：S1-S2 ({d4_12:.3f}) < S1-S3 ({d4_13:.3f}) -> {d4_12 < d4_13}")
print(f"验证（16 维）：S1-S2 ({d16_12:.3f}) < S1-S3 ({d16_13:.3f}) -> {d16_12 < d16_13}")
print("-> 距离拉开了，才谈得上「分得开」：维数越高，序列差异被刻画得越细；低维看走眼的风险是真实存在的")

print("\n今天我学会了：比较的前提是先变成数字（向量 ≈ API 鉴定条排成一行）；坐标相近 = 对象相近；今天的坐标是手造的，真 embedding 是模型学出来的。")
