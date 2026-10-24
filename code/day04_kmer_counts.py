# ============================================================
# amr-ai-lab / Day 04 — 计数与词频：把碎片变成统计量
# ------------------------------------------------------------
# 对应文章：articles/day04_kmer_counts.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：数出教学序列的 3-mer 频次谱（top10 + 总窗口数 28 + 唯一数 21），
#             并用三个对照实验亲眼看到「词袋丢了什么、没丢什么」
# 运行命令：conda activate amr_ai && python code/day04_kmer_counts.py
# 预期输出：终端打印 top10 3-mer 频次表 + 演示 A（打乱 token 顺序 -> 词频完全相等 True）
#           + 演示 B（整条序列倒着读 -> 词频不相等 False，但镜像关系 freq_rev(s)==freq(s[::-1]) 成立 True）
#           + 诚实演示（打乱碱基本身 -> 词频不相等 False，并打印有几种 3-mer 计数变了）；不产出任何文件
# 依赖：python 3.11+（本脚本纯标准库 collections/random，无第三方包；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 模型会把碎片数一数，出现多少次本身就是一种特征：
#           k-mer 频次谱 ≈ 词频（就像 GC 含量、四核苷酸频率 tetranucleotide frequency 可用于物种识别），
#           词袋 Bag-of-Words ≈ 只数不看顺序——一篇文档就是把词全倒进袋子里，排队信息不参与计数。
#           但要亲手拆穿一个常见误解：词袋丢的是【碎片的排列】（打乱 token 顺序，词频纹丝不动），
#           不是【字符的排列】（打乱碱基、倒读序列都会改变词频——窗口内的相邻关系还在）。
#           这个脚本**不**演示：构建特征向量 / 喂给任何模型（Day 14 才上模型）、
#           向量化 / embedding（Day 05）、任何第三方库（Counter 是标准库）。
# ============================================================

import random                                # 标准库：可固定种子的随机打乱
from collections import Counter              # 标准库：计数器 ≈ 手边的小 tally 本，数词频一行搞定

# 教学构造序列，非真实测序数据：与 day03 完全同一条（连载连续性），30 bp，只含 A/C/G/T
SEQ = "ACGTAATTGCACGTAGGCCATTAACGGTTA"

def kmerize(seq, k):
    # 滑窗切分（day03 的同一把刀）：从第 0 位开始每次右移 1 位，取长度 k 的小段直到切不下
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

tokens = kmerize(SEQ, 3)                     # k=3：30 bp 切出 28 个窗口，频次谱有层次（k=5 几乎全唯一，没看头）
freq = Counter(tokens)                       # 数次数：Counter 就是把每个 3-mer 出现的次数记成一本账
print(f"教学序列：{SEQ}（{len(SEQ)} bp，与 day03 同一条）")
print(f"\n=== k=3 词频谱：{len(tokens)} 个窗口 -> {len(freq)} 个唯一 3-mer ===")
for rank, (word, n) in enumerate(freq.most_common(10), start=1):
    print(f"  {rank:02d}. {word}  x{n}")     # top10 编号 + 3-mer + 次数，可拿笔对着序列手工核对
# 这张频次谱就是最朴素的特征：GC 含量、四核苷酸频率用于物种识别，走的正是同一条「数次数」的路
print(f"（窗口数 {len(tokens)}、唯一数 {len(freq)}，与 day03 对比表 k=3 那一行对上）")

print("\n=== 演示 A：把 token 全倒进袋子再打乱（词袋 Bag-of-Words）===")
rng = random.Random(42)                      # 固定种子 42：任何人重跑，打乱结果都一模一样
print(f"随机种子：{42}（固定，保证可复现）")
shuffled = tokens[:]                         # 复制一份再打乱，原始 token 列表保持不动
rng.shuffle(shuffled)                        # 打乱的是【k-mer 列表的顺序】，每个 k-mer 本身没变
freq_shuffled = Counter(shuffled)            # 词袋模型眼里：文档 = 袋子里的词，排队顺序不参与计数
print(f"打乱后词频与原词频完全相等：{freq == freq_shuffled}")
print("-> 词袋丢的就是这个：碎片出现多少次它都记得，碎片怎么排队它一概不知")

print("\n=== 演示 B：把整条序列倒过来读（SEQ[::-1]）===")
freq_rev = Counter(kmerize(SEQ[::-1], 3))    # 反转序列后重新数：窗口内的字符顺序也被倒过来了
mirror_ok = all(freq_rev[w] == freq[w[::-1]] for w in freq)   # 数学事实：倒序词频(s) == 原词频(reverse(s))
print(f"倒序序列词频与原词频相等：{freq_rev == freq}")
print(f"但镜像关系成立（倒序词频(s) == 原词频(s 倒写)）：{mirror_ok}")
print(f"例：原序列 ACG x{freq['ACG']}，倒读后 GCA x{freq_rev['GCA']}——反转连窗口内部都倒了，词频谱当然会动")

print("\n=== 诚实演示：打乱碱基本身（不是打乱 token）===")
bases = list(SEQ)                            # 这次打乱的是【字符】：把 30 个碱基拆散重排
rng.shuffle(bases)
seq_scram = "".join(bases)                   # 重组成一条新序列：窗口内的相邻关系全被换了
freq_scram = Counter(kmerize(seq_scram, 3))
diff = sum(1 for w in set(freq) | set(freq_scram) if freq[w] != freq_scram[w])   # 数有几种 3-mer 计数变了
print(f"打乱碱基后的序列：{seq_scram}")
print(f"碱基打乱后词频与原词频相等：{freq == freq_scram}（计数改变的 3-mer 有 {diff} 种）")
print("-> 词袋丢的是碎片的排列，不是字符的排列：字符一乱，窗口内的相邻关系就变，词频跟着变")

print("\n今天我学会了：数次数本身就是特征（词频 ≈ k-mer 频次谱），但词袋只认袋子不认排队——打乱 token 词频不变（顺序被丢了）；打乱碱基词频会变（窗口内的相邻还在）。")
