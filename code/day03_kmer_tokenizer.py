# ============================================================
# amr-ai-lab / Day 03 — 切分：token 就是 k-mer
# ------------------------------------------------------------
# 对应文章：articles/day03_kmer_tokenizer.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：把一条 30 bp 序列按滑窗切成 k-mer，亲眼看到「词表」长什么样、k 变大后每个碎片有多稀有
# 运行命令：conda activate amr_ai && python code/day03_kmer_tokenizer.py
# 预期输出：终端打印 26 个 5-mer（编号 01-26）+ 前 3 个供手工核对 + 唯一数 25 + 排序词表 25 项
#           + k=3/5/7 对比表（窗口数 28/26/24，唯一数 21/25/24，理论空间 64/1024/16384）
#           + N 碱基演示：跳过 3 个含 N 的窗口；不产出任何文件
# 依赖：python 3.11+（本脚本纯标准库，无第三方包；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— 模型不认识「字」，只认识按固定窗口切出来的碎片；
#           检验人切 k-mer 的手，和模型切 token 的手是同一把刀：
#           token（分词）≈ k-mer 滑窗切分，k 的大小决定看到多细（就像 k-mer 选 21 还是 31），
#           词表 vocabulary ≈ 所有出现过的 k-mer 集合，未知 token ≈ 序列里的 N 碱基（超出词表没法处理）。
#           这个脚本**不**演示：词频统计（Day 04 的活）、向量化 / embedding（Day 05）、
#           任何 NLP 库（今天连 pandas 都不用，一把滑窗刀切到底）。
# ============================================================

# 教学构造序列，非真实测序数据：恰好 30 bp，只含 A/C/G/T
# 手工核对过的三条设计约束：
#   ① 30 bp，k=5 时窗口数 = 30 - 5 + 1 = 26；
#   ② 5-mer "ACGTA" 出现两次（第 1-5 位和第 11-15 位），其余 5-mer 各只出现一次 -> 唯一数 25 < 26，演示「词表去重」；
#   ③ k=7 时 24 个 7-mer 全部唯一 -> 形成「k 变大 -> 每个 k-mer 变稀有」的对比。
SEQ = "ACGTAATTGCACGTAGGCCATTAACGGTTA"

def kmerize(seq, k):
    # 滑窗切分：从第 0 位开始，每次右移 1 位，取长度为 k 的小段，直到切不下为止
    # 这就是 k-mer 滑窗 —— 也正是模型切 token 的同一个操作（固定窗口、逐位右移、允许重叠）
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

kmers5 = kmerize(SEQ, 5)                                   # 先切 k=5：主菜
print(f"教学序列：{SEQ}（{len(SEQ)} bp，只含 A/C/G/T）")
print(f"\n=== k=5：滑窗切出 {len(kmers5)} 个 k-mer ===")
for row in range(0, len(kmers5), 5):                       # 每 5 个一行排开，方便对着序列逐个数
    print("  " + "  ".join(f"{j + 1:02d}:{w}" for j, w in enumerate(kmers5[row:row + 5], start=row)))
print(f"前 3 个（手工核对：分别对照序列第 1-5、2-6、3-7 位）：{kmers5[:3]}")
print(f"唯一 k-mer 数：{len(set(kmers5))}（< 26，因为 ACGTA 出现了两次——set 自动去重，词表就是这样攒出来的）")
vocab = sorted(set(kmers5))                                # 词表 = 所有出现过的 k-mer 集合，排个序好找
print(f"词表 vocabulary（{len(vocab)} 项）= {vocab}")

print("\n=== 换个 k 再切：k 变大 -> 理论空间指数变大、单个 k-mer 几乎不重复 ===")
print(f"{'k':>2}  {'窗口数':>4}  {'唯一数':>4}  {'理论空间 4**k':>10}")
for k in (3, 5, 7):                                        # 一次只改一个变量：只动 k，序列不动
    w = kmerize(SEQ, k)
    print(f"{k:>2}  {len(w):>4}  {len(set(w)):>4}  {4 ** k:>10}")

seq_n = SEQ[:2] + "N" + SEQ[3:]                            # 另存新串：把第 3 位改成 N，原序列不动
skipped = sum(1 for w in kmerize(seq_n, 5) if "N" in w)    # 含 N 的窗口没法进词表（未知 token），惯例是跳过
print(f"\n=== 未知 token 小演示：第 3 位改成 N -> {seq_n} ===")
print(f"含 N 的窗口跳过 {skipped} 个（k=5 时第 1-3 号窗口都罩住第 3 位）——就像模型遇到词表外的 token 只能绕着走")

print("\n今天我学会了：模型不认识「字」，只认切出来的碎片——token 就是 k-mer；k 决定看多细，词表就是见过的 k-mer 全集，N 碱基是它绕着走的未知 token。")
