# ============================================================
# amr-ai-lab / Day 09 — 编辑距离与比对得分：BLAST 算的也是相似度
# ------------------------------------------------------------
# 对应文章：articles/day09_edit_distance.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：手写 20 行 Levenshtein 编辑距离，量出三对教学序列各差几刀，
#             打印一张 4x3 DP 表让读者看懂「格子怎么填」，并说清差异 vs 相似度
# 运行命令：conda activate amr_ai && python code/day09_edit_distance.py
# 预期输出：终端打印三对序列的编辑距离（1 / 3 / 15）+ 排序判定
#           「与直觉一致: True」+ 迷你 DP 表（无需 outputs/ 文件）
# 依赖：python 3.11+（纯标准库，无第三方包，不用 numpy/biopython）
# 作者备注：这个类比想说明什么 —— 编辑距离就是「改错字」：替换=改一个字、插入=补一个字、
#           删除=划掉一个字；DP 填表像一本「最少操作步骤」的账本，每个格子问一句
#           「到这里最少改几刀」。它算的是「差异」（越大越远），AI 文献里的
#           「相似度」常常就是 1 减去归一化的差异——同一个意思换个方向说。
#           这个脚本**不**演示：回溯打印比对路径（比对篇再做）、打分/替换矩阵
#           （同上）、biopython（Day 20）、多序列比对。
# ============================================================

S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"      # 沿用 day05 主序列（GC 43.3%；教学构造序列，非真实测定值）
S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"      # 沿用 day05：S1 的近亲，第 7/14/25 位共 3 个点突变
S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"      # 沿用 day05：AT 富集陌生菌（GC 26.7%）

def edit_dp(a, b):                          # 手写 Levenshtein：填一张「最少操作步骤」的账本
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]   # dp[i][j] = a 的前 i 个改成 b 的前 j 个，最少改几刀
    for i in range(len(a) + 1): dp[i][0] = i            # 第一列：把 a 的前 i 个全划掉，就要 i 刀（删除）
    for j in range(len(b) + 1): dp[0][j] = j            # 第一行：从空串补出 b 的前 j 个，就要 j 刀（插入）
    for i in range(1, len(a) + 1):                      # 逐行填表：每个格子问一句「到这里最少改几刀」
        for j in range(1, len(b) + 1):                  # 三种操作像改错字：替换=改一个字 / 插入=补一个字 / 删除=划掉一个字
            cost = 0 if a[i - 1] == b[j - 1] else 1     # 末位相同就不花钱；本脚本所有替换代价都=1，真实比对里 A<->G（转换）与 A<->T（颠换）代价不同、蛋白层面还有 BLOSUM 一整张表——「不同的替换代价不同」是 BLAST 打分的起点，后面比对篇正式讲
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)   # 三选一取最省：替换 / 删除 / 插入
    return dp                                           # 返回整张账本（右下角就是答案，也方便拿去展示）

def edit_distance(a, b): return edit_dp(a, b)[-1][-1]   # 答案 = 账本右下角：整条 a 改成整条 b 的最少刀数

S1_del = S1[1:]                             # 模拟一条测序丢了头第一个碱基的序列（29 bp，缺失变体）
d_del, d_s1s2, d_s2s3 = edit_distance(S1, S1_del), edit_distance(S1, S2), edit_distance(S2, S3)
print("三对序列的编辑距离（= 最少改几刀）：")
print(f"  (S1, S1_del) = {d_del}（只差一个开头删除：模拟测序丢头的序列）")
print(f"  (S1, S2)     = {d_s1s2}（day05 的 3 个点突变：k-mer 说「近亲」，编辑距离给出精确的 3 刀）")
print(f"  (S2, S3)     = {d_s2s3}（陌生菌：逐位硬数 19 位不同，最优比对 15 刀——挪一挪 indel 能白捡 4 刀）")
print(f"排序与直觉一致: {d_del < d_s1s2 < d_s2s3}（丢头 < 近亲点突变 < 陌生菌）")
print("\n差异 vs 相似：编辑距离算的是「差异」（越大越远），day07 的 k-mer 距离也是差异；")
print("AI 文献里说的「相似度」，常常就是 1 减去归一化的差异——同一个意思换个方向说。")
sim = 1 - d_s1s2 / max(len(S1), len(S2))    # 相似度视角 = 1 - 差异/最长长度（示意口径，正式定义后面讲）
print(f"示意：S1 vs S2 的相似度视角 = 1 - {d_s1s2}/{len(S1)} = {sim:.3f}")

mini = edit_dp("ACG", "AG")                 # 迷你对：3 bp 改成 2 bp，只用来看懂填表
print("\n迷你 DP 表（行 = ACG 的前缀，列 = AG 的前缀，右下角就是答案）：")
print("      " + "".join(f"{h:>4}" for h in ("''", "A", "G")))   # 表头：'' 表示空前缀
for i, row in enumerate(mini):
    tag = "''" if i == 0 else "ACG"[i - 1]  # 行标签 = a 的第 i 个前缀的末字符
    print(f"  {tag:<2}  " + "".join(f"{v:4d}" for v in row))
print(f"答案 = 右下角 = {mini[-1][-1]}（把 ACG 改成 AG 只需划掉 C 这 1 刀）")

print("\n今天我学会了：编辑距离 = 最少改几刀（替换/插入/删除各算 1 是最朴素的打分）；它算的是「差异」，"
      "AI 说的「相似度」是换个方向说；真实比对会给不同替换不同的代价（BLAST 打分的起点）。")
