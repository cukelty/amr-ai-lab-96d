# ============================================================
# amr-ai-lab / Day 06 — 周六整合：一条序列变成数字的完整流水线
# ------------------------------------------------------------
# 对应文章：articles/day06_seq_pipeline.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：读出 data/sample_teaching.fasta 的 3 条教学序列，把 day03-05 的零件串成流水线
#             （切 k-mer -> Counter 计数 -> 16 维向量 -> 存 CSV），回读验证并打印三条数字结论
# 运行命令：conda activate amr_ai && python code/day06_seq_pipeline.py
# 预期输出：终端打印序列数、CSV 回读行数校验、两两欧氏距离（与 day05 一致）+ 三条数字结论；
#           data/ 下首次运行生成 sample_teaching.fasta（幂等），outputs/ 下生成 day06_kmer_features.csv
# 依赖：python 3.11+ / numpy 1.26.4（csv / pathlib 为标准库；见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— pipeline ≈ 实验 SOP（上样->扩增->电泳->读带，每步输出是下步输入）；
#           可复现 ≈ 同一份 SOP 谁做都一样：别人一条命令原样跑一遍、得到同一张表，流水线才算交付。
#           整合 day03-05 零件：kmerize 滑窗（day03）、Counter 计数（day04）、16 维向量与欧氏距离（day05）。
#           这个脚本**不**演示：命令行参数（argparse 后面要用了再教）、批量大文件优化、
#           biopython（Day 20 才登场，今天读 fasta 用纯 Python 文本处理）、任何模型训练。
# ============================================================

import csv                                  # 标准库：写/读 CSV 报告（今天不请 pandas，保持「标准库 + numpy」的克制）
from collections import Counter             # 标准库：day04 的计数器，数 k-mer 频次
from pathlib import Path                    # 标准库：跨平台路径，目录用 Path(__file__) 拼（规范 §5）
import numpy as np                          # day05 的向量库：求「差的长度」（欧氏距离）它最顺手

BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根：在哪个目录运行都定位正确
DATA_DIR = BASE_DIR / "data"                        # 教学 fasta 放 data/（规范 §6）
OUTPUT_DIR = BASE_DIR / "outputs"                   # 特征表放 outputs/（规范 §5）
FASTA_PATH = DATA_DIR / "sample_teaching.fasta"
CSV_PATH = OUTPUT_DIR / "day06_kmer_features.csv"

# 教学构造序列，非真实测序数据：与 day03/04/05 逐字符一致（连载连续性）
S1 = "ACGTAATTGCACGTAGGCCATTAACGGTTA"       # 主序列（30 bp）
S2 = "ACGTAACTGCACGAAGGCCATTAAGGGTTA"       # S1 的近亲：3 个点突变 ≈ 同种菌的不同株
S3 = "AATTAATTCGCGTAATTAATTAATTCGCGA"       # AT 富集陌生菌：与 S1 拉开差距
BASES = "ACGT"                              # 碱基字母表
DINUCS = [a + b for a in BASES for b in BASES]      # 固定 16 词表：AA AC ... TT 字典序（day04/05 同一口径）
K = 2                                       # 二核苷酸，k=2：30 bp 切出 29 个窗口

def make_fasta():
    # 首次运行生成教学 fasta，已存在则跳过（幂等：重跑不重写、不报错）。
    # 教学 fasta 由脚本生成，真实项目里你会拿到测序文件；头行注明非真实数据
    if FASTA_PATH.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    with open(FASTA_PATH, "w") as f:
        for name, seq in [("S1", S1), ("S2", S2), ("S3", S3)]:
            f.write(f">{name} teaching sample, not real data\n{seq}\n")

def read_fasta(path):
    # SOP 环节：上样。纯 Python 逐行扫：> 开头记名字，否则拼序列（biopython 到 Day 20 才登场）
    records, name, chunks = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].split()[0], []   # 名字取头行第一个词
        elif line:
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    return records

def kmerize(seq, k):
    # SOP 环节：扩增。day03 的同一把滑窗刀：固定窗口、逐位右移、允许重叠
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

def count_vector(seq):
    # SOP 环节：跑胶读带 -> 转录成表。Counter 数账（day04），再按固定 16 词表逐格取数（day05 口径）
    counts = Counter(kmerize(seq, K))
    return np.array([counts[d] for d in DINUCS], dtype=int)

def save_csv(rows):
    # SOP 环节：出报告。纯 csv 模块写表：列 = name + 16 个二核苷酸列，行 = 3 条序列
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + DINUCS)
        for name, vec in rows:
            writer.writerow([name] + [int(v) for v in vec])

def read_csv():
    # 回读验证：流水线的终点要能被别人原样读回去，才算真的交付
    with open(CSV_PATH, newline="") as f:
        lines = list(csv.reader(f))
    return lines[0], lines[1:]

def euclid(a, b):
    # day05 的同一把尺子：d = sqrt(sum((a-b)^2))
    return float(np.linalg.norm(a - b))

def main():
    make_fasta()
    records = read_fasta(FASTA_PATH)                 # SOP 第 1 步：上样
    print(f"fasta 读到 {len(records)} 条序列：{[name for name, _ in records]}")
    vecs = [(name, count_vector(seq)) for name, seq in records]   # SOP 第 2-4 步：扩增->读带->转录成表
    save_csv(vecs)                                   # SOP 第 5 步：出报告
    print(f"特征表已写入 {CSV_PATH}")

    header, rows = read_csv()                        # 回读 CSV 验证
    names = [row[0] for row in rows]
    mat = [np.array([int(x) for x in row[1:]]) for row in rows]
    print(f"回读 CSV：表头 {len(header)} 列，数据 {len(rows)} 行 == 序列数 {len(records)} -> {len(rows) == len(records)}")

    d12, d13, d23 = euclid(mat[0], mat[1]), euclid(mat[0], mat[2]), euclid(mat[1], mat[2])
    print(f"两两欧氏距离（用回读的向量算，与 day05 对照）：S1-S2 {d12:.3f} / S1-S3 {d13:.3f} / S2-S3 {d23:.3f}")

    row_sum = sum(int(x) for x in rows[0][1:])
    print("\n=== 三条数字结论 ===")
    print(f"① 3 条序列 -> 3 行 x 16 列特征表（CSV 共 17 列 = name + 16 个二核苷酸）")
    print(f"② 近亲 S1-S2 距离 {d12:.3f} < 陌生菌 S1-S3 {d13:.3f} < S2-S3 {d23:.3f}，与 day05 数值一致")
    print(f"③ 每行 16 列加总 = {row_sum} 窗口 = 30 - 2 + 1（k=2 滑窗窗口数，day03 的公式）")
    print("\n今天我学会了：流水线 = 把零件按 SOP 串起来，每步输出是下步输入；能被别人一条命令原样跑一遍、得到同一张表，才叫可复现。")

if __name__ == "__main__":
    main()
