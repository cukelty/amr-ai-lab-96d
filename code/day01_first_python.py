# ============================================================
# amr-ai-lab / Day 01 — 从一张药敏表到第一行 Python
# ------------------------------------------------------------
# 对应文章：articles/day01_first_python.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：把 6 株铜绿假单胞菌的 MIC 表读进 Python，按 CLSI 折点判 S/I/R，画一张对数轴柱状图
# 运行命令：conda activate amr_ai && python code/day01_first_python.py
# 预期输出：终端打印 1 张 MIC 表 + 1 张 S/I/R 判读表；outputs/day01_mic_bar.png（蓝=S 橙=I 红=R）
# 依赖：python 3.11+ / numpy 1.26.4 / pandas 2.2.2 / matplotlib 3.8.4（见根目录 requirements.txt）
# 作者备注：这个类比想说明什么 —— dict 就是移动端熟透的「Map / 对象字面量」，
#           DataFrame 是「内存里可编程的 Excel」，matplotlib 是「把 Excel 图表搬进代码」。
#           这个脚本**没有**演示任何 AI/ML：Day 01 只证明一件事——
#           「检验人每天在实验室看的那张表，用代码就能读进来、判出来、画出来」。
# ============================================================

from pathlib import Path                      # 标准库：跨平台路径，≈ 电脑里的资源管理器，不用手拼 '/' 或 '\'
import pandas as pd                           # 表格库：DataFrame ≈ 内存里可编程的 Excel
import matplotlib.pyplot as plt               # 绘图库：Figure ≈ 一块画布，Axes ≈ 画布上的一张图

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"   # 相对脚本自身定位：在哪个目录跑都能找到 outputs/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                     # 目录不存在就建，≈ mkdir -p，目录已存在也不报错

# 教学构造数据（非真实测定值）：6 株铜绿假单胞菌 × 3 种药物，MIC 单位 ug/mL（= mg/L）
strains = [                                   # 列表 ≈ 一串排好序的值；每个元素是一个键值对
    {"strain": "Pa-01", "meropenem": 0.5, "ciprofloxacin": 0.25,  "tobramycin": 1 },
    {"strain": "Pa-02", "meropenem": 1,   "ciprofloxacin": 0.5,   "tobramycin": 2 },
    {"strain": "Pa-03", "meropenem": 4,   "ciprofloxacin": 1,     "tobramycin": 4 },
    {"strain": "Pa-04", "meropenem": 8,   "ciprofloxacin": 2,     "tobramycin": 8 },
    {"strain": "Pa-05", "meropenem": 16,  "ciprofloxacin": 4,     "tobramycin": 16},
    {"strain": "Pa-06", "meropenem": 2,   "ciprofloxacin": 0.125, "tobramycin": 2 },
]

# CLSI M100 折点（铜绿假单胞菌，MIC ug/mL）：(S 上限, I 上限)，超过 I 上限即 R
# ⏰ 用时再查最新版 CLSI M100 / EUCAST —— 折点每年更新，妥布霉素已于 M100-Ed33(2023) 下调
BREAKPOINTS = {"meropenem": (2, 4), "ciprofloxacin": (0.5, 1), "tobramycin": (1, 2)}

def interpret(mic, s_max, i_max):             # 函数：把"心里有数"变成"代码里写死"
    if mic <= s_max: return "S"
    if mic <= i_max: return "I"
    return "R"

df = pd.DataFrame(strains).set_index("strain")                    # 菌株编号当索引，≈ Excel 里把首列设为行标题
print("=== MIC (ug/mL) ===\n", df)
calls = {d: [interpret(v, *BREAKPOINTS[d]) for v in df[d]] for d in df.columns}   # 字典推导式：一次判完全部三种药
print("\n=== S/I/R 判读 ===\n", pd.DataFrame(calls, index=df.index))

# 只画美罗培南：三种药折点不同（MEM 2/4 vs CIP 0.5/1 vs TOB 1/2），混在一根轴上画折线会误导读者
s_max, i_max = BREAKPOINTS["meropenem"]       # 拆包赋值 ≈ 一次取两个返回值
COLORS = {"S": "#1f77b4", "I": "#f39c12", "R": "#d62728"}         # 蓝=敏感 橙=中介 红=耐药
colors = [COLORS[c] for c in calls["meropenem"]]                  # 列表推导式 ≈ 一次完成筛选 + 收集
plt.figure(figsize=(8, 4.5))                                      # 画布尺寸（英寸）
plt.bar(df.index, df["meropenem"], color=colors)                  # 柱子：x=菌株 y=MIC，按判读结果上色
plt.axhline(s_max, ls="--", c="gray", lw=1)                       # 灰虚线 = S 上限
plt.axhline(i_max * 2, ls="--", c="black", lw=1)                  # 黑虚线 = R 下限（本文三种药均为 I 上限的下一个稀释档）
plt.text(0.02, s_max * 1.15, "S <= 2", fontsize=8)
plt.text(0.02, i_max * 2 * 1.15, "R >= 8", fontsize=8)
plt.yscale("log")                                                 # 对数轴：MIC 是倍比稀释，线性轴会把低值压成零
plt.ylabel("MIC (ug/mL, log scale)")                              # 图上用英文 + ASCII 单位：中文字形缺失会变方块
plt.title("P. aeruginosa: meropenem MIC (blue=S, amber=I, red=R)")
plt.tight_layout()                                                # 自动收紧边距，省得标签被裁掉

out_png = OUTPUT_DIR / "day01_mic_bar.png"
plt.savefig(out_png, dpi=200)                                     # dpi=200 → 1600x900，公众号正文够用且加载快
print(f"\n图已保存：{out_png}")
if "agg" not in plt.get_backend().lower(): plt.show()             # 有图形界面才弹窗；纯终端/CI 环境自动跳过

print("\n今天我学会了：判 R/S 的规则一旦写进代码，6 株和 6 万株跑的是同一条规则；pandas 把 Excel 搬进了代码，matplotlib 让图表可复现。")
