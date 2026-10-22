# ============================================================
# amr-ai-lab / Day 02 — 把一张表拆成「特征」和「标签」
# ------------------------------------------------------------
# 对应文章：articles/day02_features_labels.md      （文章与脚本同名前缀，见 resources/CODE_CONVENTION.md）
# 一句话作用：读出同一张 MIC 表里的特征矩阵 X 与标签列 y，并数清每列的缺失值
# 运行命令：conda activate amr_ai && python code/day02_features_labels.py
# 预期输出：终端打印 1 张带标签列的表 + S/I/R 计数（S=3 I=1 R=2）+ 每列缺失数（ciprofloxacin=1）+ X.shape=(6,3)；不产出图片
# 依赖：python 3.11+ / numpy 1.26.4 / pandas 2.2.2（见根目录 requirements.txt；本脚本纯终端输出，无需 matplotlib）
# 作者备注：这个类比想说明什么 —— 检验人每天看的那张药敏表，天然就是 X 和 y 两半：
#           三个药的 MIC 列 ≈ 一排生化反应管（API 鉴定条上每管一个反应，管子读数 = 特征），
#           判读结论那一列 ≈ 你要判读的答案（敏感 or 耐药 = 标签），
#           缺失值 ≈ 某管没做或浑浊读不出 —— AI 见了会卡住，所以今天先学会「数它」。
#           这个脚本**不**演示：训练任何模型（Day 14 才第一次上模型）、
#           缺失值填充（今天只数、只说危害）、画图（Day 02 无出图要求）。
# ============================================================

import pandas as pd                           # 表格库：DataFrame ≈ 内存里可编程的 Excel

# 教学构造值，非真实测定值：与 Day 01 同一批 6 株铜绿假单胞菌 × 3 种药，MIC 单位 ug/mL（= mg/L）
# 与 Day 01 的唯一差异：Pa-06 的 ciprofloxacin 改成 None —— 模拟「检测项没做或失败」，pandas 读入即 NaN
strains = [
    {"strain": "Pa-01", "meropenem": 0.5, "ciprofloxacin": 0.25,  "tobramycin": 1 },
    {"strain": "Pa-02", "meropenem": 1,   "ciprofloxacin": 0.5,   "tobramycin": 2 },
    {"strain": "Pa-03", "meropenem": 4,   "ciprofloxacin": 1,     "tobramycin": 4 },
    {"strain": "Pa-04", "meropenem": 8,   "ciprofloxacin": 2,     "tobramycin": 8 },
    {"strain": "Pa-05", "meropenem": 16,  "ciprofloxacin": 4,     "tobramycin": 16},
    {"strain": "Pa-06", "meropenem": 2,   "ciprofloxacin": None,  "tobramycin": 2 },
]

# CLSI M100 折点（铜绿假单胞菌，MIC ug/mL）：(S 上限, I 上限)，超过 I 上限即 R
# ⏰ 用时再查最新版 CLSI M100 / EUCAST —— 折点每年更新，妥布霉素已于 M100-Ed33(2023) 下调
BREAKPOINTS = {"meropenem": (2, 4)}

def interpret(mic, s_max, i_max):             # 判读函数沿用 Day 01：今天它负责生成「标签」
    if mic <= s_max: return "S"
    if mic <= i_max: return "I"
    return "R"

df = pd.DataFrame(strains).set_index("strain")                    # 菌株编号当索引，≈ Excel 里把首列设为行标题
df["label"] = [interpret(v, *BREAKPOINTS["meropenem"]) for v in df["meropenem"]]   # 按美罗培南判 S/I/R 作标签列（Pa-06 美罗培南不缺，判读不受缺失影响）
print("=== 整张表（特征列 + 标签列）===\n", df)

print("\n=== 标签列 y 的计数 ===\n", df["label"].value_counts())   # S=3 I=1 R=2，与 Day 01 的判读表一致

print("\n=== 每列缺失数（None 读进来就是 NaN）===\n", df.isna().sum())   # 缺失 ≈ 某管没做或浑浊读不出，只数、不填

feature_cols = ["meropenem", "ciprofloxacin", "tobramycin"]       # 特征：拿来推断的那些列（一排生化反应管的读数）
X = df[feature_cols]                                              # 特征矩阵 X ≈ 鉴定条上全部反应管的读数
y = df["label"]                                                   # 标签 y ≈ 你要判读的那个结论（敏感 or 耐药）
print(f"\n特征列：{feature_cols}；标签列：label")                  # 用手指着看：哪几列是特征、哪一列是标签
print(f"X.shape = {X.shape}（6 株 x 3 个特征）")
print(f"y.shape = {y.shape}（每株 1 个答案）")

print("\n今天我学会了：一张表天然分两半——特征是证据，标签是要猜的答案；缺失值是「没做的那管」，先数清它，AI 才不会卡住。")
