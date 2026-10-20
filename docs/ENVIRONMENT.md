# ENVIRONMENT.md — 30 分钟搞定环境（amr-ai-lab）

> 面向：完全没装过 Python、但有移动端开发经验的读者
> 阅读时间：5 分钟；操作时间：约 30 分钟（视网速）
> 本项目全程**不需要 GPU**（阶段 2 也只装 CPU 版 torch）

---

## 0. 时间账

| 步骤 | 内容 | 耗时 |
|---|---|---|
| 1 | 安装 Miniforge | 5-10 分钟 |
| 2 | 建环境 + 装包（档位 1） | 10-15 分钟 |
| 3 | 验证 + 启动 Jupyter | 5 分钟 |
| （阶段 2 再做） | 补装 torch（档位 2） | 10-20 分钟 |

---

## 1. 第一步：安装 Miniforge（≈ 给电脑装一个「依赖管理器」）

Miniforge 是 conda 的社区版发行版：**装它 = 同时得到 conda（环境管理）+ 一个干净的 Python**。类比移动端：相当于把 Gradle / CocoaPods / SPM 和一台干净的开发机一起配好。

### macOS（Intel / Apple Silicon 通用）

```bash
brew install --cask miniforge
```

装完**重开一个终端窗口**，敲 `conda --version` 能出版本号就成功。

> 没装 Homebrew？去 `https://github.com/conda-forge/miniforge/releases` 下载 `Miniforge3-MacOSX-<架构>.pkg` 双击安装。⏰ 用时再查最新 release 文件名。

### Windows

1. 去 `https://github.com/conda-forge/miniforge/releases` 下载 **`Miniforge3-Windows-x86_64.exe`**（⏰ 用时再查最新版本）
2. 双击安装，选项全部默认即可（默认**不**加入 PATH——没问题，我们用开始菜单里的 **Miniforge Prompt**）
3. 从开始菜单打开 **Miniforge Prompt**，后续所有命令都在这个窗口里敲

### Linux

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"   # ⏰ 用时再查
bash "Miniforge3-$(uname)-$(uname -m).sh"     # 一路 yes / 回车，默认装到 ~/miniforge3
source ~/.bashrc                              # zsh 用户改用：source ~/.zshrc
```

---

## 2. 第二步：建环境 + 装包（国内镜像加速）

**在项目根目录 `amr-ai-lab/` 打开终端**（Windows 用 Miniforge Prompt），逐行复制：

```bash
# 1) 建一个叫 amr_ai 的独立环境，锁 Python 3.11
#    类比移动端：给这个项目单开一个 workspace，依赖不污染全局
conda create -n amr_ai python=3.11 -y

# 2) 激活环境（每次开新终端都要先执行这一句，提示符前会出现 (amr_ai)）
conda activate amr_ai

# 3) 装档位 1 依赖（约 150MB，10-15 分钟；-i 指定清华镜像加速，可换成你自己的源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 \
    jupyterlab==4.2.4 ipykernel==6.29.4 \
    scikit-learn==1.5.0 scipy==1.13.1 biopython==1.83 requests==2.32.5
```

> 为什么连 scipy 也锁：不写 scipy 时 pip 会自动拉到 1.17+，那是 scikit-learn 1.5.0 发布之后很久的版本，两代产品的组合没人联调过；锁 1.13.1（与 sklearn 1.5.0 同期）最稳。
> 为什么 requests 是 2.32.5 不是 2.32.0：2.32.0 已被 PyPI 撤回（yanked），装的时候 pip 会告警。

> 版本为什么锁这么死、numpy 为什么必须 1.26.4：见根目录 `requirements.txt` 注释与本文「必踩坑清单」第 1 条。

> Python 版本口径：3.11–3.13 实测均可运行本项目，文档安装步骤按 3.11 走，照抄即可，不必特意换版本。

**（阶段 2 才做）补装档位 2：CPU 版 torch**

```bash
# ⚠️ 必须走 CPU index：默认源在 Linux 上会给 CUDA 版（2GB+，你用不上）
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.40.2 tokenizers==0.19.1
```

---

## 3. 第三步：验证（30 秒）

```bash
python -c "import pandas; print(pandas.__version__)"
```

输出 `2.2.2` 就成了。顺手把整组也验一遍：

```bash
python -c "import numpy, pandas, matplotlib; print(numpy.__version__, pandas.__version__, matplotlib.__version__)"
```

预期输出：`1.26.4 2.2.2 3.8.4`

---

## 4. 第四步：启动 Jupyter，跑第一个 notebook

```bash
# 在项目根目录执行；浏览器会自动打开
jupyter lab
```

1. 左侧文件树进入 `code/`，点 **Python 3 (ipykernel)** 新建 notebook（内核选 `amr_ai` 环境对应的 Python）
2. 新建 `notebooks/day01_demo.ipynb`，把 `code/day01_first_python.py` 的内容分 2-3 个 cell 粘进去，逐 cell 按 **Shift+Enter** 运行——这就是「边写边看」的即时反馈，≈ SwiftUI Preview
3. 用完在终端按 `Ctrl+C` 两次退出 Jupyter

> 教学脚本都设计成「命令行直接跑」：`python code/day01_first_python.py`。Jupyter 是你的实验台，脚本是存档的成品，两者内容一致。

---

## 5. 必踩坑清单（都是真实踩过的，附救援命令）

### 坑 ① numpy 2.x 与 torch 2.2.2 二进制不兼容

**症状**：`import torch` 报 `numpy.dtype size changed, may indicate binary incompatibility`，或任何 torch 相关 import 直接崩。
**原因**：torch 2.2.2 按 numpy 1.x 的 ABI 编译，numpy 2.x 的 C 层结构变了。
**救援**：

```bash
pip install "numpy==1.26.4"     # 降回 1.x，立即恢复
```

**预防**：永远用 `requirements.txt` 里的锁版本装，别单独升级 numpy。`pip install -U numpy` 是本项目第一大禁忌。

### 坑 ② Hugging Face 模型下载慢 / 超时（阶段 2 必踩）

**症状**：`OSError: Can't load tokenizer for 'facebook/esm2_t6_8M_UR50D'`、连接超时。
**救援**（跑代码前设置镜像环境变量）：

```bash
export HF_ENDPOINT=https://hf-mirror.com        # macOS / Linux
```

```bat
set HF_ENDPOINT=https://hf-mirror.com           :: Windows CMD
```

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"      # Windows PowerShell
```

也可以直接写进脚本最顶部（`import transformers` 之前）：
`import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`
（hf-mirror 为第三方镜像站，⏰ 用时再查可用性与最新说明）

### 坑 ③ 路径里有空格或中文

**症状**：命令找不到文件、脚本路径解析错乱、某些包报编码错误。
**军规**：项目和数据目录**只用英文、数字、连字符、下划线**。
- ✅ `D:\learn\amr-ai-lab`　✅ `~/WorkBuddy/amr-ai-lab`
- ❌ `D:\学习\生信 AI 教程`　❌ `~/Documents/我的 项目`

### 坑 ④ macOS 装完 Miniforge 提示 `command not found: conda`

**原因**：安装器改了 shell 配置，但当前终端没重新加载。
**救援**：

```bash
source ~/.zshrc        # zsh（macOS 新版默认）；bash 用户用 source ~/.bash_profile
```

还不行就**重开一个终端窗口**。

### 坑 ⑤ Windows PowerShell 报「禁止运行脚本」

**症状**：`conda activate` 报 `...cannot be loaded because running scripts is disabled on this system`。
**救援**（管理员或当前用户级别放开一次即可）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后**关掉重开** Miniforge Prompt。或者干脆绕开：全程用开始菜单的 **Miniforge Prompt**，它自带正确的初始化。

---

## 6. You are ready（自检清单）

全部打勾 = 环境合格，可以开始 Day 01：

- [ ] `conda activate amr_ai` 不报错，提示符前面出现 `(amr_ai)`
- [ ] `python -c "import pandas; print(pandas.__version__)"` 打印 `2.2.2`
- [ ] `python -c "import numpy, pandas, matplotlib; print(numpy.__version__, pandas.__version__, matplotlib.__version__)"` 打印 `1.26.4 2.2.2 3.8.4`
- [ ] 在项目根目录执行 `python code/day01_first_python.py`，终端打印 6 行判定，`outputs/day01_mic_bar.png` 出现
- [ ] `jupyter lab` 能打开浏览器并新建 notebook
- [ ] 项目路径里没有空格、没有中文

---

*维护：管一鸣 ｜ 版本快照日期：2026-10-13 ｜ 所有外部版本与链接均带 ⏰ 标记，半年后重装前请复核*
