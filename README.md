<p align="center">
    <img src="docs/abacus-logo.svg">
</p>

<p align="center">
    <a href="https://github.com/deepmodeling/abacus-develop/actions/workflows/test.yml">
        <img src="https://github.com/deepmodeling/abacus-develop/actions/workflows/test.yml/badge.svg">
    </a>
</p>

<a id="readme-top"></a>

# About ABACUS

ABACUS (**A**tomic-orbital **B**ased **A**b-initio **C**omputation at **US**tc) is an open-source package based on density functional theory (DFT). The package utilizes both plane wave and numerical atomic basis sets with the usage of pseudopotentials to describe the interactions between nuclear ions and valence electrons. ABACUS supports LDA, GGA, meta-GGA, and hybrid functionals. Apart from single-point calculations, the package allows geometry optimizations and ab-initio molecular dynamics with various ensembles. The package also provides a variety of advanced functionalities for simulating materials, including the DFT+U, VdW corrections, and implicit solvation model, etc. In addition, ABACUS strives to provide a general infrastructure to facilitate the developments and applications of novel machine-learning-assisted DFT methods (DeePKS, DP-GEN, DeepH, DeePTB etc.) in molecular and material simulations.

# Online Documentation

For detailed documentation, please refer to [our documentation website](https://abacus.deepmodeling.com/).

See our [Github Pages](https://mcresearch.github.io/abacus-user-guide/) for more tutorials and developer guides.

## 开发仓库实际操作标准指令

为了保证三人小组的代码协作顺畅，避免冲突并落实 Code Review 机制，请严格按照以下标准 Git 指令进行开发。

### 1 开始开发新功能

在开始编写新功能前，请确保基于最新的 `develop` 分支创建你的专属 feature 分支（建议按功能命名，例如钟煜晨开发 MPI 归约，可命名为 `feature/mpi-reduce`）：

```bash
#！重要！切换到abacus的路径
cd /abacus-develop

# 切换到 develop 分支
git checkout develop

# 拉取最新的 develop 代码，确保本地代码是最新的
git pull origin develop

# 基于 develop 创建并切换到新功能分支 (例如开发 MPI 归约优化)
git checkout -b feature/mpi-reduce
```

### 2 提交代码

在你的功能分支上完成部分代码编写或优化后，将代码提交到本地。建议**少量多次**提交，每个提交解决一个具体问题：

```bash
# 查看当前更改状态
git status

# 将更改的文件添加到暂存区 (或指定具体文件 git add <file>)
git add .

# 提交更改，写明规范的提交信息 (使用 perf/feat/fix 等前缀)
git commit -m "perf: 替换 MPI_Barrier 为非阻塞通信"
```

### 3 推送并合并代码

功能开发完成并自测无误后，将分支推送到远程仓库，并准备合并到 `develop` 分支。

**推荐操作（落实交叉 Review）**：在 GitHub/GitLab 等托管平台上，从 `feature/mpi-reduce` 向 `develop` 分支发起 Pull Request (PR)，由队友进行代码审查后合并。

```bash
# 将当前功能分支推送到远程仓库
git push -u origin feature/mpi-reduce
```

**本地直接合并（如果无需 Code Review）**：

```bash
# 切换回 develop 分支
git checkout develop

# 拉取最新 develop 避免冲突
git pull origin develop

# 将功能分支合并到 develop
git merge feature/mpi-reduce

# 推送合并后的 develop 到远程
git push origin develop

# 删除本地的临时功能分支（可选，保持分支列表整洁）
git branch -d feature/mpi-reduce
```

