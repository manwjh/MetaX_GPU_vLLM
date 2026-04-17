# 发布包目录说明

解压 **`muxi_gemma4_26B_TP1.tar.gz`** 后，根目录结构如下（**本文件随包提供**）。

```
muxi_gemma4_26B_TP1/
├── README.md                 # 总入口：适用前提、构建、运行、验收、补丁说明
├── Dockerfile                # 镜像构建（含 Transformers 5.x 等）
├── .env.example              # 环境变量模板（复制为 .env）
├── docs/                     # 给人看的说明文档（无 Python 入口）
│   ├── LAYOUT.md             # 本文件：目录地图
│   ├── QUICKSTART.md         # 最短上手
│   └── OFFLINE_DISTRIBUTION.md   # docker save/load 离线分发
├── patches/                  # 审计用补丁副本与 unified diff（非独立安装步骤）
│   ├── reasoning/
│   ├── moe/
│   └── exported/
├── scripts/                  # 容器入口、打补丁、启动 vLLM、烟测与包内静态检查
└── verification/             # 发版验证：报告 + 随包证据片段
    ├── README.md
    ├── reports/              # 例：2026-04-17_release.md
    └── evidence/             # 例：make check、包内 static verify、长沙矩阵摘录
```

**原则：** 根目录只放「一眼要用的」三件事——**说明**（`README.md`）、**怎么构建镜像**（`Dockerfile`）、**参数模板**（`.env.example`）；其余按 **`docs` / `patches` / `scripts` / `verification`** 四类归档。
