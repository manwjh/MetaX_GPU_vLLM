# `muxi_gemma4_26B_TP1` 离线分发说明

这份文档用于**不能直接访问镜像仓库**的场景。

## 1. 两种离线分发方式

### 方式 A：只分发源码包

你把下面这个包发给对方：

- `muxi_gemma4_26B_TP1.tar.gz`

对方自行解压、再在本地 `docker build`。

前提：

- 对方机器仍能访问 `BASE_IMAGE`
- 或者对方机器已经本地有对应基础镜像

### 方式 B：连构建好的镜像一起分发

如果对方机器**不能访问沐曦镜像仓库**，推荐你在一台能 build 的机器上先构建，再导出镜像。

## 2. 在可联网机器上构建并导出

### 2.1 构建镜像

```bash
cp .env.example .env
source .env
docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${IMAGE_NAME}" .
```

**说明：** 发布包 `Dockerfile` 在构建阶段会安装 **Transformers 5.5.0**（Gemma4 所需）。若对方未执行 `docker build`、仅把文件塞进已有镜像，会复现 `gemma4` 架构不被识别的错误。

### 2.2 导出镜像

```bash
source .env
docker save "${IMAGE_NAME}" -o muxi_gemma4_26B_TP1_image.tar
gzip -f muxi_gemma4_26B_TP1_image.tar
```

如果镜像太大，可以分卷：

```bash
split -b 8G muxi_gemma4_26B_TP1_image.tar.gz muxi_gemma4_26B_TP1_image.tar.gz.part-
```

建议一起发给对方的内容：

1. `muxi_gemma4_26B_TP1.tar.gz`
2. `muxi_gemma4_26B_TP1_image.tar.gz` 或分卷文件
3. 一份你实际使用过的 `.env`

## 3. 在离线机器上导入

### 3.1 如果是分卷文件，先合并

```bash
cat muxi_gemma4_26B_TP1_image.tar.gz.part-* > muxi_gemma4_26B_TP1_image.tar.gz
```

### 3.2 解压镜像包

```bash
gunzip muxi_gemma4_26B_TP1_image.tar.gz
```

### 3.3 导入 Docker 镜像

```bash
docker load -i muxi_gemma4_26B_TP1_image.tar
```

导入后可用下面命令确认：

```bash
docker images | rg muxi_gemma4_26B_TP1
```

## 4. 在离线机器上使用

### 4.1 解压源码包

```bash
tar -xzf muxi_gemma4_26B_TP1.tar.gz
cd muxi_gemma4_26B_TP1
```

### 4.2 如果镜像已导入，就不必再次 build

只需要：

```bash
cp .env.example .env
```

然后把 `.env` 里的 `IMAGE_NAME` 改成你导入后的镜像名。

### 4.3 启动容器

按照 `README.md` 或 `docs/QUICKSTART.md` 的 `docker run` 命令启动。

## 5. 离线分发时最容易漏的东西

1. **模型权重目录**：这个包不包含权重，必须单独准备
2. **基础镜像**：如果对方不能联网拉取，就必须一起 `docker save`
3. **`.env`**：建议把你现场验证过的那份也一起发
4. **宿主机 GPU/MACA 环境**：发布包不负责安装驱动

## 6. 最简建议

如果你就是想让别人“尽量少踩坑”，建议直接给对方这三样：

1. `muxi_gemma4_26B_TP1.tar.gz`
2. `muxi_gemma4_26B_TP1_image.tar.gz`
3. 你已经验证过的 `.env`

这样对方通常只需要：

1. `docker load`
2. 解压源码包
3. 挂模型目录
4. 跑 `docker run`

就能开始试。
