# OlmoEarth Embedding & KNN 聚类流程

这个仓库现在提供了一套从“指定区域 bbox”出发，自动获取 OlmoEarth 基础模型 embeddings，并进一步做 kNN / 聚类分析的完整流程。

你的最终目标有两个输出：

1. **该区域的 embedding**，保存为 `npy` 格式
2. **kNN / 聚类效果图**，保存为 `png` 或 `jpg` 格式

下面给出完整的使用说明、脚本职责、命令示例和推荐流程。

---

## 1. 依赖环境

建议在 Linux 环境中运行，并确保你已经安装了以下依赖：

- Python 3.10+
- `rslearn`
- `torch`
- `rasterio`
- `numpy`
- `scikit-learn`
- `matplotlib`

如果你是通过项目现有环境运行，确保 `rslearn` 的相关依赖已经可用。

---

## 2. 整体流程

推荐的流程如下：

### Step 1. 输入区域 bbox，自动下载并生成 OlmoEarth embeddings

使用：

- `scripts/compute_olmoearth_embeddings.py`

这个脚本会：

- 根据你输入的经纬度 bbox 创建一个工作目录
- 自动生成 rslearn 配置
- 自动从网上拉取 Sentinel 数据
- 自动 materialize 数据
- 跑 OlmoEarth encoder 推理
- 把 embeddings 写入 rslearn 结果目录

### Step 2. 导出 embedding raster 为 `npy`

使用：

- `scripts/extract_olmoearth_embeddings.py`

这个脚本会：

- 读取 rslearn 输出的 `embeddings` GeoTIFF
- 按窗口做池化
- 导出成适合后续分析的 `embeddings.npy`
- 同时保存 `metadata.csv`

### Step 3. 对 embedding 做 kNN / 聚类

使用：

- `scripts/knn_cluster.py`

这个脚本会：

- 读取 `embeddings.npy`
- 计算每个样本的 k 个最近邻
- 可选运行 KMeans 聚类
- 输出 CSV 和 `npy` 结果

### Step 4. 画出聚类效果图

如果你最终需要一个 `png/jpg` 图，可以在 `knn_cluster.py` 输出的 `clusters.csv` 或 `cluster_labels.npy` 基础上进一步做可视化。

你可以把聚类结果和 `metadata.csv` 联合起来，在地图上着色，或者画成二维降维图。

本仓库后续也可以继续补一个专门的可视化脚本。

---

## 3. 脚本说明

---

### 3.1 `scripts/compute_olmoearth_embeddings.py`

#### 功能

输入一个 bbox，自动完成：

- 创建运行目录
- 写入 rslearn dataset 配置
- 添加窗口
- 下载并预处理遥感影像
- 运行 OlmoEarth 推理
- 输出 embedding raster

#### 命令示例

```bash
python3 scripts/compute_olmoearth_embeddings.py \
  --box -122.4 47.6 -122.3 47.7 \
  --start 2024-01-01T00:00:00+00:00 \
  --end 2025-01-01T00:00:00+00:00
```

#### 参数说明

- `--box LON1 LAT1 LON2 LAT2`
  - 输入区域 bbox，坐标系为 `EPSG:4326`
- `--start`
  - 时间范围开始
- `--end`
  - 时间范围结束
- `--grid-size`
  - 如果区域较大，可以把大区域切成多个 window
- `--workers`
  - rslearn 下载与 materialize 的并发数量
- `--base-dir`
  - 工作目录根路径，默认 `./olmoearth_embedding_runs`
- `--skip-materialize`
  - 只生成配置和 window，不执行下载和推理

#### 运行后会生成什么

脚本会在 `--base-dir` 下创建一个运行目录，例如：

```text
./olmoearth_embedding_runs/bbox_-122.4_47.6_-122.3_47.7_20260414T123000Z/
```

内部一般会包含：

- `dataset/config.json`
- `model.yaml`
- `dataset/windows/.../layers/embeddings/.../geotiff.tif`


---

### 3.2 `scripts/extract_olmoearth_embeddings.py`

#### 功能

把 rslearn / OlmoEarth 生成的 embedding raster 导出成可用于机器学习的数组。

#### 命令示例

```bash
python3 scripts/extract_olmoearth_embeddings.py \
  --input-dir ./olmoearth_embedding_runs/bbox_-122.4_47.6_-122.3_47.7_20260414T123000Z/dataset/windows \
  --output-dir ./exported_embeddings
```

#### 参数说明

- `--input-dir`
  - 传入 rslearn 运行目录中的 `windows` 目录
- `--output-dir`
  - 输出目录
- `--glob`
  - 匹配 embedding GeoTIFF 的模式，默认：

```text
**/layers/embeddings/**/geotiff.tif
```

- `--pooling`
  - `mean`：对空间维度平均池化，输出一个向量
  - `max`：对空间维度最大池化，输出一个向量
  - `none`：保留完整 embedding map，不做池化

#### 输出文件

默认会生成：

- `embeddings.npy`
- `metadata.csv`

如果使用 `--pooling none`，会输出压缩文件：

- `embeddings.npz`

#### 推荐用法

如果你的最终目标是 kNN / 聚类，建议使用：

```bash
--pooling mean
```

这样每个 window 对应一个固定长度向量，最方便后续分析。

---

### 3.3 `scripts/knn_cluster.py`

#### 功能

对导出的 embeddings 做：

- kNN 搜索
- 可选 KMeans 聚类
- 输出结果文件

#### 命令示例：只做 kNN

```bash
python3 scripts/knn_cluster.py \
  --embeddings ./exported_embeddings/embeddings.npy \
  --metadata ./exported_embeddings/metadata.csv \
  --output-dir ./knn_results \
  --k 10
```

#### 命令示例：kNN + 聚类

```bash
python3 scripts/knn_cluster.py \
  --embeddings ./exported_embeddings/embeddings.npy \
  --metadata ./exported_embeddings/metadata.csv \
  --output-dir ./knn_results \
  --k 10 \
  --clusters 8
```

#### 参数说明

- `--embeddings`
  - 输入 `embeddings.npy` 或 `embeddings.npz`
- `--metadata`
  - 输入 `metadata.csv`
- `--output-dir`
  - 输出目录
- `--k`
  - 每个样本取多少个最近邻
- `--clusters`
  - 如果大于 0，就额外运行 KMeans
- `--metric`
  - 距离度量，默认 `cosine`
- `--random-state`
  - KMeans 随机种子

#### 输出文件

如果只做 kNN，会输出：

- `knn_neighbors.csv`
- `features_flat.npy`

如果启用聚类，还会输出：

- `clusters.csv`
- `cluster_labels.npy`
- `cluster_centers.npy`

### 3.4 `scripts/visualize_clusters.py`

#### 功能

把 `knn_cluster.py` 的结果画成二维散点图，输出 `png` 或 `jpg`。

#### 命令示例：使用 `cluster_labels.npy`

```bash
python3 scripts/visualize_clusters.py \
  --features ./knn_results/features_flat.npy \
  --clusters ./knn_results/cluster_labels.npy \
  --output ./knn_results/cluster_plot.png
```

#### 命令示例：使用 `clusters.csv`

```bash
python3 scripts/visualize_clusters.py \
  --features ./knn_results/features_flat.npy \
  --clusters-csv ./knn_results/clusters.csv \
  --output ./knn_results/cluster_plot.jpg
```

#### 参数说明

- `--features`
  - 输入 `features_flat.npy`
- `--clusters`
  - 输入 `cluster_labels.npy`
- `--clusters-csv`
  - 或输入 `clusters.csv`
- `--output`
  - 输出图片路径，支持 `.png` / `.jpg`
- `--title`
  - 图标题，默认 `OlmoEarth Cluster Visualization`

#### 输出文件

- `cluster_plot.png` 或 `cluster_plot.jpg`

---

## 4. 推荐的完整流程

下面给你一个从头到尾的推荐流程。

### 第一步：计算区域 embeddings

```bash
python3 scripts/compute_olmoearth_embeddings.py \
  --box -122.4 47.6 -122.3 47.7 \
  --start 2024-01-01T00:00:00+00:00 \
  --end 2025-01-01T00:00:00+00:00
```

### 第二步：导出 embedding 向量

```bash
python3 scripts/extract_olmoearth_embeddings.py \
  --input-dir ./olmoearth_embedding_runs/bbox_-122.4_47.6_-122.3_47.7_20260414T123000Z/dataset/windows \
  --output-dir ./exported_embeddings
```

输出：

- `./exported_embeddings/embeddings.npy`
- `./exported_embeddings/metadata.csv`

### 第三步：跑 kNN + 聚类

```bash
python3 scripts/knn_cluster.py \
  --embeddings ./exported_embeddings/embeddings.npy \
  --metadata ./exported_embeddings/metadata.csv \
  --output-dir ./knn_results \
  --k 10 \
  --clusters 8
```

输出：

- `./knn_results/knn_neighbors.csv`
- `./knn_results/clusters.csv`
- `./knn_results/cluster_labels.npy`
- `./knn_results/cluster_centers.npy`

### 第四步：画聚类图

```bash
python3 scripts/visualize_clusters.py \
  --features ./knn_results/features_flat.npy \
  --clusters ./knn_results/cluster_labels.npy \
  --output ./knn_results/cluster_plot.png
```

输出：

- `./knn_results/cluster_plot.png`

---

## 5. 你的最终目标输出

你提到最终至少要有两个输出：

### 输出 1：该区域的 embedding

建议保存为：

```text
embeddings.npy
```

这是一个 NumPy 数组，通常形状类似：

- `N x D`

其中：

- `N` = 样本数 / window 数
- `D` = embedding 维度

### 输出 2：kNN 聚类效果图

当前脚本已经能产出聚类标签，但还没有自动画图。

你可以基于：

- `clusters.csv`
- `metadata.csv`

进一步生成：

- `png`
- `jpg`

建议的图有两种：

1. **二维降维图**
   - PCA / t-SNE / UMAP 后按 cluster 上色
2. **地图可视化图**
   - 把每个 window / polygon 的聚类结果画到地图上

如果你愿意，我下一步可以继续帮你补一个 `visualize_clusters.py`，直接把 `clusters.csv` 画成 `png`。

---

## 6. 常见注意事项

### 6.1 区域太大

如果 bbox 很大，建议加：

```bash
--grid-size 1024
```

这样会自动切成多个 window，避免单个窗口太大。

### 6.2 时间范围

如果你是为了聚类区域语义，一般可以设置较长时间范围，比如一年。

如果你是想看某一时刻附近的变化，可以缩短时间范围。

### 6.3 `pooling` 选择

如果你要做 kNN / 聚类，最稳妥的是：

```bash
--pooling mean
```

这样每个 window 变成一个固定长度特征向量，更适合后续分析。

---

## 7. 建议的下一步

如果你希望整个流程真正闭环，我建议下一步我继续帮你补一个脚本：

- `scripts/visualize_clusters.py`

它可以：

- 读取 `clusters.csv`
- 结合 `metadata.csv`
- 输出聚类可视化图 `png/jpg`

这样就能完整满足你说的两个最终产物：

- `embeddings.npy`
- `cluster_effect.png`

如果你要，我下一条就直接帮你写这个可视化脚本。