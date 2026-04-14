# Potsdam 从“原始数据”开始微调 OlmoEarth（RGBIR）

> 这份文档只假设你现在**只有**两个原始文件夹：
>
> - `4_Ortho_RGBIR`（原图）
> - `5_Labels_all`（RGB 颜色标签）
>
> 其余文件都会在流程中自动生成。

---

## 0. 你当前起点（唯一前提）

假设 Potsdam 根目录是：`D:\ZHC\数据集\Potsdam`

里面只有：

- `D:\ZHC\数据集\Potsdam\4_Ortho_RGBIR`
- `D:\ZHC\数据集\Potsdam\5_Labels_all`

---

## 1. 设置路径变量（PowerShell）

在仓库根目录执行：

```powershell
$POTSDAM_ROOT="D:\ZHC\数据集\Potsdam"
$IMG_DIR="$POTSDAM_ROOT\4_Ortho_RGBIR"
$LBL_RGB_DIR="$POTSDAM_ROOT\5_Labels_all"

# 以下三个目录是流程中“新生成”的（你一开始没有是正常的）
$LBL_INDEX_DIR="$POTSDAM_ROOT\5_Labels_index"
$PROJECT_PATH="C:\Users\Aunitily\Desktop\olmoearth_projects-main\olmoearth_run_data\potsdam"
$OER_DATASET_PATH="$POTSDAM_ROOT\oer_dataset"
```

---

## 2. 先确认数据格式（可选但建议）

```powershell
python scripts/inspect_tif_type.py --path "$IMG_DIR\top_potsdam_2_10_RGBIR.tif"
python scripts/inspect_tif_type.py --path "$LBL_RGB_DIR\top_potsdam_2_10_label.tif"
```

你应看到大致结论：

- 原图：4 波段（RGBIR）
- 标签：3 波段且颜色离散（颜色编码标签）

---

## 3. 第一次“生成新目录”：把 RGB 标签转成 index 标签

你原始标签是 RGB 颜色图，训练前需要转成单通道类别图。

```powershell
python scripts/potsdam_colorlabel_to_index.py `
  --image-dir "$IMG_DIR" `
  --label-dir "$LBL_RGB_DIR" `
  --out-dir "$LBL_INDEX_DIR"
```

执行后会**自动生成**：

- `D:\ZHC\数据集\Potsdam\5_Labels_index`

里面是：`*_label_index.tif`

---

## 4. 第二次“生成新文件”：标准 annotation geojson

```powershell
python scripts/potsdam_to_oer_annotations.py `
  --image-dir "$IMG_DIR" `
  --label-dir "$LBL_INDEX_DIR" `
  --outdir "$PROJECT_PATH" `
  --start-time "2024-01-01T00:00:00+00:00" `
  --end-time "2024-12-31T23:59:59+00:00" `
  --ignore-values 255 `
  --min-pixels 16
```

执行后会在 `$PROJECT_PATH` 下生成：

- `annotation_task_features.geojson`
- `annotation_features.geojson`

---

## 5. 准备配置文件（仓库内）

确保下列配置存在于：`olmoearth_run_data/potsdam/`

- `dataset.json`
- `model.yaml`
- `olmoearth_run.yaml`

（我已经按 RGBIR 路线给你写过这三份）

---

## 6. 构建训练窗口

```powershell
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows `
  --project_path "$PROJECT_PATH" `
  --scratch_path "$OER_DATASET_PATH"
```

---

## 7. 从窗口构建数据集

```powershell
python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows `
  --project_path "$PROJECT_PATH" `
  --scratch_path "$OER_DATASET_PATH"
```

---

## 7.5 关键检查：`potsdam_rgbir` 输入层是否真的准备好了（必须）

这一步是为了避免“训练启动后才报找不到输入层”。

### 先检查是否有窗口元数据

```powershell
Get-ChildItem "$OER_DATASET_PATH\dataset\windows" -Recurse -Filter metadata.json | Select-Object -First 5
```

如果一个都没有，说明第 6 步没有成功。

### 再检查 `potsdam_rgbir` 图层是否存在

```powershell
Get-ChildItem "$OER_DATASET_PATH\dataset\layers" -Recurse | Select-String "potsdam_rgbir"
```

如果这里也找不到 `potsdam_rgbir`，通常是“本地图像层没有被成功导入到 dataset”。

### 如果 `potsdam_rgbir` 缺失，怎么做

你有两个选择：

1. **先继续调通标签流程（推荐）**：
   - 保留当前步骤，确认 label/window/split 都正确。
2. **补本地图像导入配置（下一步）**：
   - 需要在 `dataset.json` 里为 `potsdam_rgbir` 增加本地数据来源配置，或增加导入步骤，把 `4_Ortho_RGBIR` 映射进 dataset 图层。

> 注意：这一步和 Sentinel 教程不同。Sentinel 有 `data_source` 自动下载；Potsdam 本地数据需要明确“从本地哪里读”。

---

## 8. 训练前环境变量

```powershell
$env:DATASET_PATH="$OER_DATASET_PATH\dataset"
$env:TRAINER_DATA_PATH="$OER_DATASET_PATH\trainer_checkpoints"
$env:PREDICTION_OUTPUT_LAYER="output"
$env:NUM_WORKERS="8"

$env:WANDB_PROJECT="potsdam_rgbir_finetune"
$env:WANDB_NAME="oe_potsdam_rgbir_unet_p4"
$env:WANDB_ENTITY="your_wandb_entity"
```

---

## 9. 启动微调

```powershell
python -m olmoearth_projects.main olmoearth_run finetune `
  --project_path "$PROJECT_PATH" `
  --scratch_path "$OER_DATASET_PATH"
```

---

## 10. 你最初没有的目录/文件，分别在何时生成

你说得对，这三类你起初没有是正常的：

1. `5_Labels_index`：第 3 步生成
2. `annotation_task_features.geojson / annotation_features.geojson`：第 4 步生成
3. `oer_dataset`：第 6/7 步生成

---

## 11. 你之前遇到的 forkserver 报错

Windows 下确实不支持 `forkserver`。仓库已改为：

- Windows -> `spawn`
- Linux/macOS -> `forkserver`

所以现在可以继续在 Windows 跑预处理；训练建议在 Linux GPU 环境跑更稳。
