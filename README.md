# EuroSAT 三层 MLP 作业框架

这个项目是一个**不依赖 PyTorch / TensorFlow / JAX**的三层 MLP 图像分类器框架，面向 EuroSAT_RGB 数据集。

## 项目结构

```text
.
├── eurosat_mlp/
│   ├── __init__.py
│   ├── config.py          # 实验配置
│   ├── data.py            # 数据读取、划分、标准化、batch 迭代
│   ├── tools.py           # Layers、交叉熵、评价指标、SGD、JSON保存
│   ├── model.py           # 三层 MLP
│   ├── trainer.py         # 训练、测试、网格搜索
│   └── visualize.py       # 训练曲线、混淆矩阵、权重可视化
├── train.py               # 单次训练脚本
├── evaluate.py            # 导入最优权重并测试
├── search.py              # 网格搜索示例
├── visualize_weights.py   # 可视化图片的第一层隐藏层
├── outputs/               # 训练输出目录
├── requirements.txt
└── README.md
```

## 环境依赖

```bash
pip install -r requirements.txt
```

## 数据目录

默认数据目录为：

```text
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## 训练

```bash
python3 train.py \
  --data_root EuroSAT_RGB \
  --output_dir outputs/default_run \
  --hidden_dim1 256 \
  --hidden_dim2 128 \
  --activation relu \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.01 \
  --lr_decay 0.95 \
  --weight_decay 5e-5
```

训练完成后，会在 `output_dir` 下保存：

- `best_model.npz`：验证集最优权重
- `config.json`：训练配置
- `split_indices.npz`：训练/验证/测试划分
- `history.json`：训练历史
- `summary.json`：最佳验证集准确率与测试集准确率
- `confusion_matrix.csv / .npy`
- `plots/loss_curves.png`
- `plots/val_accuracy.png`
- `plots/confusion_matrix.png`

## 导入训练好的权重测试

训练好的权重位于 `outputs/best_run/best_model.npz`

```bash
python3 evaluate.py --run_dir outputs/best_run
```

测试结果会保存到 `outputs/best_run/eval/` ，包括混淆矩阵和分类错误的标签

## 网格搜索最优参数

```bash
python3 search.py
```

搜索结果会保存到 `outputs/search_runs/search_results.json`。

## 第一层权重可视化

```bash
python scripts/visualize_weights.py --run_dir outputs/best_run --max_neurons 256
```

结果会保存到 `outputs/best_run/plots/first_layer_weight/`。
