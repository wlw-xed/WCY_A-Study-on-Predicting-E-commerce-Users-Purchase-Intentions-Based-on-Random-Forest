# 基于随机森林的电商用户购买意愿预测研究

作者：王彩云  
学号：202521511226

## 数据来源

UCI Machine Learning Repository：Online Shoppers Purchasing Intention Dataset  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

代码包内已包含论文实验使用的 `online_shoppers_intention.csv`，文件小于 25 MB。

## 文件说明

- `基于随机森林的电商用户购买意愿预测研究.py`：完整的数据清洗、基础分析、模型训练、随机参数搜索、阈值优化、交叉验证、绘图和敏感性分析代码。
- `online_shoppers_intention.csv`：原始数据。
- `requirements.txt`：Python 依赖。

## 运行方法

推荐 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
python 基于随机森林的电商用户购买意愿预测研究.py \
  --data online_shoppers_intention.csv \
  --output results
```

运行结束后，`results/figures` 保存论文图像，`results/tables` 保存表格数据，结果目录根部保存最优参数、阈值和关键指标 JSON 文件。

## 可复现设置

随机种子固定为 42。由于操作系统、Python/依赖版本以及并行线程调度差异，末位小数可能有轻微变化。
