

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("./outputs/dnrti_train.txt")

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.savefig('./outputs/validation_metrics_dnrti_loss.png', bbox_inches='tight', dpi=300)
plt.show()

# 绘制验证指标曲线
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Validation Acc'], label='Validation Accuracy', color='green')
plt.plot(data['Epoch'], data['Validation Precision'], label='Validation Precision', color='red')
plt.plot(data['Epoch'], data['Validation Recall'], label='Validation Recall', color='purple')
plt.plot(data['Epoch'], data['Validation F1'], label='Validation F1 Score', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Validation Metrics Curve')
plt.legend()
plt.grid()
plt.savefig('./outputs/validation_metrics_dnrti_metric.png', bbox_inches='tight', dpi=300)
plt.show()

# 绘制精确率-召回率曲线
plt.figure(figsize=(20, 12))
plt.plot(data['Validation Recall'], data['Validation Precision'], marker='o', linestyle='-', color='brown')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall')
plt.grid()
plt.savefig('./outputs/validation_metrics_dnrti_recall.png', bbox_inches='tight', dpi=300)
plt.show()
