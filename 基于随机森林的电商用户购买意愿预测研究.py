# -*- coding: utf-8 -*-
"""基于随机森林的电商用户购买意愿预测：数据分析与模型复现实验。"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

warnings.filterwarnings('ignore')

SEED = 42
parser = argparse.ArgumentParser(description='基于随机森林的电商用户购买意愿预测复现实验')
parser.add_argument('--data', default='online_shoppers_intention.csv', help='输入CSV数据文件路径')
parser.add_argument('--output', default='results', help='结果输出目录')
args = parser.parse_args()

DATA_PATH = Path(args.data).expanduser().resolve()
ROOT = Path(args.output).expanduser().resolve()
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
ROOT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
if not DATA_PATH.exists():
    raise FileNotFoundError(f'数据文件不存在: {DATA_PATH}')

# 中文字体设置
available = {f.name for f in font_manager.fontManager.ttflist}
for candidate in ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']:
    if candidate in available:
        plt.rcParams['font.sans-serif'] = [candidate]
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 240


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def metrics_row(name: str, y_true: pd.Series, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        '模型': name,
        '阈值': threshold,
        '准确率': accuracy_score(y_true, pred),
        '精确率': precision_score(y_true, pred, zero_division=0),
        '召回率': recall_score(y_true, pred, zero_division=0),
        'F1值': f1_score(y_true, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, prob),
        'PR-AUC': average_precision_score(y_true, prob),
        'TN': int(confusion_matrix(y_true, pred)[0, 0]),
        'FP': int(confusion_matrix(y_true, pred)[0, 1]),
        'FN': int(confusion_matrix(y_true, pred)[1, 0]),
        'TP': int(confusion_matrix(y_true, pred)[1, 1]),
    }


# 1. 数据载入与基础分析
raw_df = pd.read_csv(DATA_PATH)
raw_df['Revenue'] = raw_df['Revenue'].astype(bool)
duplicate_count = int(raw_df.duplicated().sum())
# 为避免完全相同的记录同时进入训练集和测试集造成乐观偏差，建模前删除完全重复记录。
df = raw_df.drop_duplicates().reset_index(drop=True)

summary = {
    '原始样本量': int(raw_df.shape[0]),
    '建模样本量': int(df.shape[0]),
    '输入特征数': int(df.shape[1] - 1),
    '缺失值总数': int(raw_df.isna().sum().sum()),
    '删除的完全重复记录数': duplicate_count,
    '购买样本数': int(df['Revenue'].sum()),
    '未购买样本数': int((~df['Revenue']).sum()),
    '购买率': float(df['Revenue'].mean()),
}
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

class_table = df['Revenue'].map({False: '未购买', True: '购买'}).value_counts().rename_axis('类别').reset_index(name='样本数')
class_table['比例'] = class_table['样本数'] / len(df)
class_table.to_csv(TAB / 'class_distribution.csv', index=False, encoding='utf-8-sig')

# 图1 类别分布
plt.figure(figsize=(7.0, 4.6))
labels = ['未购买', '购买']
counts = [int((~df['Revenue']).sum()), int(df['Revenue'].sum())]
bars = plt.bar(labels, counts, edgecolor='black', linewidth=0.6)
plt.ylabel('会话数量')
plt.title('样本类别分布')
for bar, n in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, n + 100, f'{n}\n({n/len(df):.1%})', ha='center', va='bottom', fontsize=10)
plt.ylim(0, max(counts)*1.15)
savefig(FIG / '图1_样本类别分布.png')

# 描述统计
numeric_cols = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
                'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']
desc = df[numeric_cols].describe().T[['mean','std','min','25%','50%','75%','max']]
desc.to_csv(TAB / 'numeric_descriptive_statistics.csv', encoding='utf-8-sig')

# 图2 月份购买率
month_order = ['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
month_rate = df.groupby('Month', observed=False)['Revenue'].agg(['count','sum','mean']).reindex(month_order).dropna()
month_rate.columns = ['会话数','购买数','购买率']
month_rate.to_csv(TAB / 'month_purchase_rate.csv', encoding='utf-8-sig')
plt.figure(figsize=(8.2, 4.8))
bars = plt.bar(month_rate.index, month_rate['购买率']*100, edgecolor='black', linewidth=0.5)
plt.xlabel('月份')
plt.ylabel('购买率（%）')
plt.title('不同月份的购买转化率')
for bar, val in zip(bars, month_rate['购买率']*100):
    plt.text(bar.get_x()+bar.get_width()/2, val+0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
plt.ylim(0, max(month_rate['购买率']*100)*1.18)
savefig(FIG / '图2_月份购买转化率.png')

# 图3 访客类型购买率
visitor = df.groupby('VisitorType')['Revenue'].agg(['count','sum','mean']).sort_values('mean', ascending=False)
visitor.columns = ['会话数','购买数','购买率']
visitor.to_csv(TAB / 'visitor_purchase_rate.csv', encoding='utf-8-sig')
plt.figure(figsize=(7.2, 4.8))
cn_names = {'Returning_Visitor':'回访访客','New_Visitor':'新访客','Other':'其他'}
xlabels = [cn_names.get(i,i) for i in visitor.index]
bars = plt.bar(xlabels, visitor['购买率']*100, edgecolor='black', linewidth=0.5)
plt.ylabel('购买率（%）')
plt.title('不同访客类型的购买转化率')
for bar, val in zip(bars, visitor['购买率']*100):
    plt.text(bar.get_x()+bar.get_width()/2, val+0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
plt.ylim(0, max(visitor['购买率']*100)*1.2)
savefig(FIG / '图3_访客类型购买转化率.png')

# 图4 数值特征相关性热力图（包含目标）
corr_df = df[numeric_cols].copy()
corr_df['Revenue'] = df['Revenue'].astype(int)
corr = corr_df.corr()
plt.figure(figsize=(9.0, 7.2))
im = plt.imshow(corr, aspect='auto', vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(im, fraction=0.046, pad=0.04, label='相关系数')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=55, ha='right', fontsize=8)
plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
plt.title('数值特征相关性热力图')
savefig(FIG / '图4_数值特征相关性热力图.png')

# 2. 建模准备
X = df.drop(columns='Revenue')
y = df['Revenue'].astype(int)
categorical_cols = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
continuous_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), continuous_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
], verbose_feature_names_out=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

models = {
    '逻辑回归': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED),
    '决策树': DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=SEED),
    '支持向量机': SVC(C=2.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=SEED),
    '梯度提升树': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
    '随机森林（基线）': RandomForestClassifier(
        n_estimators=400, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=SEED
    ),
}

model_rows = []
model_probs = {}
model_pipes = {}
for name, estimator in models.items():
    start = time.time()
    pipe = Pipeline([('preprocess', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    row = metrics_row(name, y_test, prob)
    row['训练时间（秒）'] = time.time() - start
    model_rows.append(row)
    model_probs[name] = prob
    model_pipes[name] = pipe

# 3. 随机森林参数搜索
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
rf_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(random_state=SEED, n_jobs=-1))
])
param_dist = {
    'model__n_estimators': randint(250, 501),
    'model__max_depth': [None, 10, 14, 18, 24],
    'model__min_samples_split': randint(2, 16),
    'model__min_samples_leaf': randint(1, 8),
    'model__max_features': ['sqrt', 'log2', 0.4, 0.6],
    'model__class_weight': [None, 'balanced', 'balanced_subsample'],
    'model__max_samples': [None, 0.75, 0.9],
}
search = RandomizedSearchCV(
    rf_pipe, param_distributions=param_dist, n_iter=8, scoring='average_precision',
    cv=cv3, n_jobs=2, random_state=SEED, refit=True, verbose=0
)
search_start = time.time()
search.fit(X_train, y_train)
search_elapsed = time.time() - search_start
best_rf = search.best_estimator_
rf_prob = best_rf.predict_proba(X_test)[:, 1]
best_row = metrics_row('随机森林（调参后）', y_test, rf_prob)
best_row['训练时间（秒）'] = search_elapsed
model_rows.append(best_row)
model_probs['随机森林（调参后）'] = rf_prob
model_pipes['随机森林（调参后）'] = best_rf

best_params = {k.replace('model__',''):v for k,v in search.best_params_.items()}
(ROOT / 'best_params.json').write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding='utf-8')

# 4. 训练集交叉验证概率用于阈值选择，避免直接使用测试集调阈值
cv_prob = cross_val_predict(best_rf, X_train, y_train, cv=cv3, method='predict_proba', n_jobs=2)[:,1]
prec, rec, thresholds = precision_recall_curve(y_train, cv_prob)
f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
# 业务约束：正类召回率不低于0.70，在约束内选F1最高阈值；若无则选全局F1最高
valid = np.where(rec[:-1] >= 0.70)[0]
idx = valid[np.argmax(f1s[:-1][valid])] if len(valid) else int(np.argmax(f1s[:-1]))
selected_threshold = float(thresholds[idx])
threshold_row = metrics_row('随机森林（调参+阈值优化）', y_test, rf_prob, selected_threshold)
threshold_row['训练时间（秒）'] = search_elapsed
model_rows.append(threshold_row)

# 保存模型对比表
metrics_df = pd.DataFrame(model_rows)
metrics_df.to_csv(TAB / 'model_performance.csv', index=False, encoding='utf-8-sig')

threshold_compare = pd.DataFrame([
    metrics_row('默认阈值0.50', y_test, rf_prob, 0.50),
    metrics_row(f'优化阈值{selected_threshold:.3f}', y_test, rf_prob, selected_threshold),
])
threshold_compare.to_csv(TAB / 'threshold_comparison.csv', index=False, encoding='utf-8-sig')
(ROOT / 'selected_threshold.txt').write_text(f'{selected_threshold:.6f}', encoding='utf-8')

# 5. 交叉验证稳定性（调参后的随机森林，默认阈值指标用sklearn scoring）
scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc','pr_auc':'average_precision'}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_result = cross_validate(best_rf, X_train, y_train, cv=cv5, scoring=scoring, n_jobs=2)
cv_rows = []
for key in scoring:
    vals = cv_result['test_'+key]
    cv_rows.append({'指标':key, '均值':vals.mean(), '标准差':vals.std(ddof=1), '最小值':vals.min(), '最大值':vals.max()})
pd.DataFrame(cv_rows).to_csv(TAB / 'rf_cross_validation.csv', index=False, encoding='utf-8-sig')

# 图5 ROC曲线
plt.figure(figsize=(7.2, 5.6))
for name, prob in model_probs.items():
    if name == '随机森林（基线）':
        continue
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, linewidth=1.7, label=f'{name} (AUC={roc_auc_score(y_test,prob):.3f})')
plt.plot([0,1],[0,1],'--',linewidth=1.0,label='随机分类')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('不同模型的ROC曲线')
plt.legend(fontsize=8, loc='lower right')
plt.grid(alpha=0.25)
savefig(FIG / '图5_模型ROC曲线.png')

# 图6 PR曲线
plt.figure(figsize=(7.2, 5.6))
for name, prob in model_probs.items():
    if name == '随机森林（基线）':
        continue
    p, r, _ = precision_recall_curve(y_test, prob)
    plt.plot(r, p, linewidth=1.7, label=f'{name} (AP={average_precision_score(y_test,prob):.3f})')
plt.axhline(y_test.mean(), linestyle='--', linewidth=1.0, label=f'正类基准={y_test.mean():.3f}')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('不同模型的精确率-召回率曲线')
plt.legend(fontsize=8, loc='lower left')
plt.grid(alpha=0.25)
savefig(FIG / '图6_模型PR曲线.png')

# 图7 阈值与指标关系
thr_grid = np.linspace(0.05,0.90,86)
rows=[]
for th in thr_grid:
    pred=(rf_prob>=th).astype(int)
    rows.append((th,precision_score(y_test,pred,zero_division=0),recall_score(y_test,pred,zero_division=0),f1_score(y_test,pred,zero_division=0)))
thr_df=pd.DataFrame(rows,columns=['阈值','精确率','召回率','F1值'])
thr_df.to_csv(TAB / 'threshold_curve_data.csv',index=False,encoding='utf-8-sig')
plt.figure(figsize=(7.4,5.2))
plt.plot(thr_df['阈值'],thr_df['精确率'],label='精确率')
plt.plot(thr_df['阈值'],thr_df['召回率'],label='召回率')
plt.plot(thr_df['阈值'],thr_df['F1值'],label='F1值')
plt.axvline(selected_threshold,linestyle='--',linewidth=1.2,label=f'训练集选择阈值={selected_threshold:.3f}')
plt.xlabel('分类阈值')
plt.ylabel('指标值')
plt.title('随机森林分类阈值与评价指标')
plt.legend()
plt.grid(alpha=0.25)
savefig(FIG / '图7_阈值敏感性分析.png')

# 图8 混淆矩阵
pred_opt=(rf_prob>=selected_threshold).astype(int)
cm=confusion_matrix(y_test,pred_opt)
plt.figure(figsize=(5.4,4.8))
plt.imshow(cm,cmap='Blues')
plt.colorbar(fraction=0.046,pad=0.04)
plt.xticks([0,1],['预测未购买','预测购买'])
plt.yticks([0,1],['实际未购买','实际购买'])
for i in range(2):
    for j in range(2):
        plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14)
plt.title(f'优化阈值下的混淆矩阵（阈值={selected_threshold:.3f}）')
plt.ylabel('实际类别')
plt.xlabel('预测类别')
savefig(FIG / '图8_优化随机森林混淆矩阵.png')

# 6. 原始特征层面的置换重要性
perm = permutation_importance(
    best_rf, X_test, y_test, n_repeats=5, random_state=SEED,
    scoring='average_precision', n_jobs=2
)
imp = pd.DataFrame({'特征':X.columns,'重要性均值':perm.importances_mean,'重要性标准差':perm.importances_std})
imp = imp.sort_values('重要性均值',ascending=False)
imp.to_csv(TAB / 'permutation_importance.csv',index=False,encoding='utf-8-sig')

feature_cn = {
    'Administrative':'管理类页面数','Administrative_Duration':'管理类页面停留时间',
    'Informational':'信息类页面数','Informational_Duration':'信息类页面停留时间',
    'ProductRelated':'商品相关页面数','ProductRelated_Duration':'商品相关页面停留时间',
    'BounceRates':'跳出率','ExitRates':'退出率','PageValues':'页面价值','SpecialDay':'特殊日期接近度',
    'Month':'月份','OperatingSystems':'操作系统','Browser':'浏览器','Region':'地区','TrafficType':'流量来源类型',
    'VisitorType':'访客类型','Weekend':'是否周末'
}
top=imp.head(10).sort_values('重要性均值')
plt.figure(figsize=(7.6,5.8))
plt.barh([feature_cn.get(x,x) for x in top['特征']],top['重要性均值'],xerr=top['重要性标准差'],capsize=3)
plt.xlabel('PR-AUC下降量（置换重要性）')
plt.title('随机森林前10个重要特征')
plt.axvline(0,linewidth=0.8)
savefig(FIG / '图9_随机森林置换特征重要性.png')

# 图10 页面价值分组差异（对数横轴风格用log1p）
plot_df=df.copy()
plot_df['log_PageValues']=np.log1p(plot_df['PageValues'])
plt.figure(figsize=(7.0,5.0))
data0=plot_df.loc[~plot_df['Revenue'],'log_PageValues']
data1=plot_df.loc[plot_df['Revenue'],'log_PageValues']
plt.boxplot([data0,data1],labels=['未购买','购买'],showfliers=False)
plt.ylabel('log(1 + PageValues)')
plt.title('购买与未购买会话的页面价值分布')
savefig(FIG / '图10_页面价值分布差异.png')

# 导出可供论文引用的关键结果
key_results = {
    'best_cv_average_precision': float(search.best_score_),
    'best_params': best_params,
    'selected_threshold': selected_threshold,
    'test_default': metrics_row('default',y_test,rf_prob,0.5),
    'test_optimized': metrics_row('optimized',y_test,rf_prob,selected_threshold),
    'top_features': imp.head(10).to_dict(orient='records'),
    'search_elapsed_seconds': search_elapsed,
}
(ROOT/'key_results.json').write_text(json.dumps(key_results,ensure_ascii=False,indent=2),encoding='utf-8')

print(json.dumps(key_results, ensure_ascii=False, indent=2))
