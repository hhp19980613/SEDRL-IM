#coding=utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# 加载莺尾花数据集
from sklearn import datasets
# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 数据导入&分析
X, y = datasets.load_iris(return_X_y=True)
print("shape:", X.shape)
print("shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

# 模型训练
# 使用高斯朴素贝叶斯进行计算
clf = GaussianNB()
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
acc = np.sum(y_test == y_pred) / X_test.shape[0]
print("Test Acc : %.3f" % acc)

# 预测
print("shape:", X_test.shape)
y_proba = clf.predict_proba(X_test[:1])
print(clf.predict(X_test[:1]))
print("预计的概率值:", y_proba)

