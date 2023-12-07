import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer





# IMPORTANT: getting the data and exploring it

cancer = load_breast_cancer()
#print(cancer)

# check what kinf of keys we have
#print(cancer.keys())

# print(cancer['DESCR'])
# print(cancer['target'])
# print(cancer['target_names'])
# print(cancer['feature_names'])
# print(cancer['data'].shape)


# Creating Dataframe
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# print(df.dtypes)
# print(df['target'].value_counts())
# print(357 / len(df), 211/ len(df))

sns.countplot(data=df, x='target')
plt.savefig('countplot.png')  # Save as PNG image
# plt.show()

print(df.columns)
# sns.pairplot(data=df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'], hue='target')
# plt.show()

plt.figure(figsize=(20,15))
# sns.heatmap(df.corr(), annot=True, cmap='viridis_r')
# plt.show()

print(df.corr()['target'].sort_values(ascending=False))
df.corr()['target'][:-1].sort_values().plot(kind='bar')
# plt.show()





# IMPORTANT: Modeling

# Train, Test and Split
# Separate Features and Labels
X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Model Training
print("\n\nModel Training")
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)


# Model Evaluation
y_preds = svc_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, pair_confusion_matrix
print(classification_report(y_test, y_preds))