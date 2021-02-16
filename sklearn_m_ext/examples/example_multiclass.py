import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn_m_ext import custom_metrics

np.random.seed(42)

y = np.random.choice(list("ABC"), size=100, replace=True)
#y = np.random.randint(2, size=100)
X = np.random.normal(0, 1, (len(y), 5))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = LogisticRegression()
regr.fit(X_train, y_train)

pred = regr.predict(X_test)
print(classification_report(y_test, pred))

print(custom_metrics.classification_report2(y_test, pred))