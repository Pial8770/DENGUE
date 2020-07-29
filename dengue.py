import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

data = pd.read_csv("resources/Dengue.csv")
data = np.array(data)
X = data[0:, 0:-1]
X = X.astype('int')
y = data[0:, -1]
y = y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))