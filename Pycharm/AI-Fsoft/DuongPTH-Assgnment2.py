import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

link = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data'

header = ['CRM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
header_for_predict = ['CRM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
data = pd.read_csv(link, header=None, names=header, sep='\s+')

viz = data[header]
viz.hist()
# plt.figure(figsize=(20,30))
plt.show()

foo = np.random.rand(len(data)) < 0.7
train = data[foo]
test = data[~foo]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[header_for_predict])
y = np.asanyarray(train[['MEDV']])
regr.fit(x, y)
# The coefficients
print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[header_for_predict])
x = np.asanyarray(test[header_for_predict])
y = np.asanyarray(test[['MEDV']])

print("Mean absolute error     (MAE): %.2f" % np.mean(np.absolute(y_hat - y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y) ** 2))
print("Root Mean square error (RMSE): %.2f" % np.sqrt(np.mean((y_hat - y) ** 2)))
print("R2-score: %.2f" % r2_score(y_hat, y))

# -------------------------------------------------------------------------------------------------------
# page = requests.get(link)
# data = page.text
#
# rows = data.split("\n")
# print(rows)
# crm = []
# zn = []
# indus = []
# chas = []
# nox = []
# rm = []
# age = []
# dis = []
# rad = []
# tax = []
# ptradio = []
# b = []
# lstat = []
# medv = []
#
# for row_str in rows:
#     row_str = row_str.replace("   ", " ")
#     row_str = row_str.replace("  ", " ")
#     col = row_str.split(" ")
#     crm.append(float(col[1]))
#     zn.append(float(col[2]))
#     indus.append(float(col[3]))
#     chas.append(float(col[4]))
#     nox.append(float(col[5]))
#     rm.append(float(col[6]))
#     age.append(float(col[7]))
#     dis.append(float(col[8]))
#     rad.append(float(col[9]))
#     tax.append(float(col[10]))
#     ptradio.append(float(col[11]))
#     b.append(float(col[12]))
#     lstat.append(float(col[13]))
#     if (len(col) >= 13):
#         medv.append(float(col[len(14)]))
#     else:
#         medv.append(0.0)
#
# house = {'CRM': crm, \
#          'ZN': zn, '\
#          INDUS': indus, \
#          'CHAS': chas, \
#          'NOX': nox, \
#          'RM': rm, \
#          'AGE': age, \
#          'DIS': dis, \
#          'RAD': rad, \
#          'TAX': tax, \
#          'PTRATIO': ptradio, \
#          'B': b, \
#          'LSTAT': lstat, \
#          # 'MEDV': medv \
#          }
# J = pd.DataFrame(house)
# print(J)
