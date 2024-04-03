import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


df = pd.read_csv('Salary_Data.csv')

print(df.head())

plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.show()


data = df.values

print(data)

def sum_squared_error(slope, intercept, x_real, y_real):
    y_pred = intercept + (slope * x_real)
    error = (y_real - y_pred) ** 2
    return np.sum(error)


def r_squared(y_real, sum_squared_error):
    y_mean = np.mean(y_real)
    variate = np.sum((y_real - y_mean) ** 2)
    length = len(y_real)
    variance = variate / length
    
    r_squaredd = (variance - sum_squared_error) / variance
    return r_squaredd


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)


    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]

        m_gradient += -(2/n) * x * (y - ((m_now * x) + b_now))
        b_gradient += -(2/n) * (y - ((m_now * x) + b_now))

    m_new = m_now - (L * m_gradient)
    b_new = b_now - (L * b_gradient)
    return m_new, b_new


m = 0 
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.plot(df['YearsExperience'], m * df['YearsExperience'] + b, color='red')
plt.show()



# WITH SKLEARN
    
x_train, x_test, y_train, y_test = train_test_split(df['YearsExperience'], df['Salary'], test_size=0.2)
LR = linear_model.LinearRegression()

LR.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

print(LR.coef_)

print(LR.intercept_)

prediction = LR.predict(x_test.values.reshape(-1, 1))

print(prediction)

plt.scatter(x_test, y_test)

plt.plot(x_test, prediction, color='red')

plt.show()

print(LR.score(x_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1)))