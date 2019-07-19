import random

import matplotlib.pyplot as plt
import numpy as np

from algorithm import LinearRegressionGradient


X_DATA = [x for x in range(1, 200, 5)]
Y_DATA = [y + random.randint(-50, 50) for y in X_DATA]

alg = LinearRegressionGradient(
    step=0.0001,
    epochs=1000,
    values=zip(X_DATA, Y_DATA)
)

alg.gradient_descent()

x = np.linspace(min(X_DATA) - 5, max(X_DATA) + 10, 100)
y = alg.a * x + alg.b
plt.plot(X_DATA, Y_DATA, 'go')
plt.title('y={}x {} {}'.format(
    alg.a,
    '+' if alg.b > 0 else '',
    alg.b
))
plt.plot(x, y, '-r')
plt.axis([min(X_DATA) - 5, max(X_DATA) + 10, min(Y_DATA) - 5, max(Y_DATA) + 10])
plt.show()
