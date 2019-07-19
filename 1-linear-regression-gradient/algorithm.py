# y = ax + b

class LinearRegressionGradient(object):
    def __init__(self, step, epochs, values):
        self.step = step
        self.epochs = epochs
        self.values = list(map(list, values))

        self.a = 0
        self.b = 0

    def gradient_descent(self):
        for _ in range(self.epochs):
            alpha = self.step / len(self.values)

            temp_b = self.b - alpha * sum(
                [((self.a * x + self.b) - y) for x, y in self.values]
            )
            temp_a = self.a - alpha * sum(
                [((self.a * x + self.b) - y) * x for x, y in self.values]
            )

            self.a = temp_a
            self.b = temp_b
