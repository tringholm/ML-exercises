import json
import math
import tensorflow as tf
import numpy as np
from optimization.gradient_descent import gradient_descent
from utils.plotting import plotSingle


class PhaseModel(tf.Module):
    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(42)
        self.N = N
        self.a = tf.Variable(np.zeros(N), name ="a", shape=(N), dtype=tf.float32)
        self.b = tf.Variable(np.zeros(N), name ="b", shape=(N), dtype=tf.float32)
        self.c = tf.Variable(np.zeros(1), name ="c", shape=(1), dtype=tf.float32)
        self.n = tf.range(1,N+1, dtype=tf.float32)
    
    def __call__(self, x):
        return self.c + tf.reduce_sum(self.a*tf.sin(math.pi*x*self.n + self.b))


def L(model, x, y):
    val = tf.zeros([1])
    for x_i, y_i in zip(x,y):
        val = val + tf.square(y_i - model(x_i))
    return val


if __name__ == "__main__":
    with open('data/exercise1.json') as json_file:
        data = json.load(json_file)
        x = np.array(data['x'])
        y = np.array(data['y'])

    N = 3
    model = PhaseModel(N)
    plotSingle(model, x, y)

    L_val = L(model,x, y).numpy()
    print(f"Starting L: {L_val}")

    gradient_descent(model, L, x, y, 1000, 0.0005, plotSingle)
