import json
import math
import tensorflow as tf
import numpy as np
from optimization.gradient_descent import gradient_descent
from utils.plotting import plotMulti


class PhaseModel(tf.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(3) # 2 is reasonable, 3 is best
        self.N = N
        a = np.random.rand(N,M)
        a[:,0] = - a[:,0]
        self.a = tf.Variable(a, name ="a", shape=(N,M), dtype=tf.float32)
        self.b = tf.Variable(np.random.rand(N,M), name ="b", shape=(N,M), dtype=tf.float32)
        self.c = tf.Variable(np.transpose(np.array([[-1.0],[-1.0]])), name ="c", shape=(1,M), dtype=tf.float32)
        self.n = tf.stack([tf.range(1,N+1, dtype=tf.float32) for i in range(M)], 1)
    
    def __call__(self, x):
        return self.c + tf.reduce_sum(self.a*tf.sin(math.pi*x*self.n + self.b), axis = 0)


def L(model, x, y):
    val = tf.zeros([1])
    for x_i, y_i in zip(x,y):
        val = val + tf.reduce_min(tf.square(y_i - model(x_i)))
    return val


if __name__ == "__main__":
    with open('data/exercise4.json') as json_file:
        data = json.load(json_file)
        x = np.array(data['x'])
        y = np.array(data['y'])

    N = 3
    M = 2
    
    model = PhaseModel(N, M)
    plotMulti(model, x, y)

    L_val = L(model,x, y).numpy()
    print(f"Starting L: {L_val}")

    gradient_descent(model, L, x, y, 1000, 0.0005, plotMulti)

