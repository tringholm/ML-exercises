import matplotlib.pyplot as plt


def plotMulti(model, x, y):
    plt.clf()
    plt.scatter(x, y, c="b")
    f = [model(x_i) for x_i in x]
    M = f[0].shape[1]
    for m in range(M):
        f_m = [f_val[0][m].numpy() for f_val in f]
        plt.scatter(x, f_m, c = "r")
    plt.ion()
    plt.ylim(-4,2)
    plt.show()
    plt.pause(0.001)


def plotSingle(model, x, y):
    plt.clf()
    plt.scatter(x, y, c="b")
    y_pred = [model(x_i).numpy() for x_i in x]
    plt.scatter(x, y_pred, c="r")
    plt.ion()
    plt.ylim(-4,2)
    plt.show()
    plt.pause(0.001)