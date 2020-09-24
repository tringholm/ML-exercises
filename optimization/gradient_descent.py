import tensorflow as tf

def gradient_descent(model, L, x, y, n_iter, alpha = 0.01, plotfn = None):

  for n in range(n_iter):
    _gd_step(model, L, x, y, alpha)

    L_val = L(model, x, y).numpy()

    print(f"Iteration: {n + 1} L: {L_val}")
    if plotfn is not None:
        plotfn(model, x, y)

def _gd_step(model, L, x, y, alpha):

  with tf.GradientTape() as t:
    L_val = L(model, x, y)

  grad_a, grad_b, grad_c = t.gradient(L_val, [model.a, model.b, model.c])

  model.a.assign_sub(alpha * grad_a)
  model.b.assign_sub(alpha * grad_b)
  model.c.assign_sub(alpha * grad_c)