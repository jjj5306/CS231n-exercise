def GradientDescent(learning_rate):
    while True:
        data_batch = dataset.sample_data_batch()
        loss = network.forward(data_batch)
        dx = network.backward()
        x += - learning_rate * dx

def Momentum(learning_rate, mu):
    v = mu * v - learning_rate * dx
    x += v

def AdaGrad(learning_rate):
    cache += dx**2
    x += -learning_rate * dx / (np.sqrt(cache) + 1e-7)

def RMSProp(learning_rate):
    cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
    x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)

def Adam(learning_rate):
    m, v =  # ... initialize caches to zeros
    for t in xrange(1, big_number):
        dx =  # ... evaluate gradient
        m = beta1 * m + (1 - beta1) * dx  # update first moment
        v = beta2 * v + (1 - beta2) * (dx ** 2)  # update second moment
        mb = m / (1 - beta1 ** t)  # correct bias
        vb = v / (1 - beta2 ** t)  # correct bias
        x += - learning_rate * mb / (np.sqrt(vb) + 1e-7)