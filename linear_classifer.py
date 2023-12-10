import numpy as np


def softmax(predictions):
    pred = predictions.copy()
    pred = pred - np.max(pred)
    S_Max = np.zeros(pred.shape)
    it = np.nditer(pred, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        S_Max[ix] = np.exp(pred[ix])/np.sum(np.exp(pred))
        it.iternext()
    return S_Max
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):

    return -np.log(probs[target_index])
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    Vector = np.zeros(predictions.shape)
    for n in range(predictions.shape[0]):
        Vector[n, target_index[n]] = 1

    s_m = np.zeros(predictions.shape)
    for n in range(predictions.shape[0]):
        s_m[n] = softmax(predictions[n])
    loss = np.zeros(target_index.shape[0])
    for n in range(target_index.shape[0]):
        loss[n] = cross_entropy_loss(s_m[n], target_index[n])
    grad_prediction = np.zeros(predictions.shape)
    for n in range(grad_prediction.shape[0]):
        grad_prediction[n] = s_m[n] - Vector[n]

    raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    f = lambda x: reg_strength * np.sum(np.square(x))
    loss = f(W)

    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])

    grad = W.copy()
    while not it.finished:
        ix = it.multi_index

        x0 = W.copy()
        x0[ix] += reg_strength
        x1 = W.copy()
        x1[ix] -= reg_strength

        numeric_grad_at_ix = (f(x0) - f(x1)) / (2 * reg_strength)
        grad[ix] = numeric_grad_at_ix
        it.iternext()

    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):

    predictions = np.dot(X, W)
    loss, dW = softmax_with_cross_entropy(predictions, target_index)

    dW = X.transpose().dot(dW)

    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            print()

            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
