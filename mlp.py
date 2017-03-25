import numpy as np


def softmax(z2):
    exps = np.exp(z2)
    return exps / np.sum(exps, axis=1, keepdims=True)


class MLP:
    """
    (1 hidden layer) 3 layer perceptron
    """

    def __init__(self, X, y, hu_size, rand_seed=0):
        self.X, self.y = X, y
        self.num_examples = X.shape[0]
        self.input_dimension = X.shape[1]
        self.output_dimension = y.shape[1]
        self.hiddenunit_size = hu_size
        self.epsilon = 0.01
        np.random.seed(rand_seed)
        self.W1 = np.random.randn(
            self.input_dimension,
            self.hiddenunit_size) / np.sqrt(self.input_dimension)
        self.b1 = np.zeros((1, self.hiddenunit_size))
        self.W2 = np.random.rand(
            self.hiddenunit_size,
            self.output_dimension) / np.sqrt(self.hiddenunit_size)
        self.b2 = np.zeros((1, self.output_dimension))

    def feed_forward(self, x):
        z1 = np.tanh(x.dot(self.W1) + self.b1)
        z2 = softmax(z1.dot(self.W2) + self.b2)
        return {'z1': z1, 'z2': z2}

    def back_propagate(self, ff, x, y):
        z1, y_hat = ff['z1'], ff['z2']

        d3 = y_hat - y
        dW2 = np.dot(z1.T, d3)
        db2 = np.sum(d3, axis=0)
        # tanh'(x) = 1 - tanh^2(x)
        d2 = (1 - np.square(z1)) * np.dot(d3, self.W2.T)
        dW1 = np.dot(x.T, d2)
        db1 = np.sum(d2, axis=0)

        self.W2 += -self.epsilon * dW2
        self.b2 += -self.epsilon * db2
        self.W1 += -self.epsilon * dW1
        self.b1 += -self.epsilon * db1

    def get_model(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def predict(self, x):
        ff = self.feed_forward(x)
        return np.argmax(ff['z2'], axis=1)

    def data_loss(self):
        ff = self.feed_forward(self.X)
        y_hat = ff['z2']
        products = np.multiply(self.y, np.log(y_hat))
        data_loss = np.sum(products)
        return (-1. / self.num_examples) * data_loss

    def train(self,
              epoch=250000,
              print_loss=False,
              testset_X=None,
              testset_y=None,
              checkpoint=False):
        for i in range(epoch):

            ff = self.feed_forward(self.X)

            self.back_propagate(ff, self.X, self.y)

            if print_loss and i % 2000 == 0:
                print("Data loss (cross entropy) after epoch {0}: {1}".format(
                    i, self.data_loss()))

            print_accuracy = testset_X is not None and testset_y is not None
            if print_accuracy and i % 2000 == 0:
                x_len = testset_X.shape[0]
                predict_idx = self.predict(testset_X)
                correct_prediction = [
                    target[predict_idx[row_id]]
                    for row_id, target in enumerate(testset_y)
                ]
                print("Accuracy after epoch {0}: {1}%".format(
                    i, sum(correct_prediction) / x_len))

            if checkpoint and i % 2000 == 0:
                pass  # TODO implement dump parameter using np.savez with shape

    def minibatch_train(self,
                        batch_size,
                        epoch=250000,
                        print_loss=False,
                        testset_X=None,
                        testset_y=None,
                        checkpoint=False):
        for i in range(epoch):
            subset_idx = np.random.choice(
                self.num_examples, size=batch_size, replace=False)
            sample_X = self.X[subset_idx]
            sample_y = self.y[subset_idx]

            ff = self.feed_forward(sample_X)

            self.back_propagate(ff, sample_X, sample_y)

            if print_loss and i % 2000 == 0:
                print("Data loss (cross entropy) after epoch {0}: {1}".format(
                    i, self.data_loss()))

            print_accuracy = testset_X is not None and testset_y is not None
            if print_accuracy and i % 2000 == 0:
                x_len = testset_X.shape[0]
                predict_idx = self.predict(testset_X)
                correct_prediction = [
                    target[predict_idx[row_id]]
                    for row_id, target in enumerate(testset_y)
                ]
                print("Accuracy after epoch {0}: {1}%".format(
                    i, sum(correct_prediction) / x_len))

            if checkpoint and i % 2000 == 0:
                pass  # TODO implement dump parameter using np.savez with shape


def get_Xor_data():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array(
        [[1, 0], [0, 1], [0, 1], [1, 0]])


def to_indexmatrix(vector):
    res = []
    for v in vector:
        row = np.zeros(26, dtype=np.int)
        row[v] = 1
        res.append(row)
    return np.array(res)


def get_Hwl_data():
    import pandas as pd
    dataset = pd.read_csv('letter-recognition.data', header=None)
    examples_dataframe = dataset.ix[:, 1:16]
    target_letter = [ord(item) - ord('A') for item in dataset.ix[:, 0]]
    target = to_indexmatrix(target_letter)
    return examples_dataframe.as_matrix(), target


def get_training_idx(data_size, percentage):
    return np.random.choice(
        data_size, size=round(data_size * percentage), replace=False)


def get_testing_idx(data_size, training_idx):
    inverse = np.ones(data_size, dtype=np.bool)
    inverse[training_idx] = 0
    return inverse


def get_data_split(X, y, percentage):
    data_size = X.shape[0]
    training_idx = get_training_idx(data_size, percentage)
    testing_idx = get_testing_idx(data_size, training_idx)
    return X[training_idx], y[training_idx], X[testing_idx], y[testing_idx]
