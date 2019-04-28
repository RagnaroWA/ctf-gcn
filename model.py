import numpy as np
from functional import *
from utils import *
import pickle
import ctf

class GCN:
    def __init__(self, input_size, hidden_sizes, output_size, weight_decay=0, num_layers=2, nonlinearity="relu", package="scipy", args=None):
        self.layer_output = {}
        self.layer_output_nonlinear = {}
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.params = {}
        self.package = package
        self.args = args

        if nonlinearity == "relu":
            self.nonlinearity = relu
            self.nonlinearity_grad = relu_grad
        elif nonlinearity == "sigmoid":
            self.nonlinearity = sigmoid
            self.nonlinearity_grad = sigmoid_grad

        for i in range(self.num_layers):
            if i == 0:
                self.params["W" + str(i + 1)] = glorot(input_size, hidden_sizes[i])
            elif i == self.num_layers - 1:
                self.params["W" + str(i + 1)] = glorot(hidden_sizes[i - 1], output_size)
            else:
                self.params["W" + str(i + 1)] = glorot(hidden_sizes[i - 1], hidden_sizes[i])

        # change the weight matrix to ctf tensor
        if self.package == "ctf":
            for i in range(self.num_layers):
                self.params["W" + str(i + 1)] = ctf.astensor(self.params["W" + str(i + 1)])

    def forward(self, A, X, train = True):
        if train == True:
            layer_output = self.layer_output
            layer_output_nonlinear = self.layer_output_nonlinear
        else:
            temp_layer_output = {}
            temp_layer_output_nonlinear = {}
            layer_output = temp_layer_output
            layer_output_nonlinear = temp_layer_output_nonlinear

        if self.package != "ctf":
            layer_output["layer" + str(0)] = X
            for i in range(self.num_layers):
                layer_output["layer" + str(i + 1)] = A @ layer_output["layer" + str(i)] @ self.params["W" + str(i + 1)]
                if i != self.num_layers - 1:
                    layer_output_nonlinear["layer" + str(i + 1)] = self.nonlinearity(layer_output["layer" + str(i + 1)])
                else:
                    layer_output_nonlinear["layer" + str(i + 1)] = softmax(layer_output["layer" + str(i + 1)])
            return layer_output_nonlinear["layer" + str(self.num_layers)]
        else:
            # when using the package ctf
            layer_output["layer" + str(0)] = ctf.astensor(X)
            for i in range(self.num_layers):
                layer_output["layer" + str(i + 1)] = A @ layer_output["layer" + str(i)] @ self.params["W" + str(i + 1)]
                if i != self.num_layers - 1:
                    layer_output_nonlinear["layer" + str(i + 1)] = self.nonlinearity(layer_output["layer" + str(i + 1)])
                else:
                    temp_array = ctf.to_nparray(layer_output["layer" + str(i + 1)])
                    layer_output_nonlinear["layer" + str(i + 1)] = softmax(temp_array)
                    layer_output_nonlinear["layer" + str(i + 1)] = ctf.astensor(layer_output_nonlinear["layer" + str(i + 1)])
                    # print(layer_output_nonlinear["layer" + str(self.num_layers)].sp)
            return layer_output_nonlinear["layer" + str(self.num_layers)]

    def backward(self, X, y, A, mask, lr):
        gradients = self.prop_back(A, X, y, mask)
        for i in range(self.num_layers):
            self.params["W" + str(i + 1)] -= gradients["W" + str(i + 1)] * lr

    def prop_back(self, A, X, y, mask):
        gradients = {}

        preds = self.forward(A, X)
        if self.package != "ctf":
            zero_indices = np.logical_not(mask)
            preds[zero_indices] = 0
            y[zero_indices] = 0

            softmax_loss = softmax_grad(preds, y)
            gradients["W" + str(self.num_layers)] = self.layer_output_nonlinear["layer" + str(self.num_layers - 1)].T @ A.T @ softmax_loss

            last_loss = softmax_loss
            for i in range(self.num_layers - 1, 0, -1):
                out_grad = A.T @ last_loss @ self.params["W" + str(i + 1)].T
                grad = out_grad * self.nonlinearity_grad(self.layer_output["layer" + str(i)])
                gradients["W" + str(i)] = self.layer_output["layer" + str(i - 1)].T @ A.T @ grad
        else:
            zero_indices = np.logical_not(mask)
            preds = ctf.to_nparray(preds)
            y = ctf.to_nparray(y)
            preds[zero_indices] = 0
            y[zero_indices] = 0

            preds = ctf.astensor(preds)
            y = ctf.astensor(y)

            softmax_loss = softmax_grad(preds, y)
            gradients["W" + str(self.num_layers)] = self.layer_output_nonlinear["layer" + str(self.num_layers - 1)].T() @ A.T() @ softmax_loss
            last_loss = softmax_loss
            for i in range(self.num_layers - 1, 0, -1):
                out_grad = A.T() @ last_loss @ self.params["W" + str(i + 1)].T()
                grad = out_grad * self.nonlinearity_grad(self.layer_output["layer" + str(i)])
                gradients["W" + str(i)] = self.layer_output["layer" + str(i - 1)].T() @ A.T() @ grad
                print(gradients["W" + str(i)].sp)

        for i in range(self.num_layers):
            gradients["W" + str(i + 1)] += self.weight_decay * self.params["W" + str(i + 1)]
        return gradients

    def predict(self, A, X):
        scores = self.forward(A, X, train = False)
        if self.package != "ctf":
            y_pred = np.argmax(scores, axis=1)
        else:
            scores = ctf.to_nparray(scores)
            y_pred = np.argmax(scores, axis=1)
        return y_pred

    def accuracy(self, A, X, y, mask):
        if self.package == "ctf":
            y = ctf.to_nparray(y)
        num = np.sum(mask)
        scores = self.predict(A, X)
        scores = scores[mask]
        labels = np.argmax(y[mask], axis=1)
        correct = np.sum(scores == labels)
        return correct / num

    def loss(self, A, X, y, mask):
        num = np.sum(mask)
        pred = self.forward(A, X, train = False)
        if self.package == "ctf":
            y = ctf.to_nparray(y)
            pred = ctf.to_nparray(pred)
        loss = np.sum(y[mask] * np.log(pred[mask])) * -1 / num
        
        for i in range(self.num_layers):
            if self.package != "ctf":
                l2_loss = np.sum(np.linalg.norm(self.params["W" + str(i + 1)], ord='fro') * 0.5) * self.weight_decay
            else:
                l2_loss = self.params["W" + str(i + 1)].norm2() * 0.5 * self.weight_decay
            loss += l2_loss
        return loss

    def save(self, file_path):
        save_params = self.params.copy()
        if self.package == "ctf":
            for i in range(self.num_layers):
                save_params["W" + str(i + 1)] = ctf.to_nparray(self.params["W" + str(i + 1)])
        save_data(file_path, {"params": save_params, "args": self.args})

    def load(self, file_path):
        data = load_data(file_path)
        params = data['params']
        save_args = data['args']
        self.args = save_args
        load_params = params.copy()
        if self.args != None:
            self.package = self.args['package']
        if self.package == "ctf":
            for i in range(self.num_layers):
                load_params["W" + str(i + 1)] = ctf.astensor(params["W" + str(i + 1)])
        self.params = load_params