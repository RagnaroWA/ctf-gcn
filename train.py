import time
import argparse
import numpy as np
import pickle
import ctf

# Set random seed
seed = 123
np.random.seed(seed)

from model import GCN
from utils import load_data, normalize_adj, normalize_features

def train(model, features, adj, y_train, y_val, train_mask, val_mask, lr=0.1, epochs=200, patience=0, save_best=False):
    begin = time.time()
    val_loss_history = []
    min_epoch = -1
    computation_time = 0
    for epoch in range(epochs):

        # the backpropogation time
        start = time.time()
        model.backward(features, y_train, adj, train_mask, lr)
        end = time.time()

        train_loss = model.loss(adj, features, y_train, train_mask)
        val_loss= model.loss(adj, features, y_val, val_mask)

        train_accuracy = model.accuracy(adj, features, y_train, train_mask)
        val_accuracy = model.accuracy(adj, features, y_val, val_mask)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_accuracy), "time=", "{:.5f}".format(end - start))

        computation_time += (end - start)

        # if the validation loss is not decreasing in patience times, 
        if patience != 0:
            if len(val_loss_history) == 0:
                val_loss_history.append(val_loss)
                continue
            else:
                min_epoch = np.argmin(np.array(val_loss_history))
                min_loss = val_loss_history[min_epoch]
                val_loss_history.append(val_loss)
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_epoch = epoch
                    if save_best == True:
                        model.save('params/best.pkl')
                else:
                    if epoch - min_epoch >= patience:
                        print("Validation loss has not been improved for", '%04d' % patience, "epochs, reaching the specified patience.")
                        break
    # total time
    stop = time.time()
    print("Total time: {:.4f}s, ".format(stop - begin), "Computation time: {:.4f}s".format(computation_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='The weight decay')
    parser.add_argument('--patience', type=int, default=10, help='The patience for validation loss')
    parser.add_argument('--hidden_size', type=int, default=16, help='The hidden size for each hidden layer')
    parser.add_argument('--num_layers', type=int, default=2, help='The number of total layers, number of hidden + 1')
    parser.add_argument('--package', type=str, default="scipy", help='Use numpy, scipy or ctf for matrix multiplication')
    parser.add_argument('--dataset', type=str, default="cora", help='Cora or Citeer')
    parser.add_argument('--save_best', type=bool, default=False, help='Save the best model according to validation loss')
    args = parser.parse_args()

    print("Params: lr={:.4f}, epochs={}, weight_decay={:.5f}, patience={}, hidden_size={}, num_layers={}, package={}, dataset={}"\
    .format(args.lr, args.epochs, args.weight_decay, args.patience, args.hidden_size, args.num_layers, args.package, args.dataset))

    if args.dataset == "cora":
        data = load_data("data/cora/cora.pkl")

    adj = data['adj']
    features = data['features']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    train_mask = data['train_index']
    val_mask = data['val_index']
    test_mask = data['test_index']
    adj = normalize_adj(adj)
    features = normalize_features(features)

    if args.package == "numpy":
        features = features.toarray()
        adj = adj.toarray()
    elif args.package == "ctf":
        y_train = ctf.astensor(y_train)
        y_val = ctf.astensor(y_val)
        y_test = ctf.astensor(y_test)

        features = features.toarray()
        adj = adj.toarray()
        adj = ctf.astensor(adj)
        features = ctf.astensor(features)

    input_size = features.shape[1]
    output_size = y_train.shape[1]

    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    patience = args.patience
    num_layers = args.num_layers
    hidden_sizes = [args.hidden_size] * (num_layers - 1)

    save_args = args.__dict__

    model = GCN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, weight_decay=weight_decay, num_layers=num_layers, package=args.package, args=save_args)
    train(model, features, adj, y_train, y_val, train_mask, val_mask, lr=lr, epochs=epochs, patience=patience, save_best=args.save_best)

    # model.load('params/best.pkl')
    # test_loss = model.loss(adj, features, y_test, test_mask)
    test_accuracy = model.accuracy(adj, features, y_test, test_mask)
    print("Test Accuracy:","{:.5f}".format(test_accuracy))
