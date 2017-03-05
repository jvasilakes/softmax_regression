from __future__ import division, print_function

import sys
import numpy as np
import scipy.stats as stats

from collections import Counter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("trainfile", help="Training data file (CSV)")
parser.add_argument("testfile", help="Test data file (CSV)")
parser.add_argument("-l", "--l2", type=float, default=0.1,
                    help="L2 regularization parameter (default; 0.1)")
parser.add_argument("-e", "--eta", type=float, default=1e-3,
                    help="Step size for gradient descent (default: 1e-3)")
parser.add_argument("-i", "--numiterations", type=int, default=100,
                help="Number of training iterations (default: 100)")


class GradientDescentTrainer(object):
    def __init__(self, train_x, train_y, l2=1, eta=0.1):
        self.x = train_x
        self.y = train_y
        self.w = stats.truncnorm.rvs(0, 0.2, size=(self.y.shape[1], self.x.shape[1]))
#        self.w = np.zeros([self.y.shape[1], self.x.shape[1]])  # classes x features
        self.l2 = l2   # L2 regularization parameter
        self.eta = eta  # Step size
        self._x_by_class = None  # Features set to non-zero for each given class

    def print_params(self):
        print("Gradient descent optimization of cross-entropy cost function")
        print("  with L2 regularization.")
        print("L2: {0}".format(self.l2))
        print("eta: {0}".format(self.eta))

    def evaluate(self, test_x, refs):
        probs = self.softmax(xs=test_x)
        correct = np.argmax(probs, axis=0) == np.argmax(refs, axis=1)
        accuracy = 100 * np.sum(correct) / len(refs)
        print("Test Accuracy: {0:.2f} ({1}/{2})"
               .format(accuracy, np.sum(correct), len(refs)))

    def train(self, num_iter=5):
        print("Training for {0} iterations".format(num_iter))
        print_every = int(round(num_iter**(0.75), -1))
        for i in xrange(num_iter):
            probs = self.softmax()
            cost = self.compute_cost(probs)
            if i % print_every == 0:
                print("Cost at epoch {0}: {1}".format(i, cost))
            gradient = self.compute_gradient(probs)
            self.update_weights(gradient)
        probs = self.softmax()
        cost = self.compute_cost(probs)
        print("Final cost: {0}".format(cost))

    def softmax(self, xs=None):
        if xs is not None:
            x = xs 
        else:
            x = self.x
        scores = np.dot(x, self.w.T) # Numerator
        probs = np.exp(scores).T / np.sum(np.exp(scores), axis=1)
        return probs

    def compute_cost(self, probs):
        """
        Cross-Entropy cost function.
        """
        norm = -1/self.x.shape[0]
        reg = (self.l2 / 2) * np.sum(self.w * self.w)
        cost = norm * np.sum(self.y * np.log(probs.T)) + reg
        return cost

    def compute_gradient(self, probs): 
        gradient = np.zeros([self.y.shape[1], self.x.shape[1]])
        if not self._x_by_class:
            self._x_by_class = []
            for y in xrange(self.y.shape[1]):
                indicator = np.argmax(self.y, axis=1) == y
                self._x_by_class.append(indicator * self.x.T)
        for y in xrange(self.y.shape[1]):
            norm = -1/self.x.shape[0]
            reg = self.l2 * self.w[y]
            grad_y = norm * np.sum(self._x_by_class[y] - (probs[y] * self.x.T), axis=1) + reg
            gradient[y] = grad_y
        return gradient

    def update_weights(self, gradient):
        self.w = self.w - self.eta * gradient

# =================
# FEATURE FUNCTIONS
# =================

def features_from_file(infile):
    features = []
    targets = []
    with open(infile, 'r') as inF:
        for line in inF:
            line = line.split(',')
            targets.append(line[0])
            features.append([int(f) for f in line[1:]])
    return np.array(features), np.array(targets)


def to_onehot(targets, classmap):
    y_t = np.zeros([len(targets), len(classmap)])
    for i,_ in enumerate(y_t):
        index = classmap[targets[i]]
        y_t[i][index] = 1
    return y_t

def from_onehot(predictions, classmap):
    inv_classmap = dict([(i,v) for (v,i) in classmap.items()])
    words = [inv_classmap[i] for i in predictions]
    return words

# =================
# EVALUATION FUNCTIONS
# =================

def compute_precision_recall(confusions):
    tps = [confusions[i][i] for i in range(len(confusions))]
    total_hyps = np.sum(confusions, axis=1)
    total_refs = np.sum(confusions, axis=0)
    precisions = tps / total_hyps
    recalls = tps / total_refs
    return precisions, recalls

def print_confusions(classmap, hyps, refs):
    num_classes = len(classmap)
    words = [k for k,v in sorted(classmap.items(), key=lambda x: x[1])]
    confusions = np.zeros([num_classes, num_classes], dtype='int') # hyps x refs
    for hyp, ref in zip(hyps, refs):
        ref_idx = np.argmax(ref)
        confusions[hyp][ref_idx] += 1
    template = "{: <6}" + "{: <7}" * (num_classes+1)
    print(template.format("", *words + ["TOTAL"]))  # header
    for word, conf in zip(words, confusions):
        total = np.sum(conf)
        conf = np.append(conf, [total])
        print(template.format(word, *conf))
    total_hyps = np.sum(confusions, axis=0)
    total_hyps = np.append(total_hyps, "")
    print(template.format("TOTAL", *total_hyps))
    precisions, recalls = compute_precision_recall(confusions) 
    slots = "{:.2f} " * num_classes
    print("Precision/class: " + slots.format(*precisions))
    print("Recall/class: " + slots.format(*recalls))

def compute_baseline(train_targets, test_refs):
    # Find the most common class in the training data
    counts = Counter(train_targets)
    most_common = counts.most_common(1)[0][0]
    # Always predict most common class in the training data.
    predictions = [(most_common, ref) for ref in test_refs]
    num_correct = len([pred for (pred, ref) in predictions if pred == ref])
    accuracy = 100 * num_correct / len(predictions)
    print("Baseline Accuracy ('{0}'): {1:.2f} ({2}/{3})"
           .format(most_common, accuracy, num_correct, len(test_refs)))


def main(trainfile, testfile, l2, eta, num_iter):
    print("Reading features")
    x, targets = features_from_file(trainfile)
    num_classes = len(sorted(np.unique(targets)))
    classmap = dict(zip(set(targets), range(num_classes)))
    y_t = to_onehot(targets, classmap)
    test_x, refs = features_from_file(testfile)
    test_y = to_onehot(refs, classmap)

    GD = GradientDescentTrainer(x, y_t, l2=l2, eta=eta)
    GD.print_params()
    compute_baseline(targets, refs)
    GD.train(num_iter=num_iter)

    # Results
    predictions = np.argmax(GD.softmax(xs=test_x), axis=0)
    GD.evaluate(test_x, test_y)
    print_confusions(classmap, predictions, test_y)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.trainfile, args.testfile, args.l2, args.eta, args.numiterations)
