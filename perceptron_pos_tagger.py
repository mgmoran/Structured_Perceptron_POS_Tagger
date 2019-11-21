### new version ###
from collections import defaultdict
import os
import sys
import subprocess
import numpy as np
import random
from data_structures import vocabulary, labels
from scorer import compute_acc

class Perceptron_POS_Tagger(object):
    def __init__(self,):
        ''' Modify if necessary.
        '''
        self.unk = '*UNKNOWN*'
        self.start = '*START*'
        self.stop = '*STOP*'
        self.featuredict = dict()
        self.featureset = set()
        self.labelset = labels
        self.labeldict = dict()
        self.theta = None
        self.transitions = None
        self.a = None

    def viterbi(self,test_instance):
        sent = test_instance.snt
        v = np.zeros((len(self.labelset), len(sent)), dtype=np.int64)
        backpointer = np.zeros((len(self.labelset), len(sent)),dtype=float)
        inverted_labeldict = {k: v for (v, k) in self.labeldict.items()}
        transitions = np.zeros((45,45))
        for i in range(len(self.transitions)):
            transitions[i] = self.theta[self.transitions[i]]
        step = 0
        features = test_instance.featurelist[step]
        feature_matrix = np.ones((len(self.labelset), len(features)))
        for i in range(len(features)):
            feature = features[i]
            feature_matrix[:, i] *= self.theta[feature]
        v[:, step] = np.sum(feature_matrix, axis=1)
        for i in range(1, len(sent)):
            step = i
            features = test_instance.featurelist[step]
            feature_matrix = np.ones((len(self.labelset),len(features)))
            for i in range(len(features)):
                feature = features[i]
                feature_matrix[:, i] *= self.theta[feature]
            v[:,step] = np.amax(v[:, step - 1,None] + transitions + np.sum(feature_matrix,axis=1),axis=0)
            backpointer[:,step] = np.argmax(v[:, step - 1,None] + transitions + np.sum(feature_matrix,axis=1), axis=0)
        ### termination step ###
        bestpathpointer = np.argmax(v[:, step])
        best_path = [bestpathpointer]
        while step > 0:
            prev = backpointer[bestpathpointer][step]
            best_path.append(prev)
            bestpathpointer = prev
            step = step - 1
        if isinstance(sent[0],tuple):
            sent = [token[0] for token in sent]
        bestpath = list(zip(sent, [inverted_labeldict[i] for i in best_path[::-1]]))
        return bestpath

    def tag(self, test_data):
        print("tagging dev data...")
        tagged_data = []
        for sent in test_data:
            tagged_data.append(self.viterbi(sent))
        return tagged_data

    def train(self, train_data, dev_gold, dev_plain):
        ''' Implement the Perceptron training algorithm here.
        '''
        print("training...")
        for word in vocabulary:
            self.featureset.update(["word0=" + word, "word1=" + word, "word_1=" + word, "word2=" + word, "word_2=" + word])
            if len(word) > 3:
                self.featureset.update(["prefix=" + word[:3]])
            if len(word) > 4:
                self.featureset.update(["suffix=" + word[-3:]])
        training_data = train_data
        ## making room for previous tag feature ##
        self.featureset.update(["pos_1=" + x for x in self.labelset])
        self.featureset = list(self.featureset)
        self.labelset = list(self.labelset)
        self.transitions = ["pos_1=" + x for x in self.labelset]
        featureindices = zip(self.featureset,range(len(self.featureset)))
        labelindices = zip(self.labelset,range(len(self.labelset)))
        self.featuredict.update({k[0]:k[1] for k in featureindices})
        self.labeldict.update({k[0]:k[1] for k in labelindices})
        ### every feature assigned to a weight, initialized at 0.0 ###
        self.theta = dict()
        for feature in self.featureset:
            self.theta[feature] = np.zeros((len(self.labelset)),dtype=float)
        self.theta["bias"] = np.ones((len(self.labelset)),dtype=float)
        ### training ###
        lastdevacc = 0
        acc = 0
        self.a = self.theta
        t  = 0
        converged = False
        while converged==False:
            print("epoch = " + repr(t))
            random.shuffle(training_data)
            for instance in training_data:
                gold = [token[1] for token in instance.snt]
                predicted = [token[1] for token in self.viterbi(instance)]
                if predicted !=gold:
                    self.update_weights(gold, instance, predicted)
            t +=1
            for feature in self.theta:
                self.a[feature] += self.theta[feature]
            lastdevacc = acc
            acc = self.check_dev_accuracy(dev_gold, dev_plain, 'auto_dev.tagged')
            print(acc)
            if acc < lastdevacc:
                converged = True
        for feature in self.a:
            self.a[feature] /= t
        self.theta = self.a


    def update_weights(self, gold, instance, predicted):
        zipped = list(zip(gold, predicted))
        for position in range(len(zipped)):
            correct = zipped[position][0]
            correct_label_index = self.labeldict[correct]
            predicted_tag = zipped[position][1]
            predicted_label_index = self.labeldict[predicted_tag]
            if correct != predicted_tag:
                for feature in instance.featurelist[position]:
                    ### handling unknowns ###
                    try:
                        self.theta[feature][predicted_label_index] -= 1
                        self.theta[feature][correct_label_index] += 1
                    except KeyError:
                        feature_stem = feature.split("=")[0]  ## word1, word_2, etc.
                        feature = feature_stem + "=" + self.unk  ## construct unk feature from it
                        self.theta[feature][predicted_label_index] -= 1
                        self.theta[feature][correct_label_index] += 1
                if position != 0:
                    previous_tag = zipped[position - 1][1]
                    self.theta["pos_1=" + previous_tag][predicted_label_index] -= 1
                    self.theta["pos_1=" + previous_tag][correct_label_index] += 1


    def check_dev_accuracy(self,dev_gold,dev_plain,preds_file):
        auto_dev_data = self.tag(dev_plain)
        with open(preds_file, 'w') as f:
            for sent in auto_dev_data:
                sent = ['_'.join(token) for token in sent]
                f.write(" ".join(sent) + "\n")
        return compute_acc(dev_gold,preds_file)

    def run_scorer(self,gold_file,preds_file):

        if not os.path.exists(preds_file):
            print(
                "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(preds_file))
            sys.exit(-1)
        python = 'python3.5'
        scorer = './scorer.py'
        gold = gold_file
        auto = preds_file
        command = "{} {} {} {}".format(python, scorer, gold, auto)

        print("Running scorer with command:", command)
        proc = subprocess.Popen(
            command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
            universal_newlines=True
        )
        proc.wait()