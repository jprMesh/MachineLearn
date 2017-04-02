from math import sqrt
import re
from heapq import *


class ML:
    def __init__(self):
        self.oindex = 13

    def loadfile(self, file):
        with open(file, 'r') as df:
            data = [line.strip().split(',') for line in df]
        self.train = [data[x] for x in range(len(data)) if x % 3 != 0]
        self.test = [data[x] for x in range(len(data)) if x % 3 == 0]

    def prediction(self, data_entry):
        return None

    def runtest(self):
        correct = 0
        for data_entry in self.test:
            pred = self.prediction(data_entry)
            if pred == None:
                continue
            if (pred > 0 and int(data_entry[self.oindex]) > 0) or pred == int(data_entry[self.oindex]):
                correct += 1
        return 100*correct/len(self.test)


class KNN(ML):
    def __init__(self, k, file):
        super().__init__()
        self.loadfile(file)
        self.k = k

    def prediction(self, data_entry):
        nn = self._neighbors(data_entry, self.k)
        predic = 0
        for n in nn:
            predic += int(int(n[1]) > 0) * 2 - 1
        return int(predic >= 0)

    def _distance(self, node1, node2, omit_index):
        d = sqrt(sum([(float(node1[x]) - float(node2[x]))**2
                      for x in range(len(node1))
                      if re.match("-?\d*\.?\d+$", node1[x])
                         and re.match("-?\d*\.?\d+$", node2[x])
                         and x != omit_index]))
        return d

    def _neighbors(self, node, k):
        nnbors = []
        for element in self.train:
            distance = self._distance(element, node, self.oindex)
            heappush(nnbors, (distance, element[self.oindex]))
        return [heappop(nnbors) for x in range(k)]


class DTree(ML):
    def __init__(self, file):
        super().__init__()
        self.loadfile(file)

    def doTrain(self, maxdepth, minrecords):
        self.maxdepth = maxdepth
        self.minrecords = minrecords
        self.root = self._bestSplit(range(len(self.train)))
        self._extendTree(self.root, 1)

    def prediction(self, data_entry, treenode=None):
        if treenode == None:
            treenode = self.root
        if not re.match("-?\d*\.?\d+$", data_entry[treenode['index']]):
            return None

        if float(data_entry[treenode['index']]) > float(treenode['value']):
            if isinstance(treenode['left'], dict):
                return self.prediction(data_entry, treenode['left'])
            else:
                return float(treenode['left'])
        else:
            if isinstance(treenode['right'], dict):
                return self.prediction(data_entry, treenode['right'])
            else:
                return float(treenode['right'])

    def _gini(self, indices1, indices2):
        gini_coeff = 0
        for ind in [indices1, indices2]:
            if len(ind) == 0:
                continue
            #print([int(self.train[x][self.oindex]) for x in ind])
            prop0 = ([int(self.train[x][self.oindex]) for x in ind].count(0)
                          / float(len(ind)))
            p1 = [int(self.train[x][self.oindex]) for x in ind]
            prop1 = (len(p1) - p1.count(0)) / float(len(ind))
            gini_coeff += prop0 * (1 - prop0)
            gini_coeff += prop1 * (1 - prop1)
        return gini_coeff

    def _split(self, indices, s_ind, s_val):
        l1, l2 = [], []
        for x in indices:
            if not re.match("-?\d*\.?\d+$", self.train[x][s_ind]):
                continue
            if float(self.train[x][s_ind]) > float(s_val):
                l1.append(x)
            else:
                l2.append(x)
        return l1, l2

    def _bestSplit(self, indices):
        bestindex = None
        bestvalue = None
        bestgini = 65535
        bestsplits = None
        for row in indices:
            for field_index in range(len(self.train[row]) - 1):
                if not re.match("-?\d*\.?\d+$", self.train[row][field_index]):
                    continue
                splits = self._split(indices, field_index, self.train[row][field_index])
                gini = self._gini(splits[0], splits[1])
                if gini < bestgini:
                    bestindex = field_index
                    bestvalue = self.train[row][field_index]
                    bestgini = gini
                    bestsplits = splits
        return {'index':bestindex, 'value':bestvalue, 'splitindices':bestsplits}

    def _getTerminal(self, indices):
        vals = [self.train[x][-1] for x in indices]
        return max(vals, key=vals.count)

    def _extendTree(self, node, depth):
        left, right = node['splitindices']
        if not (left and right):
            node['left'] = self._getTerminal(left + right)
            node['right'] = node['left']
            return
        if depth >= self.maxdepth:
            node['left'] = self._getTerminal(left)
            node['right'] = self._getTerminal(right)
            return
        
        if len(left) <= self.minrecords:
            node['left'] = self._getTerminal(left)
        else:
            node['left'] = self._bestSplit(left)
            self._extendTree(node['left'], depth+1)
        
        if len(right) <= self.minrecords:
            node['right'] = self._getTerminal(right)
        else:
            node['right'] = self._bestSplit(right)
            self._extendTree(node['right'], depth+1)


knn = KNN(5, 'processed.cleveland.data')
#print(knn.runtest())

dt = DTree('processed.cleveland.data')
dt.doTrain(5, 5) # (maxdepth, minrecords)
print(dt.runtest())

