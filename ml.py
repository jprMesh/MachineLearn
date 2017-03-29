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

    def prediction(self, node):
        return None

    def runtest(self):
        correct = 0
        for node in self.test:
            pred = self.prediction(node)
            if pred > 0 and int(node[self.oindex]) > 0 or pred == int(node[self.oindex]):
                correct += 1
        return correct/len(self.test)


class KNN(ML):
    def __init__(self, k, file):
        super().__init__()
        self.loadfile(file)
        self.k = k

    def prediction(self, node):
        nn = self._neighbors(node, self.k)
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


knn = KNN(5, 'processed.cleveland.data')
print(knn.runtest())
