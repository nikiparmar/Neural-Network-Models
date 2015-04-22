from neural_network import *
import sys
import numpy as np


def csvReader(path):
        in1 = open(path,'r')
        tx = []
        ty =[]
        for line in in1:
                words  = line.strip().split(",")
		tempy = np.zeros(no_output)
                tempy[mapp[words[-1]]] = 1
                temp = []
                for i in range(0, len(words)-1):
                        temp.append(float(words[i]))
                tx.append(np.array([temp]))
		ty.append(tempy)
        return (tx, ty)


if __name__ == "__main__":
	global no_output
	no_output = 2
        mapp = dict()
        mapp["Iris-setosa"] = 0
        mapp["Iris-versicolor"] = 1
	network = NeuralNetwork(4,5,4,2)
        path = sys.argv[1]
	(trainx, trainy) = csvReader(path)
	'''
	trainx = []
	trainy = []
	for line in open(path):
		line = line.strip().split("\t")
		tx = []
		ty = []
		for x in line[0].split(" "):
			tx.append(int(x))
		trainx.append(np.array([tx]))
		for y in line[1].split(" "):
			ty.append(int(y))
		trainy.append(np.array([ty]))
	'''
	network.train(trainx[1:80], trainy[1:80])
	network.test(trainx[80:99])	
