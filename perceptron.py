import sys
from numpy import *

def csvReader(path):
	in1 = open(path,'r')
	tx = []
	ty =[]
	for line in in1:
		words  = line.strip().split(",")
		#ty.append(mapp[words[-1]])
		temp = []
		for i in range(0, len(words)-1):
			temp.append(float(words[i]))
		temp.append(mapp[words[-1]])
		tx.append(array(temp))
	return (tx, ty)
	
class Perceptron:
	
	def __init__(self, features):
		self.weights = zeros(features)
		self.bias = 0
		self.d = features

	def train(self,training, max_iter):
		
		for i in range(0, max_iter):
			trainx = training
			random.shuffle(trainx)
			for tx in trainx:
				ty = tx[-1]
				x = tx[:-1]	
				a = dot(x, self.weights) + self.bias
				if (multiply(ty, a)) <=0:
					self.weights = add(self.weights, multiply(ty,x))
					self.bias += ty

	def test(self,testx):
		a = dot(testx[:-1], self.weights) + self.bias
		if a > 0:
			return 1
		else:
			return -1

def __main__:
	mapp = dict()
	mapp["Iris-setosa"] = -1
	mapp["Iris-versicolor"] = 1
	path = sys.argv[1]
	(f, y) = csvReader(path)
	percep = Perceptron(4)
	percep.train(f[1:80], 100)
	for i in range(81,100):
		print(str(percep.test(f[i])) + " " + str(f[i][-1]))

