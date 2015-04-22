import sys
import numpy as np

class NeuralNetwork(object):
	def __init__(self,n_input, n_hidden, n_layers,n_output):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

		self.n_layers = n_layers
		if self.n_hidden == 0:
			self.n_weights = n_input  * n_output
		else:
			self.n_weights = (n_input * n_hidden) +\
					((n_hidden*n_hidden) * (n_layers-3))+\
					(n_hidden*n_output)  #((n_hidden*n_hidden + n_hidden) * (n_layers-1))+\ 	
		self.set_weights(self.generate_weights(self.n_weights))
		
		self.n_bias = n_hidden*(self.n_layers-2) + n_output
		self.set_bias(self.generate_weights(self.n_bias)) 

	def generate_weights(self, count):
		return np.random.uniform(-0.1, 0.1, size = (1, count)).tolist()[0]
	
	def set_weights(self, weights):
		if self.n_hidden == 0:
			self.weights = [np.array(weights).reshape(self.n_output, self.n_input)]
		else:
			self.weights = [np.array(weights[:(self.n_input)*self.n_hidden]).reshape(self.n_input, self.n_hidden)]
			start = (self.n_input)*self.n_hidden
			for i in range(0, self.n_layers-3):
				self.weights += [np.array(weights[start:start+(self.n_hidden*self.n_hidden)]).reshape(self.n_hidden, self.n_hidden)]
				start += (self.n_hidden*self.n_hidden)
                        self.weights += [np.array(weights[start:]).reshape(self.n_hidden, self.n_output)]

	def set_bias(self, bias):
		self.bias = [np.array(bias[:self.n_hidden]).reshape(1, self.n_hidden)]
		start = self.n_hidden
		for i in range(0, self.n_layers-3):
			self.bias += [np.array(bias[start:start+self.n_hidden]).reshape(1, self.n_hidden)]
			start += self.n_hidden
		self.bias += [np.array(bias[start:]).reshape(1, self.n_output)]

	def feedforward(self, input_vec):
		input_values = input_vec
		output = input_vec
		for bias, weight_layer in zip(self.bias,self.weights):
			z = np.dot(output, weight_layer) + bias	
			output = self.activation(z)
		return output
			
	def activation(self, value):
		return np.where(value < 0, 0, 1)
	
	def activation_prime(self, vec):
		return 1

	def 


	def train(self, train_vec, output_vec):
		alpha = 0.3
		for i in range(1,100):
			for x,y in zip(train_vec, output_vec):
				(delta_weights, delta_bias) = self.backprop(x, y)
				self.weights = [ w - (alpha*nw) for w, nw in zip(self.weights, delta_weights)]
				self.bias = [ b -(alpha*nb) for b, nb in zip(self.bias, delta_bias)]

	def test(self, train_vec):
		for t in train_vec:
			output = self.feedforward(t)
			print output
			

	def backprop(self,input_vec, output_vec):
		activations = [input_vec]
		output = input_vec
		zs = []

		new_weights = [np.zeros(w.shape) for w in self.weights]
		new_bias = [np.zeros(b.shape) for b in self.bias]

		for bias, weight_layer in zip(self.bias,self.weights):	
                        z =  np.dot(output, weight_layer) + bias
			zs.append(z)
			output = self.activation(z)
			activations.append(output)

		# Delta for last layer, error function derivate
		delta = self.cost_derivative(activations[-1], output_vec)
		new_weights[-1] = np.dot(activations[-2].transpose(), delta)
		new_bias[-1] = delta
		
		# now calculate the delta for all other layers

		for l in range(2, self.n_layers):
			delta = np.dot(delta, self.weights[-l+1].transpose())
			new_weights[-l] = np.dot(activations[-l-1].transpose(), delta)
			new_bias[-l] = delta
		return (new_weights, new_bias) 

	def cost_derivative(self, predicted, actual):
		return np.subtract(predicted,actual)
