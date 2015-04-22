import sys
import numpy as np

class NeuralNetwork(object):
	def __init__(self,n_input, n_hidden, n_layers,n_output):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

		self.n_layers = n_layers
		if self.n_hidden == 0:
			self.n_weights = n_inputs  * n_output
		else:
			self.n_weights = (n_inputs * n_hidden) +\
					(n_hidden*n_output)  #((n_hidden*n_hidden + n_hidden) * (n_layers-1))+\ 	
		self.set_weights(self.generate_weights(n_weights))
		
		self.n_bias = n_hidden + n_output
		self.set_bias(self.generate_weights(n_bias)) 

	def generate_weights(self, count):
		return np.random.uniform(-0.1, 0.1, size = (1, count)).tolist()[0]
	
	def set_weights(self, weights):
		if self.n_hidden == 0:
			self.weights = [np.array(weights).reshape(self.n_outputs, self.n_inputs + 1)]
		else:
			self.weights = [np.array(weights[:(self.n_inputs+1)*self.n_hidden]).reshape(self.n_inputs+1, self.n_hidden)]
			#self.weights += [np.array(weights[:n_inputs+1]).reshape()]
                        self.weights += [np.array(weights[(self.n_inputs+1)*self.n_hidden:]).reshape(self.n_hidden+1, self.n_output)]

	def set_bias(self, bias):
		self.bias = [np.array(bias[:self.n_hidden]).reshape(1, self.n_hidden)]
		self.bias += [np.array(bias[self.n_hidden:]).reshape(1, self.n_output)]

	def feedforward(self, input_vec):
		input_values = input_vec
		output = input_vec
		for bias, weight_layer in zip(self.bias,self.weights):
			output = self.activation(input_vec, weight_layer, bias)
			input_vec = output
		return output
			
	def activation(self, input_vec, weights, bias):
		return  np.dot(input_vec, weight_layer) + bias
	
	def activation_prime(self, vec):
		return 1

	def train(self, train_vec, output_vec):
		alpha = 0.3
		for x,y in zip(train_vec, output_vec):
			(delta_weights, delta_bias) = self.backprop(x, y)
			self.weights = [ w - (alpha*nw) for w, nw in zip(self.weights, delta_weights)]
			self.bias = [ b -(alpha*nb) for b, nb in zip(self.bias, delta_bias)]

	def backprop(self,input_vec, output_vec):
		activations = [input_vec]
		output = x
		zs = []

		new_weights = [np.zeros(w.shape) for w in self.weights]
		new_bias = [np.zeros(b.shape) for b in self.bias]

		for bias, weight_layer in zip(self.bias,self.weights):	
                        z =  np.dot(output, weight_layer) + bias
			zs.append(z)
			output = self.activation(z)
			activations.append(output)

		# Delta for last layer, error function derivate
		delta = cost_derivative(activations[-1], output_vec)*activation_prime(zs[-1])
		new_weights[-1] = np.dot(activations[-2].transpose(), delta)
		new_bias[-1] = delta
		
		# now calculate the delta for all other layers

		for l in range(2, n_layers):
			delta = np.dot(delta, self.weights[-l+1].transpose())*activation_prime(zs[-l])
			new_weights[-l] = np.dot(activations[-l-1].transpose(), delta)
			new_bias[-l] = delta
		return (new_weights, new_bias) 
