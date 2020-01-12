# Neural Net
# - In this file we have an incomplete skeleton of
# a neural network implementation.  Follow the instructions in the
# problem description and complete the NotImplemented methods below.
#
import math
import random
import functools
import numpy as np
from utility import alphabetize, abs_mean
import copy
import matplotlib.pyplot as plt

def sigmoid(m):
	return 1 / (1 + math.exp(-m))

class ValuedElement(object):
	"""
	This is an abstract class that all Network elements inherit from
	"""
	def __init__(self,name,val):
		self.my_name = name
		self.my_value = val

	def set_value(self,val):
		self.my_value = val

	def get_value(self):
		return self.my_value

	def get_name(self):
		return self.my_name

	def __repr__(self):
		return "%s(%1.2f)" %(self.my_name, self.my_value)

class DifferentiableElement(object):
	"""
	This is an abstract interface class implemented by all Network
	parts that require some differentiable element.
	"""
	def output(self):
		raise NotImplementedError("This is an abstract method")

	def dOutdX(self, elem):
		raise NotImplementedError("This is an abstract method")

	def clear_cache(self):
		"""clears any precalculated cached value"""
		pass

class Input(ValuedElement,DifferentiableElement):
	"""
	Representation of an Input into the network.
	These may represent variable inputs as well as fixed inputs
	(Thresholds) that are always set to -1.
	"""
	def __init__(self,name,val):
		ValuedElement.__init__(self,name,val)
		DifferentiableElement.__init__(self)

	def output(self):
		"""
		Returns the output of this Input node.
		
		returns: number (float or int)
		"""
		return self.get_value()

	def dOutdX(self, elem):
		"""
		Returns the derivative of this Input node with respect to 
		elem.

		elem: an instance of Weight

		returns: number (float or int)
		"""
		return 0
	def output_weights(self):
		return 0

class Weight(ValuedElement):
	"""
	Representation of an weight into a Neural Unit.
	"""
	def __init__(self,name,val):
		ValuedElement.__init__(self,name,val)
		self.next_value = None

	def set_next_value(self,val):
		self.next_value = val

	def update(self):
		self.my_value = self.next_value


class Neuron(DifferentiableElement):
	"""
	Representation of a single sigmoid Neural Unit.
	"""
	def __init__(self, name, inputs, input_weights, use_cache=True):
		assert len(inputs)==len(input_weights)
		for i in range(len(inputs)):
			assert isinstance(inputs[i],(Neuron,Input))
			assert isinstance(input_weights[i],Weight)
		DifferentiableElement.__init__(self)
		self.my_name = name
		self.my_inputs = inputs # list of Neuron or Input instances
		self.my_weights = input_weights # list of Weight instances
		self.use_cache = use_cache
		self.clear_cache()
		self.my_descendant_weights = None
		self.my_direct_weights = None

	def get_descendant_weights(self):
		"""
		Returns a mapping of the names of direct weights into this neuron,
		to all descendant weights. For example if neurons [n1, n2] were connected
		to n5 via the weights [w1,w2], neurons [n3,n4] were connected to n6
		via the weights [w3,w4] and neurons [n5,n6] were connected to n7 via
		weights [w5,w6] then n7.get_descendant_weights() would return
		{'w5': ['w1','w2'], 'w6': ['w3','w4']}
		"""
		if self.my_descendant_weights is None:
			self.my_descendant_weights = {}
			inputs = self.get_inputs()
			weights = self.get_weights()
			for i in range(len(weights)):
				weight = weights[i]
				weight_name = weight.get_name()
				self.my_descendant_weights[weight_name] = set()
				input = inputs[i]
				if not isinstance(input, Input):
					descendants = input.get_descendant_weights()
					for name, s in descendants.items():
						st = self.my_descendant_weights[weight_name]
						st = st.union(s)
						st.add(name)
						self.my_descendant_weights[weight_name] = st

		return self.my_descendant_weights

	def isa_descendant_weight_of(self, target, weight):
		"""
		Checks if [target] is a indirect input weight into this Neuron
		via the direct input weight [weight].
		"""
		weights = self.get_descendant_weights()
		if weight.get_name() in weights:
			return target.get_name() in weights[weight.get_name()]
		else:
			raise Exception("weight %s is not connect to this node: %s"
							%(weight, self))

	def has_weight(self, weight):
		"""
		Checks if [weight] is a direct input weight into this Neuron.
		"""
		return weight.get_name() in self.get_descendant_weights()

	def get_weight_nodes(self):
		return self.my_weights

	def clear_cache(self):
		self.my_output = None
		self.my_doutdx = {}

	def output(self):
		# Implement compute_output instead!!
		if self.use_cache:
			# caching optimization, saves previously computed output.
			if self.my_output is None:
				self.my_output = self.compute_output()
			return self.my_output
		return self.compute_output()

	def compute_output(self):
		"""
		Returns the output of this Neuron node, using a sigmoid as
		the threshold function.

		returns: number (float or int)
		"""
		neuron_output = 0
		for i, my_input in enumerate(self.get_inputs()):
			neuron_output += my_input.output() * self.my_weights[i].get_value()
		return sigmoid(neuron_output)


	def dOutdX(self, elem):
		# Implement compute_doutdx instead!!
		if self.use_cache:
			# caching optimization, saves previously computed dOutdx.
			if elem not in self.my_doutdx:
				self.my_doutdx[elem] = self.compute_doutdx(elem)
			return self.my_doutdx[elem]
		return self.compute_doutdx(elem)

	def compute_doutdx(self, elem):
		"""
		Returns the derivative of this Neuron node, with respect to weight
		elem, calling output() and/or dOutdX() recursively over the inputs.

		elem: an instance of Weight

		returns: number (float/int)
		"""
		for i, my_weight in enumerate(self.get_weights()):
			if(my_weight.get_name() == elem.get_name()):
				neuron_output = self.output()
				return self.get_inputs()[i].output() * neuron_output * (1 - neuron_output)
		doutdx = 0
		for i, my_input in enumerate(self.get_inputs()):
			neuron_output = self.output()
			if(self.isa_descendant_weight_of(elem, self.get_weights()[i])):
				doutdx += self.get_weights()[i].get_value() * neuron_output * (1 - neuron_output) * my_input.dOutdX(elem)
		return doutdx

	def get_weights(self):
		return self.my_weights

	def get_inputs(self):
		return self.my_inputs

	def get_name(self):
		return self.my_name

	def __repr__(self):
		return "Neuron(%s)" %(self.my_name)

	def output_weights(self):
		weights_l2 = 0
		for weight in self.get_weights():
			weights_l2 += weight.get_value()**2
		for myinput in self.get_inputs():
			weights_l2 += myinput.output_weights()
		return weights_l2

class PerformanceElem(DifferentiableElement):
	"""
	Representation of a performance computing output node.
	This element contains methods for setting the
	desired output (d) and also computing the final
	performance P of the network.

	This implementation assumes a single output.
	"""
	def __init__(self,input,desired_value):
		assert isinstance(input,(Input,Neuron))
		DifferentiableElement.__init__(self)
		self.my_input = input
		self.my_desired_val = desired_value

	def output(self):
		"""
		Returns the output of this PerformanceElem node.
		
		returns: number (float/int)
		"""
		return -0.5*((self.my_desired_val - self.get_input().output())**2)

	def dOutdX(self, elem):
		"""
		Returns the derivative of this PerformanceElem node with respect
		to some weight, given by elem.

		elem: an instance of Weight

		returns: number (int/float)
		"""
		return (self.my_desired_val - self.get_input().output()) * self.get_input().dOutdX(elem)

	def set_desired(self,new_desired):
		self.my_desired_val = new_desired

	def get_input(self):
		return self.my_input

class RegularizedPerformanceElem(PerformanceElem):

	def __init__(self,input,desired_value):
		assert isinstance(input,(Input,Neuron))
		PerformanceElem.__init__(self, input, desired_value)
		self.landa = 0.0003

	def output(self):
		"""
		Returns the output of this PerformanceElem node.
		
		returns: number (float/int)
		"""
		return -0.5*((self.my_desired_val - self.get_input().output())**2) - 0.5 * self.landa * self.my_input.output_weights()


	def dOutdX(self, elem):
		"""
		Returns the derivative of this PerformanceElem node with respect
		to some weight, given by elem.

		elem: an instance of Weight

		returns: number (int/float)
		"""
		# raise NotImplementedError("Implement me!")
		return ((self.my_desired_val - self.get_input().output()) * self.get_input().dOutdX(elem))  - self.landa * elem.get_value()


class Network(object):
	def __init__(self,performance_node,neurons):
		self.inputs =  []
		self.weights = []
		self.performance = performance_node
		self.output = performance_node.get_input()
		self.neurons = neurons[:]
		self.neurons.sort(key=functools.cmp_to_key(alphabetize))
		for neuron in self.neurons:
			self.weights.extend(neuron.get_weights())
			for i in neuron.get_inputs():
				if isinstance(i,Input) and not ('i0' in i.get_name()) and not i in self.inputs:
					self.inputs.append(i)
		self.weights.reverse()
		self.weights = []
		for n in self.neurons:
			self.weights += n.get_weight_nodes()

	@classmethod
	def from_layers(self,performance_node,layers):
		neurons = []
		for layer in layers:
			if layer.get_name() != 'l0':
				neurons.extend(layer.get_elements())
		return Network(performance_node, neurons)

	def clear_cache(self):
		for n in self.neurons:
			n.clear_cache()

	def finite_defference(self):
		for weight in self.weights:

			self.clear_cache()

			output1 = self.performance.dOutdX(weight)

			performance_output = self.performance.output()
			weight.set_value(weight.get_value() + 0.00000001)

			self.clear_cache()

			new_performance_output = self.performance.output()

			output2 = (new_performance_output - performance_output)/0.00000001

			if(np.abs(output2 - output1) > 0.001):
				return False
		return True

def plot_decision_boundary(network, xmin, xmax, ymin, ymax):
	xtest = xmin
	ytest = ymin
	plot_xtest = []
	plot_ytest = []
	while(ytest <= ymax):
		while(xtest <= xmax):
			data = [xtest, ytest]
			for i in range(len(network.inputs)):
				network.inputs[i].set_value(data[i])

			network.clear_cache()

			if(network.output.output() > 0.5):
				plot_xtest.append(xtest)
				plot_ytest.append(ytest)
			xtest = xtest + 0.01

		ytest = ytest + 0.01
		xtest = xmin
	plt.axis([xmin, xmax, ymin, ymax])
	plt.scatter(plot_xtest, plot_ytest, marker='o')

	plt.show()


def seed_random():
	"""Seed the random number generator so that random
	numbers are deterministically 'random'"""
	random.seed(0)
	np.random.seed(0)

def random_weight():
	"""Generate a deterministic random weight"""
	# We found that random.randrange(-1,2) to work well emperically 
	# even though it produces randomly 3 integer values -1, 0, and 1.
	return random.randrange(-1, 2)

	# Uncomment the following if you want to try a uniform distribuiton 
	# of random numbers compare and see what the difference is.
	# return random.uniform(-1, 1)

	# When training larger networks, initialization with small, random
	# values centered around 0 is also common, like the line below:
	# return np.random.normal(0,0.1)

def make_neural_net_basic():
	"""
	Constructs a 2-input, 1-output Network with a single neuron.
	This network is used to test your network implementation
	and a guide for constructing more complex networks.

	Naming convention for each of the elements:

	Input: 'i'+ input_number
	Example: 'i1', 'i2', etc.
	Conventions: Start numbering at 1.
				 For the -1 inputs, use 'i0' for everything

	Weight: 'w' + from_identifier + to_identifier
	Examples: 'w1A' for weight from Input i1 to Neuron A
			  'wAB' for weight from Neuron A to Neuron B

	Neuron: alphabet_letter
	Convention: Order names by distance to the inputs.
				If equal distant, then order them left to right.
	Example:  'A' is the neuron closest to the inputs.

	All names should be unique.
	You must follow these conventions in order to pass all the tests.
	"""
	i0 = Input('i0', -1.0) # this input is immutable
	i1 = Input('i1', 0.0)
	i2 = Input('i2', 0.0)

	w1A01 = Weight('w1A01', 1)
	w2A01 = Weight('w2A01', 1)
	w0A01  = Weight('wA01', 1)

	# Inputs must be in the same order as their associated weights
	A01 = Neuron('A01', [i1,i2,i0], [w1A01,w2A01,w0A01])
	P = PerformanceElem(A01, 0.0)

	# Package all the components into a network
	# First list the PerformanceElem P, Then list all neurons afterwards
	net = Network(P,[A01])
	return net

def make_neural_net_two_layer():
	"""
	Create a 2-input, 1-output Network with three neurons.
	There should be two neurons at the first level, each receiving both inputs
	Both of the first level neurons should feed into the second layer neuron.

	See 'make_neural_net_basic' for required naming convention for inputs,
	weights, and neurons.
	"""
	seed_random()
	i0 = Input("i0", -1.0)
	i1 = Input('i1', 0.0)
	i2 = Input('i2', 0.0)

	w1A01 = Weight('w1A01', random_weight())
	w1A02 = Weight('w1A02', random_weight())
	w2A01 = Weight('w2A01', random_weight())
	w2A02 = Weight('w2A02', random_weight())
	w0A01 = Weight('w0A01', random_weight())
	w0A02 = Weight('w0A02', random_weight())
	wA01A03 = Weight('wA01A03', random_weight())
	wA02A03 = Weight('wA02A03', random_weight())
	w0A03 = Weight('w0A03', random_weight())

	A01 = Neuron("A01", [i1, i2, i0], [w1A01, w2A01, w0A01])
	A02 = Neuron("A02", [i1, i2, i0], [w1A02, w2A02, w0A02])
	A03 = Neuron("A03", [A01, A02, i0], [wA01A03, wA02A03, w0A03])

	P = PerformanceElem(A03, 0.0)

	net = Network(P, [A01, A02, A03])
	return net


def make_neural_net_challenging():
	"""
	Design a network that can in-theory solve all 3 problems described in
	the lab instructions.  Your final network should contain
	at most 5 neuron units.

	See 'make_neural_net_basic' for required naming convention for inputs,
	weights, and neurons.
	"""
	raise NotImplementedError("Implement me!")
   

def make_neural_net_two_moons():
	"""
	Create an overparametrized network with 40 neurons in the first layer
	and a single neuron in the last. This network is more than enough to solve
	the two-moons dataset, and as a result will over-fit the data if trained
	excessively.

	See 'make_neural_net_basic' for required naming convention for inputs,
	weights, and neurons.
	"""
	seed_random()
	i0 = Input("i0", -1.0)
	i1 = Input('i1', 0.0)
	i2 = Input('i2', 0.0)

	w1A01 = Weight('w1A01', random_weight())
	w1A02 = Weight('w1A02', random_weight())
	w1A03 = Weight('w1A03', random_weight())
	w1A04 = Weight('w1A04', random_weight())
	w1A05 = Weight('w1A05', random_weight())
	w1A06 = Weight('w1A06', random_weight())
	w1A07 = Weight('w1A07', random_weight())
	w1A08 = Weight('w1A08', random_weight())
	w1A09 = Weight('w1A09', random_weight())
	w1A10 = Weight('w1A10', random_weight())
	w1A11 = Weight('w1A11', random_weight())
	w1A12 = Weight('w1A12', random_weight())
	w1A13 = Weight('w1A13', random_weight())
	w1A14 = Weight('w1A14', random_weight())
	w1A15 = Weight('w1A15', random_weight())
	w1A16 = Weight('w1A16', random_weight())
	w1A17 = Weight('w1A17', random_weight())
	w1A18 = Weight('w1A18', random_weight())
	w1A19 = Weight('w1A19', random_weight())
	w1A20 = Weight('w1A20', random_weight())
	w1A21 = Weight('w1A21', random_weight())
	w1A22 = Weight('w1A22', random_weight())
	w1A23 = Weight('w1A23', random_weight())
	w1A24 = Weight('w1A24', random_weight())
	w1A25 = Weight('w1A25', random_weight())
	w1A26 = Weight('w1A26', random_weight())
	w1A27 = Weight('w1A27', random_weight())
	w1A28 = Weight('w1A28', random_weight())
	w1A29 = Weight('w1A29', random_weight())
	w1A30 = Weight('w1A30', random_weight())
	w1A31 = Weight('w1A31', random_weight())
	w1A32 = Weight('w1A32', random_weight())
	w1A33 = Weight('w1A33', random_weight())
	w1A34 = Weight('w1A34', random_weight())
	w1A35 = Weight('w1A35', random_weight())
	w1A36 = Weight('w1A36', random_weight())
	w1A37 = Weight('w1A37', random_weight())
	w1A38 = Weight('w1A38', random_weight())
	w1A39 = Weight('w1A39', random_weight())
	w1A40 = Weight('w1A40', random_weight())
	w2A01 = Weight('w2A01', random_weight())
	w2A02 = Weight('w2A02', random_weight())
	w2A03 = Weight('w2A03', random_weight())
	w2A04 = Weight('w2A04', random_weight())
	w2A05 = Weight('w2A05', random_weight())
	w2A06 = Weight('w2A06', random_weight())
	w2A07 = Weight('w2A07', random_weight())
	w2A08 = Weight('w2A08', random_weight())
	w2A09 = Weight('w2A09', random_weight())
	w2A10 = Weight('w2A10', random_weight())
	w2A11 = Weight('w2A11', random_weight())
	w2A12 = Weight('w2A12', random_weight())
	w2A13 = Weight('w2A13', random_weight())
	w2A14 = Weight('w2A14', random_weight())
	w2A15 = Weight('w2A15', random_weight())
	w2A16 = Weight('w2A16', random_weight())
	w2A17 = Weight('w2A17', random_weight())
	w2A18 = Weight('w2A18', random_weight())
	w2A19 = Weight('w2A19', random_weight())
	w2A20 = Weight('w2A20', random_weight())
	w2A21 = Weight('w2A21', random_weight())
	w2A22 = Weight('w2A22', random_weight())
	w2A23 = Weight('w2A23', random_weight())
	w2A24 = Weight('w2A24', random_weight())
	w2A25 = Weight('w2A25', random_weight())
	w2A26 = Weight('w2A26', random_weight())
	w2A27 = Weight('w2A27', random_weight())
	w2A28 = Weight('w2A28', random_weight())
	w2A29 = Weight('w2A29', random_weight())
	w2A30 = Weight('w2A30', random_weight())
	w2A31 = Weight('w2A31', random_weight())
	w2A32 = Weight('w2A32', random_weight())
	w2A33 = Weight('w2A33', random_weight())
	w2A34 = Weight('w2A34', random_weight())
	w2A35 = Weight('w2A35', random_weight())
	w2A36 = Weight('w2A36', random_weight())
	w2A37 = Weight('w2A37', random_weight())
	w2A38 = Weight('w2A38', random_weight())
	w2A39 = Weight('w2A39', random_weight())
	w2A40 = Weight('w2A40', random_weight())
	w0A01 = Weight('w0A01', random_weight())
	w0A02 = Weight('w0A02', random_weight())
	w0A03 = Weight('w0A03', random_weight())
	w0A04 = Weight('w0A04', random_weight())
	w0A05 = Weight('w0A05', random_weight())
	w0A06 = Weight('w0A06', random_weight())
	w0A07 = Weight('w0A07', random_weight())
	w0A08 = Weight('w0A08', random_weight())
	w0A09 = Weight('w0A09', random_weight())
	w0A10 = Weight('w0A10', random_weight())
	w0A11 = Weight('w0A11', random_weight())
	w0A12 = Weight('w0A12', random_weight())
	w0A13 = Weight('w0A13', random_weight())
	w0A14 = Weight('w0A14', random_weight())
	w0A15 = Weight('w0A15', random_weight())
	w0A16 = Weight('w0A16', random_weight())
	w0A17 = Weight('w0A17', random_weight())
	w0A18 = Weight('w0A18', random_weight())
	w0A19 = Weight('w0A19', random_weight())
	w0A20 = Weight('w0A20', random_weight())
	w0A21 = Weight('w0A21', random_weight())
	w0A22 = Weight('w0A22', random_weight())
	w0A23 = Weight('w0A23', random_weight())
	w0A24 = Weight('w0A24', random_weight())
	w0A25 = Weight('w0A25', random_weight())
	w0A26 = Weight('w0A26', random_weight())
	w0A27 = Weight('w0A27', random_weight())
	w0A28 = Weight('w0A28', random_weight())
	w0A29 = Weight('w0A29', random_weight())
	w0A30 = Weight('w0A30', random_weight())
	w0A31 = Weight('w0A31', random_weight())
	w0A32 = Weight('w0A32', random_weight())
	w0A33 = Weight('w0A33', random_weight())
	w0A34 = Weight('w0A34', random_weight())
	w0A35 = Weight('w0A35', random_weight())
	w0A36 = Weight('w0A36', random_weight())
	w0A37 = Weight('w0A37', random_weight())
	w0A38 = Weight('w0A38', random_weight())
	w0A39 = Weight('w0A39', random_weight())
	w0A40 = Weight('w0A40', random_weight())
	w0A41 = Weight('w0A41', random_weight())
	wA01A41 = Weight('wA01A41', random_weight())
	wA02A41 = Weight('wA02A41', random_weight())
	wA03A41 = Weight('wA03A41', random_weight())
	wA04A41 = Weight('wA04A41', random_weight())
	wA05A41 = Weight('wA05A41', random_weight())
	wA06A41 = Weight('wA06A41', random_weight())
	wA07A41 = Weight('wA07A41', random_weight())
	wA08A41 = Weight('wA08A41', random_weight())
	wA09A41 = Weight('wA09A41', random_weight())
	wA10A41 = Weight('wA10A41', random_weight())
	wA11A41 = Weight('wA11A41', random_weight())
	wA12A41 = Weight('wA12A41', random_weight())
	wA13A41 = Weight('wA13A41', random_weight())
	wA14A41 = Weight('wA14A41', random_weight())
	wA15A41 = Weight('wA15A41', random_weight())
	wA16A41 = Weight('wA16A41', random_weight())
	wA17A41 = Weight('wA17A41', random_weight())
	wA18A41 = Weight('wA18A41', random_weight())
	wA19A41 = Weight('wA19A41', random_weight())
	wA20A41 = Weight('wA20A41', random_weight())
	wA21A41 = Weight('wA21A41', random_weight())
	wA22A41 = Weight('wA22A41', random_weight())
	wA23A41 = Weight('wA23A41', random_weight())
	wA24A41 = Weight('wA24A41', random_weight())
	wA25A41 = Weight('wA25A41', random_weight())
	wA26A41 = Weight('wA26A41', random_weight())
	wA27A41 = Weight('wA27A41', random_weight())
	wA28A41 = Weight('wA28A41', random_weight())
	wA29A41 = Weight('wA29A41', random_weight())
	wA30A41 = Weight('wA30A41', random_weight())
	wA31A41 = Weight('wA31A41', random_weight())
	wA32A41 = Weight('wA32A41', random_weight())
	wA33A41 = Weight('wA33A41', random_weight())
	wA34A41 = Weight('wA34A41', random_weight())
	wA35A41 = Weight('wA35A41', random_weight())
	wA36A41 = Weight('wA36A41', random_weight())
	wA37A41 = Weight('wA37A41', random_weight())
	wA38A41 = Weight('wA38A41', random_weight())
	wA39A41 = Weight('wA39A41', random_weight())
	wA40A41 = Weight('wA40A41', random_weight())

	A01 = Neuron("A01", [i1, i2, i0], [w1A01, w2A01, w0A01])
	A02 = Neuron("A02", [i1, i2, i0], [w1A02, w2A02, w0A02])
	A03 = Neuron("A03", [i1, i2, i0], [w1A03, w2A03, w0A03])
	A04 = Neuron("A04", [i1, i2, i0], [w1A04, w2A04, w0A04])
	A05 = Neuron("A05", [i1, i2, i0], [w1A05, w2A05, w0A05])
	A06 = Neuron("A06", [i1, i2, i0], [w1A06, w2A06, w0A06])
	A07 = Neuron("A07", [i1, i2, i0], [w1A07, w2A07, w0A07])
	A08 = Neuron("A08", [i1, i2, i0], [w1A08, w2A08, w0A08])
	A09 = Neuron("A09", [i1, i2, i0], [w1A09, w2A09, w0A09])
	A10 = Neuron("A10", [i1, i2, i0], [w1A10, w2A10, w0A10])
	A11 = Neuron("A11", [i1, i2, i0], [w1A11, w2A11, w0A11])
	A12 = Neuron("A12", [i1, i2, i0], [w1A12, w2A12, w0A12])
	A13 = Neuron("A13", [i1, i2, i0], [w1A13, w2A13, w0A13])
	A14 = Neuron("A14", [i1, i2, i0], [w1A14, w2A14, w0A14])
	A15 = Neuron("A15", [i1, i2, i0], [w1A15, w2A15, w0A15])
	A16 = Neuron("A16", [i1, i2, i0], [w1A16, w2A16, w0A16])
	A17 = Neuron("A17", [i1, i2, i0], [w1A17, w2A17, w0A17])
	A18 = Neuron("A18", [i1, i2, i0], [w1A18, w2A18, w0A18])
	A19 = Neuron("A19", [i1, i2, i0], [w1A19, w2A19, w0A19])
	A20 = Neuron("A20", [i1, i2, i0], [w1A20, w2A20, w0A20])
	A21 = Neuron("A21", [i1, i2, i0], [w1A21, w2A21, w0A21])
	A22 = Neuron("A22", [i1, i2, i0], [w1A22, w2A22, w0A22])
	A23 = Neuron("A23", [i1, i2, i0], [w1A23, w2A23, w0A23])
	A24 = Neuron("A24", [i1, i2, i0], [w1A24, w2A24, w0A24])
	A25 = Neuron("A25", [i1, i2, i0], [w1A25, w2A25, w0A25])
	A26 = Neuron("A26", [i1, i2, i0], [w1A26, w2A26, w0A26])
	A27 = Neuron("A27", [i1, i2, i0], [w1A27, w2A27, w0A27])
	A28 = Neuron("A28", [i1, i2, i0], [w1A28, w2A28, w0A28])
	A29 = Neuron("A29", [i1, i2, i0], [w1A29, w2A29, w0A29])
	A30 = Neuron("A30", [i1, i2, i0], [w1A30, w2A30, w0A30])
	A31 = Neuron("A31", [i1, i2, i0], [w1A31, w2A31, w0A31])
	A32 = Neuron("A32", [i1, i2, i0], [w1A32, w2A32, w0A32])
	A33 = Neuron("A33", [i1, i2, i0], [w1A33, w2A33, w0A33])
	A34 = Neuron("A34", [i1, i2, i0], [w1A34, w2A34, w0A34])
	A35 = Neuron("A35", [i1, i2, i0], [w1A35, w2A35, w0A35])
	A36 = Neuron("A36", [i1, i2, i0], [w1A36, w2A36, w0A36])
	A37 = Neuron("A37", [i1, i2, i0], [w1A37, w2A37, w0A37])
	A38 = Neuron("A38", [i1, i2, i0], [w1A38, w2A38, w0A38])
	A39 = Neuron("A39", [i1, i2, i0], [w1A39, w2A39, w0A39])
	A40 = Neuron("A40", [i1, i2, i0], [w1A40, w2A40, w0A40])
	A41 = Neuron("A41", [A01, A02, A03, A04, A05, A06, A07, 
		A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18,
		A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, 
		A30, A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, i0], 
		[wA01A41, wA02A41, wA03A41, wA04A41, wA05A41, wA06A41, wA07A41, 
		wA08A41, wA09A41, wA10A41, wA11A41, wA12A41, wA13A41, wA14A41, wA15A41, wA16A41, wA17A41, wA18A41,
		wA19A41, wA20A41, wA21A41, wA22A41,wA23A41, wA24A41, wA25A41, wA26A41, wA27A41, wA28A41, wA29A41, 
		wA30A41, wA31A41, wA32A41, wA33A41,wA34A41, wA35A41, wA36A41, wA37A41, wA38A41, wA39A41, wA40A41, w0A41])

	P = RegularizedPerformanceElem(A41, 0.0)
	# P = PerformanceElem(A41, 0.0)

	net = Network(P, [A01, A02, A03, A04, A05, A06, A07, 
		A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18,
		A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, 
		A30, A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, A41])
	return net

def train(network,
		  data,      # training data
		  rate=1.0,  # learning rate
		  target_abs_mean_performance=0.0001,
		  max_iterations=10000,
		  verbose=False):
	"""Run back-propagation training algorithm on a given network.
	with training [data].   The training runs for [max_iterations]
	or until [target_abs_mean_performance] is reached.
	"""

	iteration = 0
	while iteration < max_iterations:
		fully_trained = False
		performances = []  # store performance on each data point
		for datum in data:
			# set network inputs
			for i in range(len(network.inputs)):
				network.inputs[i].set_value(datum[i])

			# set network desired output
			network.performance.set_desired(datum[-1])

			# clear cached calculations
			network.clear_cache()

			# compute all the weight updates
			for w in network.weights:
				w.set_next_value(w.get_value() +
								 rate * network.performance.dOutdX(w))

			# set the new weights
			for w in network.weights:
				w.update()

			# save the performance value
			performances.append(network.performance.output())

			# clear cached calculations
			network.clear_cache()

		# compute the mean performance value
		abs_mean_performance = abs_mean(performances)

		if abs_mean_performance < target_abs_mean_performance:
			if verbose:
				print("iter %d: training complete.\n"\
					  "mean-abs-performance threshold %s reached (%1.6f)"\
					  %(iteration,
						target_abs_mean_performance,
						abs_mean_performance))
			break

		iteration += 1

		if iteration % 10 == 0 and verbose:
			print("iter %d: mean-abs-performance = %1.6f"\
				  %(iteration,
					abs_mean_performance))

	plot_decision_boundary(network, -2, 2, -2, 2)
	# print(network.finite_defference())

def test(network, data, verbose=False):
	"""Test the neural net on some given data."""
	correct = 0
	for datum in data:

		for i in range(len(network.inputs)):
			network.inputs[i].set_value(datum[i])

		# clear cached calculations
		network.clear_cache()

		result = network.output.output()
		prediction = round(result)

		network.clear_cache()

		if prediction == datum[-1]:
			correct+=1
			if verbose:
				print("test(%s) returned: %s => %s [%s]" %(str(datum),
														   str(result),
														   datum[-1],
														   "correct"))
		else:
			if verbose:
				print("test(%s) returned: %s => %s [%s]" %(str(datum),
														   str(result),
														   datum[-1],
														   "wrong"))

	return float(correct)/len(data)
