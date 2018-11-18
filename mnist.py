import numpy as np
import gzip

Y = []
m = []    
nfilters = []
nfilters.insert(0,1)     
npools = []
nConv = 0
i = 0

def convertToOneHot(vector, num_classes=None):

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def getMnist():
	with gzip.open('train-images-idx3-ubyte.gz','rb') as f:
		data = np.frombuffer(f.read(), np.uint8, offset = 16)
	data = data.reshape(60000,784)
	data = np.divide(data, 1)

	with gzip.open('train-labels-idx1-ubyte.gz','rb') as f:
		labels = np.frombuffer(f.read(), np.uint8, offset = 8)
	labels = convertToOneHot(labels, 10)
	labels = labels.reshape(60000, 10, 1)

	with gzip.open('t10k-images-idx3-ubyte.gz','rb') as f:
		testdata = np.frombuffer(f.read(), np.uint8, offset = 16)
	testdata = testdata.reshape(10000, 784)
	testdata = np.divide(testdata, 1)

	with gzip.open('t10k-labels-idx1-ubyte.gz','rb') as f:
		testlabels = np.frombuffer(f.read(), np.uint8, offset = 8)
	testlabels = convertToOneHot(testlabels, 10)
	testlabels = testlabels.reshape(10000, 10, 1)


	return data, labels, testdata, testlabels

class multiply():
	def __init__(self):                  
		pass

	def feedforward(self,x,y):	
		self.X = x
		self.Y = y	
		return np.matmul(self.X,self.Y)

	def backprop(self,dz):                  # weights, activatoions and derivatives
		self.dz = dz
		return np.matmul(self.dz,np.transpose(self.Y)), np.matmul(np.transpose(self.X),self.dz)   # self.y = (activations) self.x = (weights)

class sigmoid():
	def __init__(self):
		pass
	
	def feedforward(self,y,x):
		self.x = x.astype(np.float128)
		self.sigma = 1/(1+np.exp(-self.x))
		return self.sigma  

	def backprop(self,dz):
		self.dz = dz
		self.size = np.size(self.x)                         # size of the pre-activations as its not a function argument
		self.one = np.ones((self.size,1), dtype=float)      # one matrix , to calculate the gradient
		# return 0, np.multiply(self.dz,np.subtract(self.one,self.sigma)) 
		return 0, self.dz
		# return 0, np.multiply(self.dz,np.multiply(self.sigma,np.subtract(self.one,self.sigma)))

class relu():
	def __init__(self):
		pass

	def feedforward(self,y,x):
		self.X = x
		# self.max = np.amax(self.X)
		# if self.max>1:
		# 	self.X = self.X/self.max
		return np.maximum(self.X,0)        

	def backprop(self,dz):
		self.dz = dz
		self.size = np.size(self.X) 
		self.relu = np.zeros([self.size,1])
		self.relu[self.X > 0] = 1
		return 0, np.multiply(self.dz,self.relu)

class softmax():
	def __init__(self):
		pass

	def feedforward(self,y,x):
		self.x = x             
		self.max = np.amax(self.x)
		self.num = np.exp(self.x-self.max)
		self.den = np.sum(self.num)
		self.sigma = np.divide(self.num,self.den)
		# print self.x,self.max,self.num, self.sigma
		return self.sigma

	def backprop(self,dz):
		self.dz = dz
		self.size = np.size(self.x)                         # size of the pre-activations as its not a function argument
		self.one = np.ones((self.size,1), dtype=float)      # one matrix , to calculate the gradient 
		# self.dx = -np.matmul(self.x,np.transpose(self.x))
		# self.dx = np.add(np.multiply(1-np.identity(self.size),self.dx),np.diag(np.multiply(np.subtract(self.one,self.sigma),self.sigma)))
		self.dx = -np.matmul(self.sigma,np.ones([1,self.size]))
		self.dx = np.add(np.identity(self.size),self.dx)
		# print("soft", self.dx)
		return 0, np.matmul(self.dx,self.dz)
		# return 0 , self.dz


class add():

	def __init__(self):
		pass

	def feedforward(self,x,y):
		self.x = x
		self.y = y
		return np.add(self.x,self.y)

	def backprop(self,dz):
		self.dz = dz
		return self.dz, self.dz

class convolution():

	def __init__(self):
		pass

	def feedforward(self,x,y):
		global i
		self.x = x                                           #kernel
		self.y = y                                           #images
		Y.insert(i,self.y)
		self.M = m[i]               #weights
		self.n = np.size(np.asarray(self.y),1)
		self.nf = 1
		self.af = 1
		for self.p in range(i+2):
			self.nf *= nfilters[self.p]
		for self.p in range(i+1):
			self.af *= nfilters[self.p]
		if self.n%2 != 0:
			self.y = np.concatenate((self.y,np.zeros([self.af,1,self.n])),axis = 1)
			self.y = np.concatenate((self.y,np.zeros([self.af,self.n+1,1])),axis = 2)
			self.n += self.n%2
		self.o = int(self.n-self.M+1)
		self.convolution = np.zeros([self.nf,self.o,self.o])        #kernel size
		self.c = 0
		for self.yf in range(self.af):
			for self.f in range(nfilters[i+1]): 
				for self.i in range(self.o):
					for self.j in range(self.o):
						self.convolution[self.c,self.i,self.j] = np.sum(np.multiply(self.x[self.f],self.y[self.yf][self.i:self.i+self.M , self.j:self.j+self.M]))
				self.c += 1
		if npools[i] == 0:
			i += 1
		# self.max = np.amax(self.convolution)
		# if self.max>1:
		# 	self.convolution = self.convolution/self.max
		# print self.convolution
		return self.convolution

	def backprop(self,dz):
		global i
		global m
		self.dw = np.zeros([nfilters[i+1],m[i],m[i]])
		self.da = np.asarray(Y[i])
		self.size = np.size(np.asarray(Y[i]),2)
		self.span = self.size-m[i]+1
		if self.size%2 != 0:
			dz = np.delete(dz,self.size-m[i]+1,1)
			dz = np.delete(dz,self.size-m[i]+1,2)
		self.dz = dz
		self.da[self.da != 0] = 0
		self.s = 0
		self.nf = 1
		self.yf = 1
		for self.p in range(i+2):
			self.nf *= nfilters[self.p]
		for self.p in range(i+1):
			self.yf *= nfilters[self.p]
		for self.af in range(self.yf):
			for self.f in range(nfilters[i+1]):
				for self.i in range(m[i]):                     
					for self.j in range(m[i]):
						self.dw[self.f] += self.dz[self.s,self.i,self.j]*Y[i][self.af,self.i:self.i+m[i] , self.j:self.j+m[i]]
						self.da[self.af,self.i:self.i+self.span , self.j:self.j+self.span] = np.add(self.da[self.af,self.i:self.i+self.span , self.j:self.j+self.span],self.x[self.f,self.i,self.j]*self.dz[self.s])
				self.s += 1
		i -= 1
		if i == -1: 
			del Y[:]
		# print "dw", self.dw, "da", self.da
		return self.dw, self.da 

class pooling():
	def __init__(self):
		pass

	def feedforward(self,y,x):
		global i
		global m
		self.x = x
		self.nf = 1
		self.yf = 1
		for self.p in range(i+2):
			self.nf *= nfilters[self.p]
		for self.p in range(i+1):
			self.yf *= nfilters[self.p]
		self.size = int(np.sqrt(np.size(self.x)/self.nf))
		self.pooled = np.zeros([self.nf,self.size//2,self.size//2])
		self.pool = np.zeros([self.nf,self.size,self.size])
		for self.f in range(self.nf):
			self.r = 0
			for self.i in range(self.size//2):
				self.c = 0
				for self.j in range(self.size//2):
					self.arg = np.argmax(self.x[self.f,self.r:self.r+2 , self.c:self.c+2])
					self.pool[self.f,self.r+self.arg//2,self.c+self.arg%2] = 1
					self.pooled[self.f,self.i,self.j] = np.amax(self.x[self.f,self.r:self.r+2 , self.c:self.c+2])
					self.c += 2
				self.r += 2
		if i == nConv-1:
			self.s = self.nf*(self.size//2)**2
			i += 1
			# print "pool", self.pooled.reshape(self.s,1)
			return self.pooled.reshape(self.s, 1)
		else:
			i += 1
			return self.pooled

	def backprop(self,dz):
		global i
		self.backpool = np.zeros([self.nf,self.size,self.size])
		self.dz = dz.reshape(self.nf,self.size//2,self.size//2)
		for self.f in range(self.nf):
			self.r = 0
			for self.i in range(self.size//2):
				self.c = 0
				for self.j in range(self.size//2):
					self.pool[self.f,self.r:self.r+2, self.c:self.c+2] = self.pool[self.f,self.r:self.r+2, self.c:self.c+2]*self.dz[self.f,self.i,self.j]
					self.c += 2
				self.r += 2
		self.backpool = self.pool
		# print "bpool", self.backpool
		return 0,self.backpool

class raspberry(object):

	nlist = []                                        # list of all weights nd biases in w,b,0,w,b,0 order
	nodes = []                                     # list of computational nodes
	gradients = []
	momentone = []
	momenttwo = []
	rank = 0  
	nrank = 0                                     
	nfeeds = 0 
	efeeds = 0 
	avgloss = 0
	total = 0
	correct = 0 
	beta1 =  0.9
	beta2 = 0.999                             

	def __init__(self):
		pass

	def addcream(self,inputs,neurons,activation):     # inputs = no of inputs to the layer, neurons = no of neurons in the layer      

		self.inputs = inputs               # assignment to an instance variable for use in other methods
		self.neurons = neurons
		self.activation = activation		
		self.biases = np.zeros([self.neurons,1])  
		self.nodes.insert(self.rank, multiply())
		self.nodes.insert(self.rank+1, add())
		if self.activation == "relu": 
			self.nodes.insert(self.rank+2, relu())
			self.weights = np.random.normal(0,1/np.sqrt(self.inputs/2),(self.neurons,self.inputs))
		if self.activation == "sigmoid": 
			self.nodes.insert(self.rank+2, sigmoid())
			self.weights = np.random.normal(0,1/np.sqrt(self.inputs),(self.neurons,self.inputs))
		if self.activation == "softmax": 
			self.nodes.insert(self.rank+2, softmax())
			self.weights = np.random.normal(0,1/np.sqrt(self.inputs),(self.neurons,self.inputs))
		self.nlist.insert(self.rank, self.weights)    #inserting every weights nd biases matrices into nlist in w,b,0,w,b,0 order
		self.nlist.insert(self.rank+1, self.biases)
		self.nlist.insert(self.rank+2, 0)
		self.gradients.insert(self.rank, np.zeros([self.neurons,self.inputs]))
		self.gradients.insert(self.rank+1, self.biases)
		self.gradients.insert(self.rank+2, 0)
		self.momentone.insert(self.rank, np.zeros([self.neurons,self.inputs]))
		self.momentone.insert(self.rank+1, self.biases)
		self.momentone.insert(self.rank+2, 0)
		self.momenttwo.insert(self.rank, np.zeros([self.neurons,self.inputs]))
		self.momenttwo.insert(self.rank+1, self.biases)
		self.momenttwo.insert(self.rank+2, 0)
		self.rank += 3

	def addcheese(self,size,nf,p):                   #size = size of the square kernel, npools = no of pooling layers
		global m
		global nfilters
		global npools
		global nConv
		m.insert(nConv,size)
		nfilters.insert(nConv+1,nf)
		npools.insert(nConv,p)
		nConv += 1
		self.size = size
		self.nf = nf
		self.p = p
		self.weights = np.zeros((nf,self.size,self.size))
		self.weights = self.weights.astype(np.float32)
		for self.f in range(nf):
			self.weights[self.f,:,:] = np.random.normal(0,1/float(self.size),(self.size,self.size))
		self.nodes.insert(self.rank, convolution())
		self.rank += 1
		self.nlist.insert(self.nrank, self.weights)
		self.gradients.insert(self.nrank, np.zeros((nf,self.size,self.size)))
		self.momentone.insert(self.nrank, np.zeros((nf,self.size,self.size)))
		self.momenttwo.insert(self.nrank, np.zeros((nf,self.size,self.size)))
		self.nrank += 1
		for self.i in range(self.p):
			self.nodes.insert(self.rank, pooling())
			self.rank += 1	
			self.nlist.insert(self.nrank, 0)
			self.gradients.insert(self.nrank, 0)
			self.momentone.insert(self.nrank, 0)
			self.momenttwo.insert(self.nrank, 0)
			self.nrank += 1

	def feedforward(self,glows,labels):    # glows = images 
		global i
		# if self.nfeeds%batch == 0: self.avgloss = 0 
		self.nfeeds += 1 
		self.pos = 0                        # for the position in the list
		self.feed = glows                       # input image is given here, nd subsequently other layers' activations during feedforward      
		self.labels = labels
		for self.i in self.nlist:
			# print "feed:", self.feed
			self.feed = self.nodes[self.pos].feedforward(self.i,self.feed)		
			self.pos+=1
		i -= 1
		self.backfeed = -self.labels                     #backprop starts here
		self.pos = self.rank-1                       #for traversing the computational node nd nlist
		# print "nfeeds:", self.nfeeds
		for k in range(self.rank):
			self.updates, self.backfeed = self.nodes[self.pos].backprop(self.backfeed)
			# self.momentone[self.pos] = np.add(self.beta1*np.asarray(self.momentone[self.pos]),(1-self.beta1)*self.updates)
			# self.momentone[self.pos] = self.momentone[self.pos]/(1.-self.beta1**self.nfeeds) 		
			# self.momenttwo[self.pos] = np.add(self.beta2*np.asarray(self.momenttwo[self.pos]),(1-self.beta2)*np.square(self.updates))
			# self.momenttwo[self.pos] = np.sqrt(self.momenttwo[self.pos]/(1.-self.beta2**self.nfeeds))	
			# self.gradients[self.pos] += np.divide(self.momentone[self.pos],self.momenttwo[self.pos]+1e-7)
			# self.gradients[self.pos] += np.asarray(self.updates)
			self.momentone[self.pos] = np.add(0.99*np.asarray(self.momentone[self.pos]),self.updates)
			self.gradients[self.pos] += np.asarray(self.momentone[self.pos])
			self.pos -= 1	
		# print "labels:", self.labels	
		self.avgloss += -np.sum(np.multiply(self.labels,np.log(1e-7+self.feed)))
		# print(-np.sum(np.multiply(self.labels,np.log(1e-7+self.feed))))
		# self.avgloss += -np.sum(np.add(np.multiply(self.labels,np.log(1e-7+self.feed)),np.multiply(1-self.labels,np.log(1.0000001-self.feed))))

	def backprop(self,alpha):
		global batch
		self.pos = self.rank-1                       #for traversing the computational node nd nlist
		self.alpha = alpha
		for k in range(self.rank):
			# print self.gradients[self.pos]
			self.gradients[self.pos] = self.gradients[self.pos] / 8      #averaging the gradients obtained
			self.nlist[self.pos] = self.nlist[self.pos] - self.alpha*self.gradients[self.pos]
			self.gradients[self.pos].fill(0)
			self.pos -= 1
		print ("loss:", self.avgloss/float(self.nfeeds))
	 	# print "weights0:", self.nlist[0]
	 	# print "weights1:", self.nlist[2]
	 	# print "weights2:", self.nlist[4]
	 	# print "biases2:" , self.nlist[5]
	 	# print "weights3:", self.nlist[7]
	 	# print "biases3:" , self.nlist[8]
	 	# self.nfeeds = 0

	def evaluate(self,images,labels):
		self.efeeds += 1
		self.p = 0                        # for the position in the list
		self.f = images                      # input image is given here, nd subsequently other layers' activations during feedforward      
		self.l = labels
		self.total += 1
		for self.i in self.nlist:
			if  self.efeeds >= 9999:
				print self.f
			self.f = self.nodes[self.p].feedforward(self.i,self.f)		
			self.p += 1
		if np.argmax(self.f) == np.argmax(self.l): 
			self.correct += 1
		if self.total == 10000:
			print (self.correct) 
			# self.accuracy = self.correct/self.total
			# print "accuracy: ", self.accuracy

	def bake(self,epoch,batch,alpha):
		global i
		data, labels, testdata, testlabels = getMnist()
		self.batch = batch
		for self.k in range(epoch):
			for self.j in range(60000//batch):
				for self.i in range(batch*self.j,batch*(self.j+1)):
					self.d = data[self.i].reshape(1,28,28)
					# self.d = data[self.i].reshape(data[self.i].shape[0], 1)/255.
					self.d[self.d>0] = 1.
					# self.l = labels[self.i].reshape(labels[self.i].shape[0], 1)
					i = 0			
					self.feedforward(self.d,labels[self.i])
				print ((self.j+1)*batch, self.k)   
				self.backprop(alpha)
		for self.i in range(10000):
			print (self.i)
			self.td = testdata[self.i].reshape(1,28,28)
			# self.td = testdata[self.i].reshape(testdata[self.i].shape[0], 1)/255.
			self.td[self.td>0] = 1.
			# self.tl = testlabels[self.i].reshape(testlabels[self.i].shape[0], 1)
			i = 0
			self.evaluate(self.td,testlabels[self.i])
		print "weights0:", self.nlist[0]
		print "weights1:", self.nlist[2]
		# print "weights2:", self.nlist[4]
		print "biases2:" , self.nlist[3]
		print "weights3:", self.nlist[5]
		print "biases3:" , self.nlist[6]


model = raspberry()
model.addcheese(3,10,1)
# model.addcheese(3,10,1)
model.addcream(1690,42,'relu')
model.addcream(42,10,'softmax')
model.bake(1,8,0.005)