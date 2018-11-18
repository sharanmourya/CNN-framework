import pickle
import numpy as np
import cv2
import gzip
Y = []
m = []     #size of filter
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

def readPickle(file):
	with open(file,'rb') as fo:
		a = pickle.load(fo, encoding = 'bytes')
	labels = np.asarray(a[b'labels'])
	data = np.asarray(a[b'data'])
	r, g, b = np.split(data, 3 ,axis = 1)
	r = r.reshape(-1,32,32)
	g = g.reshape(-1,32,32)
	b = b.reshape(-1,32,32)
	pics = [r,b,g]
	pics_resized = np.zeros([r.shape[0],28,28,3])
	for i in range(r.shape[0]):
		for j in range(3):
			pics_resized[i,:,:,j] = cv2.resize(pics[j][i], (28,28))
	return pics_resized, labels

def getCifar():
	for i in range(5):
		file = 'cifar-10-batches-py/data_batch_'+str(i+1)
		if i == 0:
			data, labels = readPickle(file)
		else:
			dataTemp, labelsTemp = readPickle(file)
			data = np.concatenate((data, dataTemp))
			labels = np.concatenate((labels, labelsTemp))
	testdata, testlabels = readPickle('cifar-10-batches-py/test_batch')
	testlabels = convertToOneHot(testlabels)
	testlabels = testlabels.reshape(10000,10,1)
	labels = convertToOneHot(labels)
	labels = labels.reshape(50000,10,1)
	with open('cifar_data.pkl','wb') as f:
		pickle.dump([data, labels, testdata, testlabels], f, pickle.HIGHEST_PROTOCOL)
	return data, labels, testdata, testlabels
data, labels, testdata, testlabels = getCifar()
print(np.shape(data), np.shape(labels), np.shape(testdata), np.shape(testlabels))

class multiply():
	def __init__(self):                  
		pass

	def feedforward(self,x,y):	
		self.X = x
		self.Y = y	
		return np.matmul(self.X,self.Y)

	def backprop(self,dz):                  # weights, activatoions and derivatives
		self.dz = dz
		# print("weights", np.matmul(self.dz,np.transpose(self.Y)))
		return np.matmul(self.dz,np.transpose(self.Y)), np.matmul(np.transpose(self.X),self.dz)   # self.y = (activations) self.x = (weights)

class sigmoid():
	def __init__(self):
		pass
	
	def feedforward(self,y,x):
		self.x = x#.astype(np.float128)
		self.sigma = 1/(1+np.exp(-self.x))
		return self.sigma  

	def backprop(self,dz):
		self.dz = dz
		self.size = np.size(self.x)                         # size of the pre-activations as its not a function argument
		self.one = np.ones((self.size,1), dtype=float)      # one matrix , to calculate the gradient 
		return 0, np.multiply(self.dz,np.subtract(self.one,self.sigma))
		return 0 , self.dz

class relu():
	def __init__(self):
		pass

	def feedforward(self,y,x):
		self.X = x
		return np.maximum(self.X,0)        

	def backprop(self,dz):
		self.dz = dz
		self.size = np.size(self.X) 
		self.relu = np.zeros([self.size,1])
		self.relu[self.X > 0] = 1
		# print("relu", np.multiply(self.dz,self.relu))
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
		return self.sigma

	def backprop(self,dz):
		global i
		self.dz = dz
		self.size = np.size(self.x)                         # size of the pre-activations as its not a function argument
		self.one = np.ones((self.size,1), dtype=float)      # one matrix , to calculate the gradient 
		# self.dx = -np.matmul(self.x,np.transpose(self.x))
		# self.dx = np.add(np.multiply(1-np.identity(self.size),self.dx),np.diag(np.multiply(np.subtract(self.one,self.sigma),self.sigma)))
		self.dx = -np.matmul(self.sigma,np.ones([1,self.size]))
		self.dx = np.add(np.identity(self.size),self.dx)
		# print("soft", np.matmul(self.dx,self.dz))
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
		# print("add", self.dz)
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
		self.n = np.size(np.asarray(self.y),2)
		self.nf = 1
		self.af = 1
		for self.p in range(i+2):
			self.nf *= nfilters[self.p]
		for self.p in range(i+1):
			self.af *= nfilters[self.p]
		if self.n%2 != 0:
			self.y = np.concatenate((self.y,np.zeros([self.af,3,1,self.n])),axis = 2)
			self.y = np.concatenate((self.y,np.zeros([self.af,3,self.n+1,1])),axis = 3)
			self.n += self.n%2
		self.o = int(self.n-self.M+1)
		self.convolution = np.zeros([self.nf,3,self.o,self.o])        #kernel size
		self.c = 0
		for self.yf in range(self.af):
			for self.f in range(nfilters[i+1]): 
				for self.i in range(self.o):
					for self.j in range(self.o):
						self.convolution[self.c,:,self.i,self.j] = np.sum(np.multiply(self.x[self.f],self.y[self.yf][:,self.i:self.i+self.M , self.j:self.j+self.M]))
				self.c += 1
		if npools[i] == 0:
			i += 1
		return self.convolution

	def backprop(self,dz):
		global i
		global m
		print (i)
		self.dw = np.zeros((nfilters[i+1],3,m[i],m[i]))
		self.da = np.asarray(Y[i])
		self.size = np.size(np.asarray(Y[i]),2)
		self.span = self.size-m[i]+1
		if self.size%2 != 0:
			dz = np.delete(dz,self.size-m[i]+1,2)
			dz = np.delete(dz,self.size-m[i]+1,3)
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
						for self.k in range(3):
							self.dw[self.f,self.k] += self.dz[self.s,self.k,self.i,self.j]*Y[i][self.af,self.k,self.i:self.i+m[i] , self.j:self.j+m[i]]
							self.da[self.af,self.k,self.i:self.i+self.span , self.j:self.j+self.span] += self.x[self.f,self.k,self.i,self.j]*self.dz[self.s,self.k]
				self.s += 1
		i -= 1
		if i == -1: 
			del Y[:]
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
		self.size = int(np.sqrt(np.size(self.x)/self.nf/3))
		self.pooled = np.zeros([self.nf,3,self.size//2,self.size//2])
		self.pool = np.zeros([self.nf,3,self.size,self.size])
		for self.f in range(self.nf):
			self.r = 0
			for self.i in range(self.size//2):
				self.c = 0
				for self.j in range(self.size//2):
					for self.k in range(3):
						self.arg = np.argmax(self.x[self.f,self.k,self.r:self.r+2 , self.c:self.c+2])
						self.pool[self.f,self.k,self.r+self.arg//2,self.c+self.arg%2] = 1
						self.pooled[self.f,self.k,self.i,self.j] = np.amax(self.x[self.f,self.k,self.r:self.r+2 , self.c:self.c+2])
					self.c += 2
				self.r += 2
		if i == nConv-1:
			self.s = 3*self.nf*(self.size//2)**2
			i += 1
			return self.pooled.reshape(self.s, 1)
		else:
			i += 1
			return self.pooled

	def backprop(self,dz):
		global i
		self.backpool = np.zeros([self.nf,3,self.size,self.size])
		self.dz = dz.reshape(self.nf,3,self.size//2,self.size//2)
		for self.f in range(self.nf):
			self.r = 0
			for self.i in range(self.size//2):
				self.c = 0
				for self.j in range(self.size//2):
					for self.k in range(3):
						self.pool[self.f,self.k,self.r:self.r+2, self.c:self.c+2] = self.pool[self.f,self.k,self.r:self.r+2, self.c:self.c+2]*self.dz[self.f,self.k,self.i,self.j]
					self.c += 2
				self.r += 2
		self.backpool = self.pool
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
		self.nlist.insert(self.nrank, self.weights)    #inserting every weights nd biases matrices into nlist in w,b,0,w,b,0 order
		self.nlist.insert(self.nrank+1, self.biases)
		self.nlist.insert(self.nrank+2, 0)
		self.gradients.insert(self.nrank, np.zeros([self.neurons,self.inputs]))
		self.gradients.insert(self.nrank+1, self.biases)
		self.gradients.insert(self.nrank+2, 0)
		self.momentone.insert(self.nrank, np.zeros([self.neurons,self.inputs]))
		self.momentone.insert(self.nrank+1, self.biases)
		self.momentone.insert(self.nrank+2, 0)
		self.momenttwo.insert(self.nrank, np.zeros([self.neurons,self.inputs]))
		self.momenttwo.insert(self.nrank+1, self.biases)
		self.momenttwo.insert(self.nrank+2, 0)
		self.nrank += 3
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
		self.weights = np.zeros((nf,3,self.size,self.size))
		# self.weights = self.weights.astype(np.float32)
		for self.f in range(nf):
			for self.k in range(3):
				self.weights[self.f,self.k,:,:] = np.random.normal(0,1/self.size,(self.size,self.size))
		# print(self.weights)
		self.nodes.insert(self.rank, convolution())
		self.rank += 1
		self.nlist.insert(self.nrank, self.weights)
		self.gradients.insert(self.nrank, np.zeros((nf,3,self.size,self.size)))
		self.momentone.insert(self.nrank, np.zeros((nf,3,self.size,self.size)))
		self.momenttwo.insert(self.nrank, np.zeros((nf,3,self.size,self.size)))
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
		self.nfeeds += 1 
		self.labels = labels
		self.pos = 0                     # for the position in the list
		self.feed = glows                       # input image is given here, nd subsequently other layers' activations during feedforward      
		for self.i in range(self.rank):
			# print("weights",self.i, self.nlist[self.pos])	
			self.feed = self.nodes[self.pos].feedforward(self.nlist[self.pos],self.feed)
			self.pos+=1
		i -= 1
		self.backfeed = -self.labels
		self.pos = self.rank-1                       #for traversing the computational node nd nlist
		for k in range(self.rank):
			self.updates, self.backfeed = self.nodes[self.pos].backprop(self.backfeed)
			# print("rank",self.rank)
			# print ("updates",self.pos, self.updates, "backfeed",self.pos, self.backfeed)
			# self.momentone[self.pos] = np.add(self.beta1*np.asarray(self.momentone[self.pos]),(1-self.beta1)*self.updates)
			# self.momentone[self.pos] = self.momentone[self.pos]/(1.-self.beta1**self.nfeeds) 		
			# self.momenttwo[self.pos] = np.add(self.beta2*np.asarray(self.momenttwo[self.pos]),(1-self.beta2)*np.square(self.updates))
			# self.momenttwo[self.pos] = np.sqrt(self.momenttwo[self.pos]/(1.-self.beta2**self.nfeeds))	
			# self.gradients[self.pos] += np.divide(self.momentone[self.pos],self.momenttwo[self.pos]+1e-7)
			# self.gradients[self.pos] += np.asarray(self.updates)
			self.momentone[self.pos] = np.add(0.99*np.asarray(self.momentone[self.pos]),self.updates)
			self.gradients[self.pos] += np.asarray(self.momentone[self.pos])
			self.pos -= 1	
			# print("feed:", self.feed)
			# print("labels:", self.labels)	
		self.avgloss += -np.sum(np.multiply(self.labels,np.log(1e-7+self.feed)))
		# print("loss", self.avgloss)
			# self.avgloss += -np.sum(np.add(np.multiply(self.labels,np.log(1e-7+self.feed)),np.multiply(1-self.labels,np.log(1.0000001-self.feed))))

	def backprop(self,alpha):
		self.pos = self.nrank-1                       #for traversing the computational node nd nlist
		self.alpha = alpha
		for l in range(self.nrank):
			self.gradients[self.pos] = self.gradients[self.pos] / float(self.batch)       #averaging the gradients obtained
			# print(self.gradients[self.pos])
			self.nlist[self.pos] = self.nlist[self.pos] - self.alpha*self.gradients[self.pos]
			np.asarray(self.gradients[self.pos]).fill(0)
			self.pos -= 1
		# if self.nfeeds % 1000 == 0:
		print ("GD loss:", self.avgloss/self.nfeeds)
		# self.nfeeds = 0

	def test(self,images,labels):
		self.pos = 0                        # for the position in the list
		self.feed = images                      # input image is given here, nd subsequently other layers' activations during feedforward      
		self.labels = labels
		self.total += 1     
		for self.i in range(self.rank):
			self.feed = self.nodes[self.pos].feedforward(self.nlist[self.pos],self.feed)	
			self.pos+=1
		if np.argmax(self.feed) == np.argmax(self.labels): 
			self.correct += 1
		if self.total == 10000:
			print (self.correct) 

	def bake(self,epoch,batch,alpha):
		global i
		data, labels, testdata, testlabels = getCifar()
		data = data.astype(float)
		testdata = testdata.astype(float)
		self.batch = batch
		for self.k in range(epoch):
			# print (self.k)   
			for self.j in range(50000//batch):
				for self.i in range(batch*self.j,batch*(self.j+1)):
					self.d = data[self.i].reshape(1,3,28,28)/255.
					# self.d[self.d>0] = 1
					i = 0			
					self.feedforward(self.d,labels[self.i])
				self.backprop(alpha)
				# if self.j%1000 == 0:
				print(self.j*batch)
				# print(self.nlist[0])
				# print(self.nlist[2])
				# # print(self.nlist[3])
				# print(self.nlist[4])
				# print(self.nlist[6])
				# print(self.nlist[7])
		for self.i in range(10000):
			print ("testing:", self.i) 
			self.td = testdata[self.i].reshape(1,3,28,28)/255.
			# self.td[self.td>0] = 1
			i = 0
			self.test(self.td,testlabels[self.i])

model = raspberry()
# model.addcheese(5,10,1)
# model.addcheese(3,10,1)
# model.addcheese(3,10,1)
# model.addcream(12000,32,'relu')
model.addcheese(3,32,1)
# model.addcheese(3,16,1)
# model.addcheese(3,8,1)
# model.addcheese(3,8,1)
model.addcream(16224,512,'relu')
# model.addcream(128,64,'relu')
model.addcream(512,10,'softmax')
model.bake(1,1,1e-3)