# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
import numpy as np
from numpy import *
np.random.seed(1)      
data=np.genfromtxt("hero.txt",dtype = 'str', delimiter = ',')
hero = data[:,:10]
result = data[:,10]
result = result.astype(np.float64)
unique = np.unique(hero)
num = hero.shape[0]
X1 = np.zeros((num,97))
X2 = np.zeros((num,97))
y = result - 1
y = y.astype('uint8')
for i in range(num):
    for j in range(97):
        if unique[j] in hero[i,:5]:
            X1[i,j] = 1
        elif unique[j] in hero[i,5:]:
            X2[i,j] = 1
X = X1 - X2
 
# initialize parameters randomly
D = 97
K = 2
# initialize parameters randomly
h1 = 64 # size of hidden layer
h2 = 16
W1 = 0.01 * np.random.randn(D,h1)
b1 = np.zeros((1,h1))
W2 = 0.01 * np.random.randn(h1,h2)
b2 = np.zeros((1,h2))
W3 = 0.01 * np.random.randn(h2,K)
b3 = np.zeros((1,K))
 
# some hyperparameters
step_size = 0.5
reg = 1e-3 # regularization strength
batch_size = 1000
# gradient descent loop
num_examples = X.shape[0]
for i in range(100000):
  index = np.random.random_integers(num_examples,size=(batch_size)) - 1
  x1 = X[index,:]
  y1 = y[index]
  # evaluate class scores, [N x K]
  hidden1 = np.maximum(0, np.dot(x1, W1) + b1) # note, ReLU activation
  hidden2 = np.maximum(0, np.dot(hidden1, W2) + b2)
  scores = np.dot(hidden2, W3) + b3
   
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
   
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(batch_size),y1])
  data_loss = np.sum(corect_logprobs)/batch_size 
  reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
  loss = data_loss + reg_loss
  if i % 100 == 0:
    print("iteration %d: loss %f" % (i, loss))
   
  # compute the gradient on scores
  dscores = probs
  dscores[range(batch_size),y1] -= 1
  dscores /= batch_size
   
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW3 = np.dot(hidden2.T, dscores)
  db3 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden2 = np.dot(dscores, W3.T)
  # backprop the ReLU non-linearity
  dhidden2[hidden2 <= 0] = 0
  # finally into W,b
  dW2 = np.dot(hidden1.T, dhidden2)
  db2 = np.sum(dhidden2, axis=0, keepdims=True)
  dhidden1 = np.dot(dhidden2,W2.T)
  dhidden1[hidden1 <= 0] = 0
  dW1 = np.dot(x1.T, dhidden1)
  db1 = np.sum(dhidden1, axis=0, keepdims=True)
   
  # add regularization gradient contribution
  dW3 += reg * W3
  dW2 += reg * W2
  dW1 += reg * W1
   
  # perform a parameter update
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2
  W3 += -step_size * dW3
  b3 += -step_size * db3
 
# evaluate training set accuracy
W1 = np.round(W1,3)
W2 = np.round(W2,3)
W3 = np.round(W3,3)
hidden1 = np.maximum(0, np.dot(X, W1) + b1)
hidden2 = np.maximum(0, np.dot(hidden1, W2) + b2)
scores = np.dot(hidden2, W3) + b3
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
 
 
w11 = []
w22 = []
w33 = []
b11 = []
b22 = []
b33 = []
 
for i in range(W1.shape[0]):
    w11.append(''.join(str(W1[i].tolist()).split(" ")))
for i in range(W2.shape[0]):
    w22.append(''.join(str(W2[i].tolist()).split(" ")))
for i in range(W3.shape[0]):
    w33.append(''.join(str(W3[i].tolist()).split(" ")))
for i in range(b1.shape[0]):
    b11.append(''.join(str(b1[i].tolist()).split(" ")))
for i in range(b2.shape[0]):
    b22.append(''.join(str(b2[i].tolist()).split(" ")))
for i in range(b3.shape[0]):
    b33.append(''.join(str(b3[i].tolist()).split(" ")))
 
np.savetxt('W1.txt',[w11],fmt='%s',delimiter=',')
np.savetxt('b1.txt',[b11],fmt='%s',delimiter=',')
np.savetxt('W2.txt',[w22],fmt='%s',delimiter=',')
np.savetxt('b2.txt',[b22],fmt='%s',delimiter=',')
np.savetxt('W3.txt',[w33],fmt='%s',delimiter=',')
np.savetxt('b3.txt',[b33],fmt='%s',delimiter=',')
