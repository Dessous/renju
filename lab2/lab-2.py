
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # L2 - Multilayer perceptron
# 
# ### Papers
# 1. [TensorFlow](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
# 
# ### TensorFlow
# 1. [Installing TensorFlow](https://www.tensorflow.org/install/)
# 2. [Basics of TensorFlow](https://www.tensorflow.org/get_started/get_started)
# 3. [Mnist with TensorFlow](https://www.tensorflow.org/get_started/mnist/pros)
# 4. [TensorFlow Mechanics](https://www.tensorflow.org/get_started/mnist/mechanics)
# 5. [Visualization](https://www.tensorflow.org/get_started/graph_viz)
# 
# 
# ### One more thing
# 1. [Jupyter tutorial](https://habrahabr.ru/company/wunderfund/blog/316826/)
# 2. [Plot.ly](https://plot.ly/python/)
# 3. [Widgets](http://jupyter.org/widgets.html)

# ### 1. Linear multi-classification problem
# 
# We have already learned binary linear classifier
# $$y = \text{sign}(w^Tx).$$
# There are [several approaches](https://en.wikipedia.org/wiki/Multiclass_classification) to solve the problem of multi-class classification. For example [reduction](https://en.wikipedia.org/wiki/Multiclass_classification#Transformation_to_Binary) of problem to binary classifier or [modification](https://en.wikipedia.org/wiki/Support_vector_machine#Multiclass_SVM) of the known model. However we are interested in approaches that is applied in neural networks.
# 
# For each class $c \in 1, \dots, |C|$ we have an individual row $w_i$ of matrix $W$. Then the probability of $x$ belonging to a particular class is equal to
# $$p_i = \frac{\exp(w^T_ix)}{\sum_j \exp(w^T_jx)}.$$
# This is nothing, but [softmax](https://en.wikipedia.org/wiki/Softmax_function) function of $Wx$.
# $$(p_1, \dots, p_{|C|}) = \text{softmax}(Wx).$$
# 
# If you look closely, $\text{softmax}$ is a more general variant of sigmoid. To see this, it suffices to consider the case $|C|=2$. As usual the training can be reduced to minimization of the empirical risk, namely, optimization problem
# $$\arg\min_W Q(W) = \arg\min_W -\frac{1}{\mathcal{l}}\sum_y\sum_i [y = i] \cdot \ln(p_i(W)).$$
# Actually, the maximization of the log-likelihood is written above.
# 
# #### Exercises
# 1. Find $\frac{dQ}{dW}$ in matrix form (hint: start with $\frac{dQ}{dw_i}$ for begining).
# 2. Please plot several mnist images (e.g using grid 5x5).
# 3. Train linear multi-label classifier for [mnist](https://www.kaggle.com/c/digit-recognizer) dataset with TensorFlow (possible, [this tutorial](https://www.tensorflow.org/get_started/mnist/pros) can help you).
# 4. Chek accuracy on train and validation sets.
# 5. Use a local [TensorBoard instance](https://www.tensorflow.org/get_started/graph_viz) to visualize resulted graph (no need to include in lab).

# ##### Ex.2

# In[2]:


data = np.loadtxt("train.csv", delimiter=',', skiprows=1)


# In[3]:


data[:,1:] = data[:,1:] / 255
labels = np.zeros((data[:,0].shape[0], 10))
for i, j in enumerate(data[:,0]):
    labels[i, int(j)] = 1
x_train, x_val, y_train, y_val = train_test_split(data[:,1:], labels, test_size=0.2, random_state=17)
print(x_train.shape)
print(x_val.shape)


# In[4]:


fig, axes = plt.subplots(nrows=3, ncols=6)

for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')
fig.set_figheight(8)
fig.set_figwidth(16)
plt.show()


# ##### Ex.3

# In[4]:


def batch(X, Y, batch_size):
    start = 0
    end = start + batch_size
    while (end < X.shape[0]):
        yield (X[start:end, :], Y[start:end, :])
        start = end
        end += batch_size


# In[6]:


with tf.device("/gpu:0"):
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x,W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    W = tf.zeros([784,10])
    b = tf.zeros([10])
    for i in range(5):
        for cur_batch in batch(x_train, y_train, 100):
            train_step.run(feed_dict={x: cur_batch[0], y_: cur_batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy on train set: ', accuracy.eval(feed_dict={x: x_train, y_: y_train}))
    acc_without_regularization = accuracy.eval(feed_dict={x: x_val, y_: y_val})
    print('accuracy on validation set: ', acc_without_regularization)

    sess.close()


# Let's briefly touch on themes of regularization. As was discussed before, there are different approaches. We focus on the modification of loss function.
# 
# $$\arg\min_W -\frac{1}{\mathcal{l}}\sum_y\sum_i [y = i] \cdot \ln(p_i(W)) + \lambda_1 L_1(W) + \lambda_2 L_2(W)$$
# 
# 1. $L_1(W) = sum_{i,j} |w_{i,j}|$ - sparsify weights (force to not use uncorrelated features)
# 2. $L_2(W) = sum_{i,j} w_{i,j}^2$ - minimize weights (force to not overfit)
# 
# #### Exercises
# 1. Train model again using both type of regularization.
# 2. Plot matrix of weights.
# 3. Which pixels have zero weights? What does it mean?
# 4. Have you improved accuracy on validation?

# In[7]:


lmbd1 = 0.001
lmbd2 = 0.001

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b
L1 = tf.reduce_sum(tf.abs(W))
L2 = tf.reduce_sum(tf.pow(W, 2))

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) \
        + lmbd1 * L1 + lmbd2 * L2

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(5):
    for cur_batch in batch(x_train, y_train, 100):
        train_step.run(feed_dict={x: cur_batch[0], y_: cur_batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy on validation set with regularization: ', accuracy.eval(feed_dict={x: x_val, y_: y_val}))

A = np.asarray(W.eval())

fig, axes = plt.subplots(nrows=2, ncols=5)

for i, ax in enumerate(axes.flat):
    img = ax.imshow(A[:,i].reshape((28, 28)), cmap='gray')
    plt.colorbar(img, ax=ax)
fig.set_figheight(10)
fig.set_figwidth(25)
plt.show()

sess.close()


# Нулевой вес имеют пиксели, по которым нельзя сделать никаких выводов. То есть это либо "фоновые" пиксели, либо, если говорить про конкретную цифру, пиксели, которые встречаются и в этой цифре и в других с примерно одинаковой частотой.

# Качество с регуляризацией только ухудшается. Видимо, это связано с тем, что градиентный спуск и так хорошее решение при заданных ограничениях, и дополнительные ограничения (регуляризация) только ухудшают точность.

# ### 2. Universal approximation theorem
# 
# What if we add more layers to our model? Namely, we train two matrix $W_2$ and $W_1$
# $$softmax(W_2\cdot(W_1x)).$$
# 
# At first glance adding more parameters helps to increase the generalizing ability of the model. Buy actually we have the same model $softmax(Wx)$, where $W = W_2\cdot W_1$. But everyting changes with adding ome more layer. Let's add nonlinear function $\sigma$ between $W_2$ and $W_1$
# 
# $$softmax(W_2\cdot \sigma(W_1x)).$$
# 
# Kurt Hornik showed in 1991 that it is not the specific choice of the nonlinear function, but rather the multilayer feedforward architecture itself which gives neural networks the potential of being universal approximators. The output units are always assumed to be linear. For notational convenience, only the single output case will be shown. The general case can easily be deduced from the single output case.
# 
# Let $\sigma(\cdot)$ be a nonconstant, bounded, and monotonically-increasing continuous function.
# Let $\mathcal{S}_m \subset \mathbb{R}^m$ denote any compact set. 
# Then, given any $\varepsilon > 0$ and any coninuous function $f$ on $\mathcal{S}_m$, there exist an integer $N$ and real constants $v_i$, $b_i$ amd real vectors $w_i$ that
# 
# $$\left| \sum _{i=1}^{N}v_{i}\sigma \left( w_i^{T} x+b_i \right) - f(x) \right| < \varepsilon, ~~~ \forall x \in \mathcal{S}_m.$$
# 
# The theorem has non-constructive proof, it meams that no estimates for $N$ and no method to find approximation's parameters.
# 
# #### Exercises
# 1. Let $\sigma$ – [heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) and $x \in \{0, 1\}^2$. Prove that $y = \sigma(wx + b)$ can approximate boolean function **OR** (hint: use constructive proof).
# 2. What about **AND** function?
# 3. Is it possible to implement **XOR**? Prove your words.
# 4. Prove that 2-layer network can implement any boolean function.
# 
# #### More useful facts:
# 1. A 2-layer network in in $\mathbb{R}^n$ allows to define convex polyhedron..
# 2. A 3-layer network in в $\mathbb{R}^n$ allows to define a not necessarily convex and not even necessarily connected area.

# ### 3. Backpropagation
# Backpropagation is a method used to calculate the error contribution of each layer after a batch of data. It is a special case of an older and more general technique called automatic differentiation. In the context of learning, backpropagation is commonly used by the gradient descent optimization algorithm to adjust the weight of layers by calculating the gradient of the loss function. This technique is also sometimes called backward propagation of errors, because the error is calculated at the output and distributed back through the network layers. The main motivation of method is simplify evaluation of gradient which is complex problem for multilayer nets.
# 
# We need the following notation. Let $(y^1,\dots,y^n) = f(x^1,\dots,x^n)$ is some differentiable function and $\frac{dy}{dx}$ is matrix
# $$\frac{dy}{dx} = \Big[ J_{ij} = \frac{\partial y^i}{\partial x^j} \Big]$$
# 
# Without violating the generality, we can assume that each layer is a function $x_{i} = f(x_{i-1}, w_i)$. As last layer we add loss function, so we can assume our multi-layer net as function $Q(x_0) = Q(f_n(f_{n-1}(\dots, w_{n-1}), w_n))$.
# 
# #### Forward step
# Propagation forward through the network to generate the output values. Calculation of the loss function.
# 
# #### Backward step
# Let's look at last layer. We can simply find $\frac{dQ}{dx_n}$. Now we can evaluate 
# 
# $$\frac{dQ}{dw_n} = \frac{dQ}{dx_n}\frac{dx_n}{dw_n} \text{ and } \frac{dQ}{dx_{n-1}} = \frac{dQ}{dx_n}\frac{dx_n}{dx_{n-1}}$$
# 
# Now we need calculate $\frac{dQ}{dw_{n-2}}$ и $\frac{dQ}{dx_{n-2}}$. But we have the same situation. We know $\frac{dQ}{dx_k}$, so can evaluate $\frac{dQ}{dw_k}$ and $\frac{dQ}{dx_{k-1}}$. Repeating this operation we find all the gradients. Now it's only remains to make a gradient step to update weights.
# 
# #### Exercises
# 1. Read more about [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
# 2. Train 2 layer net. Use sigmoid as nonlinearity.
# 3. Check accuracy on validation set.
# 4. Use [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) or LeakyReLu as nonlinearity. Compare accuracy and convergence with previous model.
# 5. Play with different architectures (add more layers, regularization and etc).
# 6. Show your best model.
# 7. How does quality change with adding layers. Prove your words, train model for 2, 3, 5, 7 and 10 layers.
# 8. Using backpropagation find optimal  digit 8 for your net.*

# In[6]:


def n_layer_net(
    x_train,
    y_train,
    learning_rate,
    layers_dim,
    nonlinearity=tf.sigmoid,
    weights_regularization=None,
    bias_regularization=None,
    batch_size=100,
    epochs_count=50,
    shuffle=False,
    var_initializer=tf.truncated_normal_initializer
):
    with tf.device("/gpu:0"):
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        layers = []
        # make layers
        for i, layer_dim in enumerate(layers_dim):
            if (i == 0):
                layers.append(tf.layers.dense(inputs=x,
                                              units=layer_dim,
                                              activation=nonlinearity,
                                              kernel_initializer=var_initializer,
                                              kernel_regularizer=weights_regularization,
                                              bias_regularizer=bias_regularization))
            elif (i == len(layers_dim) - 1):
                layers.append(tf.layers.dense(inputs=layers[i - 1],
                                              units=layer_dim,
                                              kernel_initializer=var_initializer,
                                              kernel_regularizer=weights_regularization,
                                              bias_regularizer=bias_regularization))
            else:
                layers.append(tf.layers.dense(inputs=layers[i - 1],
                                              units=layer_dim,
                                              activation=nonlinearity,
                                              kernel_initializer=var_initializer,
                                              kernel_regularizer=weights_regularization,
                                              bias_regularizer=bias_regularization))

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=layers[-1]))

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess.run(init)
        
        correct_prediction = tf.equal(tf.argmax(layers[-1],1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        train_acc_trace = []
        val_acc_trace = []

        for i in range(epochs_count):
            if (i != 0 and shuffle):
                perm = np.random.permutation(x_train.shape[0]);
                x_train = x_train[perm,:]
                y_train = y_train[perm,:]
            for cur_batch in batch(x_train, y_train, batch_size):
                train_step.run(feed_dict={x: cur_batch[0], y_: cur_batch[1]})
            train_acc_trace.append(accuracy.eval(feed_dict={x: x_train, y_: y_train}))
            val_acc_trace.append(accuracy.eval(feed_dict={x: x_val, y_: y_val}))

        sess.close()
        
        return train_acc_trace, val_acc_trace


# ##### Ex. 2-3
# Простая двуслойная нейронная сеть

# In[10]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[500, 10],
                                             batch_size=100,
                                             epochs_count=20,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])


# ##### Ex. 4

# In[11]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[500, 10],
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

y11 = train_acc_trace
y12 = val_acc_trace

train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[500, 10],
                                             nonlinearity=tf.nn.relu,
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])
y21 = train_acc_trace
y22 = val_acc_trace


# In[12]:


plt.plot(np.arange(1, 101), y11, label="train")
plt.plot(np.arange(1, 101), y12, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with sigmoid")
plt.legend()
plt.show()

plt.plot(np.arange(1, 101), y21, label="train")
plt.plot(np.arange(1, 101), y22, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with ReLu")
plt.legend()
plt.show()


# ##### Ex. 5-6

# In[60]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.002,
                                             layers_dim=[1000, 10],
                                             batch_size=100,
                                             epochs_count=60,
                                             weights_regularization=tf.contrib.layers.l2_regularizer(0.5),
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])


# Точность ~98% - это лучшая точность, которой мне удалось достичь.

# ### Сравнение разного кол-ва слоев

# ##### три слоя

# In[19]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[1000, 500, 10],
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

y11 = train_acc_trace
y12 = val_acc_trace
print("sigmoid:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[1000, 500, 10],
                                             nonlinearity=tf.nn.relu,
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))


y21 = train_acc_trace
y22 = val_acc_trace
print("ReLu:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

plt.plot(np.arange(1, 101), y11, label="train")
plt.plot(np.arange(1, 101), y12, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with sigmoid")
plt.legend()
plt.show()

plt.plot(np.arange(1, 101), y21, label="train")
plt.plot(np.arange(1, 101), y22, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with ReLu")
plt.legend()
plt.show()


# ##### пять слоев

# In[20]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[1000, 750, 500, 250, 10],
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

y11 = train_acc_trace
y12 = val_acc_trace
print("sigmoid:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.005,
                                             layers_dim=[1000, 750, 500, 250, 10],
                                             nonlinearity=tf.nn.relu,
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))


y21 = train_acc_trace
y22 = val_acc_trace
print("ReLu:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

plt.plot(np.arange(1, 101), y11, label="train")
plt.plot(np.arange(1, 101), y12, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with sigmoid")
plt.legend()
plt.show()

plt.plot(np.arange(1, 101), y21, label="train")
plt.plot(np.arange(1, 101), y22, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with ReLu")
plt.legend()
plt.show()


# ##### 7 слоев

# In[21]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.0005,
                                             layers_dim=[1000, 850, 750, 500, 350, 250, 10],
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

y11 = train_acc_trace
y12 = val_acc_trace
print("sigmoid:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.0005,
                                             layers_dim=[1000, 850, 750, 500, 350, 250, 10],
                                             nonlinearity=tf.nn.relu,
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))


y21 = train_acc_trace
y22 = val_acc_trace
print("ReLu:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

plt.plot(np.arange(1, 101), y11, label="train")
plt.plot(np.arange(1, 101), y12, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with sigmoid")
plt.legend()
plt.show()

plt.plot(np.arange(1, 101), y21, label="train")
plt.plot(np.arange(1, 101), y22, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with ReLu")
plt.legend()
plt.show()


# При большом количестве слоев пришлось на порядок снижать learning rate, т.к. в ином случае сеть просто не учится.

# ##### 10 слоев

# In[22]:


train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.00005,
                                             layers_dim=[1100, 1000, 850, 750, 600, 500, 350, 250, 100, 10],
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))

y11 = train_acc_trace
y12 = val_acc_trace
print("sigmoid:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

train_acc_trace, val_acc_trace = n_layer_net(x_train,
                                             y_train,
                                             learning_rate=0.00005,
                                             layers_dim=[1100, 1000, 850, 750, 600, 500, 350, 250, 100, 10],
                                             nonlinearity=tf.nn.relu,
                                             batch_size=100,
                                             epochs_count=100,
                                             var_initializer=tf.truncated_normal_initializer(stddev=1./784))


y21 = train_acc_trace
y22 = val_acc_trace
print("ReLu:")
print("accuracy on train set: ", train_acc_trace[-1])
print("accuracy on validation set: ", val_acc_trace[-1])

plt.plot(np.arange(1, 101), y11, label="train")
plt.plot(np.arange(1, 101), y12, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with sigmoid")
plt.legend()
plt.show()

plt.plot(np.arange(1, 101), y21, label="train")
plt.plot(np.arange(1, 101), y22, label="validation")
plt.xlabel(r"epochs count")
plt.title(r"accuracy with ReLu")
plt.legend()
plt.show()


# ### 4. Autoencoders
# An autoencoder is an network used for unsupervised learning of efficient codings. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction. Also, this technique can be used to train deep nets.
# 
# Architecturally, the simplest form of an autoencoder is a feedforward net very similar to the multilayer perceptron (MLP), but with the output layer having the same number of nodes as the input layer, and with the purpose of reconstructing its own inputs. Therefore, autoencoders are unsupervised learning models. An autoencoder always consists of two parts, the encoder and the decoder. Encoder returns latent representation of the object (compressed representation, usuallu smaller dimension), but decoder restores object from this latent representation. Autoencoders are also trained to minimise reconstruction errors (e.g. MSE).
# 
# Various techniques exist to prevent autoencoders from learning the identity and to improve their ability to capture important information:
# 1. Denoising autoencoder - take a partially corrupted input.
# 2. Sparse autoencoder - impose sparsity on the hidden units during training (whilst having a larger number of hidden units than inputs).
# 3. Variational autoencoder models inherit autoencoder architecture, but make strong assumptions concerning the distribution of latent variables.
# 4. Contractive autoencoder - add an explicit regularizer in objective function that forces the model to learn a function that is robust to slight variations of input values.
# 
# #### Exercises
# 1. Train 2 layers autoencoder that compressed mnist images to $\mathbb{R}^3$ space.
# 2. For each digit plot several samples in 3D axis (use "%matplotlib notebook" mode or plotly). How do digits group?
# 3. Train autoencoder with more layers. What are results?
# 4. Use autoencoder to pretrain 2 layers (unsupervised) and then train the following layers with supervised method.

# In[24]:


def n_layer_encoder(
    x_train,
    y_train,
    learning_rate,
    layers_dim,
    nonlinearity=tf.sigmoid,
    batch_size=100,
    epochs_count=50,
    shuffle=False,
    var_initializer=tf.truncated_normal_initializer(stddev=1./784)
):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    
    for i, layer_dim in enumerate(layers_dim):
        if (i == 0):
            layers.append(tf.layers.dense(inputs=x,
                                          units=layer_dim,
                                          activation=nonlinearity,
                                          kernel_initializer=var_initializer))
        else:
            layers.append(tf.layers.dense(inputs=layers[i - 1],
                                          units=layer_dim,
                                          activation=nonlinearity,
                                          kernel_initializer=var_initializer))

    mse = tf.losses.mean_squared_error(x, layers[-1])

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(epochs_count):
        if (i != 0 and shuffle):
            perm = np.random.permutation(x_train.shape[0]);
            x_train = x_train[perm,:]
            y_train = y_train[perm,:]
        for cur_batch in batch(x_train, y_train, batch_size):
            train_step.run(feed_dict={x: cur_batch[0]})

    compress = np.array(layers[len(layers_dim) // 2 - 1].eval(feed_dict={x: x_train}))
    
    tf.reset_default_graph()    
    sess.close()

    return compress


# In[48]:


x_compressed = n_layer_encoder(x_train,
                             y_train,
                             learning_rate=0.005,
                             batch_size=100,
                             epochs_count=50,
                             layers_dim = [200, 3, 200, 784])


print(x_compressed.shape)


# In[52]:


color_map = ['red', 'green', 'blue', 'yellow', 'black', 'brown', 'orange', 'pink', 'grey', 'purple']

classes_of_points = [[] for i in range(10)]
for i, label in enumerate(y_train[:]):
    for j in range(10):
        if (label[j] == 1):
            classes_of_points[j].append(x_compressed[i])


# In[57]:


get_ipython().run_line_magic('matplotlib', 'notebook')
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
            
for color in range(10):
    for i, point in enumerate(classes_of_points[color]):
        if (i > 10):
            break
        if i == 0:
            ax.scatter(point[0], point[1], point[2], c=color_map[color], label=str(color))
        else:
            ax.scatter(point[0], point[1], point[2], c=color_map[color])
plt.legend()
plt.show()
        


# Видно, что точки сгруппированы довольно логично. Единица расположена особняком от всех, шестерка и ноль тоже. Остальные хотя и сгрупированы по цветам, но находятся ближе друг к другу. Так как между ними не такая большая разница

# In[55]:


x_compressed = n_layer_encoder(x_train,
                             y_train,
                             learning_rate=0.009,
                             batch_size=100,
                             epochs_count=50,
                             layers_dim = [500, 200, 3, 200, 500, 784])


print(x_compressed.shape)


# In[58]:


get_ipython().run_line_magic('matplotlib', 'notebook')

color_map = ['red', 'green', 'blue', 'yellow', 'black', 'brown', 'orange', 'pink', 'grey', 'purple']
            
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
            
for color in range(10):
    for i, point in enumerate(classes_of_points[color]):
        if (i > 10):
            break
        if i == 0:
            ax.scatter(point[0], point[1], point[2], c=color_map[color], label=str(color))
        else:
            ax.scatter(point[0], point[1], point[2], c=color_map[color])
plt.legend()
plt.show()
        


# После добавления одного слоя общие черты сохранились, но мне кажется, что группы стали чуть удаленнее друг от друга, что хорошо.

# In[67]:


def n_layer_with_encoder(
    x_train,
    y_train,
    learning_rate,
    layers_dim,
    encoder_dim,
    nonlinearity=tf.sigmoid,
    batch_size=100,
    epochs_count=50,
    shuffle=False,
    var_initializer=tf.truncated_normal_initializer(stddev=1./784)
):
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
    enc_layers = []
    for i, dim in enumerate(encoder_dim):
        if (i == 0):
            enc_layers.append(tf.layers.dense(inputs=x,
                                              units=dim,
                                              activation=nonlinearity,
                                              kernel_initializer=var_initializer))
        else:
            enc_layers.append(tf.layers.dense(inputs=enc_layers[i - 1],
                                              units=dim,
                                              activation=nonlinearity,
                                              kernel_initializer=var_initializer))
            
    layers = []
    for i, layer_dim in enumerate(layers_dim):
        if (i == 0):
            layers.append(tf.layers.dense(inputs=enc_layers[1],
                                          units=layer_dim,
                                          activation=nonlinearity,
                                          kernel_initializer=var_initializer))
        elif (i == len(layers_dim) - 1):
            layers.append(tf.layers.dense(inputs=layers[i - 1],
                                          units=layer_dim,
                                          kernel_initializer=var_initializer))
        else:
            layers.append(tf.layers.dense(inputs=layers[i - 1],
                                          units=layer_dim,
                                          activation=nonlinearity,
                                          kernel_initializer=var_initializer))

    mse = tf.losses.mean_squared_error(x, enc_layers[-1])

    enc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=layers[-1]))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(epochs_count):
        if (i != 0 and shuffle):
            perm = np.random.permutation(x_train.shape[0]);
            x_train = x_train[perm,:]
            y_train = y_train[perm,:]
        for cur_batch in batch(x_train, y_train, batch_size):
            enc_train_step.run(feed_dict={x: cur_batch[0]})
            
    for i in range(epochs_count):
        if (i != 0 and shuffle):
            perm = np.random.permutation(x_train.shape[0]);
            x_train = x_train[perm,:]
            y_train = y_train[perm,:]
        for cur_batch in batch(x_train, y_train, batch_size):
            train_step.run(feed_dict={x: cur_batch[0], y_: cur_batch[1]})
        
    correct_prediction = tf.equal(tf.argmax(layers[-1],1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_acc = accuracy.eval(feed_dict={x: x_val, y_: y_val})
    train_acc = accuracy.eval(feed_dict={x: x_train, y_: y_train})
    
    tf.reset_default_graph()    
    sess.close()

    return train_acc, val_acc


# In[75]:


train_acc, val_acc = n_layer_with_encoder(x_train,
                                          y_train,
                                          learning_rate=0.005,
                                          encoder_dim=[200,100,200,784],
                                          layers_dim=[50, 10],
                                          batch_size=100,
                                          epochs_count=20)
print("accuracy on train:", train_acc)
print("accuracy on validation:", val_acc)


# В данной нейронке используется сжатие 784->200->100.
# Затем используется модель с одним скрытым слоем размера 50.
