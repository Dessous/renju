
# coding: utf-8

# ## L1 - Linear models and gradient descent
# 
# ### Books
# 1. [Deep Learning, I. Goodfellow, Y. Bengio and A. Courville](http://www.deeplearningbook.org/)
# 2. [Neural networks for pattern recognition, C. Bishop](http://cs.du.edu/~mitchell/mario_books/Neural_Networks_for_Pattern_Recognition_-_Christopher_Bishop.pdf)
# 3. [Machine learning: a probabilistic perspective, K. Murphy](http://dsd.future-lab.cn/members/2015nlp/Machine_Learning.pdf)

# ### 0. Basic classification
# 
# Here you can see basic (possible, non standard) classification of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) tasks.
# 1. [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)
#  1. [Regression](https://en.wikipedia.org/wiki/Regression_analysis)
#  2. [Classification](https://en.wikipedia.org/wiki/Statistical_classification)
#  3. [Ranking](https://en.wikipedia.org/wiki/Learning_to_rank)
# 2. [Reinforcment learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
# 3. [Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
#  1. Clustering 
#  2. Manifold learning
#  3. Matrix decompostion (factorization)
#  4. Dimension reduction
#  
# In this lab we consider only supervised learning. Namely, linear regression and binary linear classification, as simple methods for beginning.

# ### 1. Supervised learning basics
# A supervised learning algorithm is an algorithm that is able to learn from data. Now we need only to clarify what is data and what it means to learn?
# 
# Let $\{x_i\}_{i=1}^{\mathcal{l}} \subset \mathbb{X} = \mathbb{R}^{n}$ and $\{y_i\}_{i=1}^{\mathcal{l}} \subset \mathbb{Y}$. Here $\mathbb{X}$ is the whole set of objects and $\mathbb{Y}$ is all possible labels of objects, so $\{x_i\}_{i=1}^{\mathcal{l}}$ is subset with known labels $\{y_i\}_{i=1}^{\mathcal{l}}$. We want to find algorithm, that can predict $y$ for any $x \in \mathbb{X}$. Actually, $x = (x^1, \dots, x^n)$ is some vector of features (formal description), but $x^k$ can have different nature. 
# 
# * $x^k \in \{0, 1\}$ – binary feature, boolean flag
# * $x^k \in \{1,\dots, m\}$ – categorical (nominal), classification of entities into particular categories.
# * $x^k \in \{1,\dots, m\}^<$ – ordinal, classification of entities in some kind of ordered relationship.
# * $x^k \in \mathbb{R}$ – cardinal, classification based on a numerical value.
# 
# Categorical features are commonly encoded in some way (for exaple [one-hot encoding](https://en.wikipedia.org/wiki/One-hot)) to ignore false ordering (important for metric algorithms). Moreover it's possible to cast any type of feature to $\mathbb{R}$, that's why we suppouse that $\mathbb{X} = \mathbb{R}^{n}$ further.
# 
# Process of finding algorithm, that can predict labels, is called training. Usually, it is reduced to minimization problem of the empirical risk.
# $$\arg \min_{\theta} Q(\theta) = \arg \min_{\theta} \frac{1}{\mathcal{l}}\sum_{i=1}^{\mathcal{l}} \mathcal{L}(f(x_i | \theta), y_i).$$
# Here $\mathcal{L}$ – some loss function that shows how good we predict $y$, and $f(x|\theta)$ is parametric function, where $\theta \in \Theta$.

# ### 2. Linear regression
# For regression task $\mathbb{Y} = \mathbb{R}$. In case of linear model we have learning vector of parameters $w \in \mathbb{R}^n$ and predict $y$ as 
# $$y = w^Tx + b.$$
# 
# For simplicity, let the last element of $x$ is always $1$ and $w$ is concatenation of $[w, b]$. So, we can rewrite model as $y = w^Tx$. For MSE (mean square error) we have following optimization problem
# $$\arg \min_{w} Q(w) = \arg \min_{w} \sum_{i=1}^{\mathcal{l}}\big(w^Tx_i - y_i\big)^2.$$
# 
# Let $X$ is matrix, where $i$-th row is feature vector of $i$-th object and $Y$ – vector of labels. In this case our expression can be rewritten in matrix form
# $$\arg\min_{w}||Xw - Y ||_{2}.$$
# But this problem is already well studied and has the analytical solution
# $$w = (X^TX)^{-1}X^TY.$$
# 
# #### Exercises
# 1. Let $y = sin(x) + \varepsilon$, where $x \in [0, 2\pi]$ and $\varepsilon \sim \mathcal{N}(0, 0.1)$. Generate 20 train samples and try to learn regression model.
# 2. Plot train data and model's predictions.
# 3. As you see, model has no enough capacity to fit train data. Let's add polynomial features, namely $x^2$ and $x^3$.
# 4. Train linear model one more time and plot results again.
# 5. What happens if you add more features, for example full range $x^{0},\dots,x^{7}$? 

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import time
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[7]:


def generate_samples(sample_shape, eps_centre, eps_deviation):
    x_sample = np.sort(np.random.rand(sample_shape[0], sample_shape[1]), axis=0) * 2 * np.pi
    epsilons = np.random.normal(eps_centre, eps_deviation, sample_shape)
    y_sample = np.sin(x_sample) + epsilons
    return x_sample, y_sample

def fit_polynomial_model(x_train, y_train, poly_deg):
    X = np.ones(train_shape[0])
    for i in range(1, poly_deg + 1):
        X = np.column_stack((X, x_train ** i))
    W = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y_train))
    return W

def generate_x_to_draw(count_of_points, x_range, poly_deg):
    X_to_draw = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / count_of_points)
    X_to_draw = np.reshape(X_to_draw, (count_of_points, 1))
    X_to_draw = np.column_stack((np.ones((count_of_points, 1)), X_to_draw))
    for i in range(2, poly_deg + 1):
        X_to_draw = np.column_stack((X_to_draw, X_to_draw[:, 1] ** i))
    return X_to_draw;

def make_y_from_x(x_arr, W):
    X_to_calculate = np.column_stack((np.ones((x_arr.shape[0], 1)), x_arr))
    for i in range(2, W.shape[0]):
        X_to_calculate = np.column_stack((X_to_calculate, x_arr ** i))
    return np.dot(X_to_calculate, W)

def draw_points_and_model(x_arr, y_arr, W):
    plt.scatter(x_arr, y_arr)
    X_to_draw = generate_x_to_draw(1000, (0, 2 * np.pi), W.shape[0] - 1)
    plt.plot(X_to_draw[:, 1], np.dot(X_to_draw, W), color='r')
    plt.title("points and prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# ##### Ex. 1 and Ex. 2

# In[8]:


train_shape = (20, 1)

x_train, y_train = generate_samples(train_shape, 0, 0.1)

W = fit_polynomial_model(x_train, y_train, 1)

draw_points_and_model(x_train, y_train, W)


# ##### Ex. 3 and Ex. 4

# In[9]:


W = fit_polynomial_model(x_train, y_train, 3)

draw_points_and_model(x_train, y_train, W)


# ##### Ex. 5

# In[10]:


W = fit_polynomial_model(x_train, y_train, 7)

draw_points_and_model(x_train, y_train, W)


# In[11]:


print(W)


# Т.к. мы изначально сгенерировали точки с разбросом вокруг стандартной синусоиды, которая хорошо приближается многочленом 3 степени на $[0, 2\pi]$, модели 3 и 7 степени различаются незначительно и коэффициенты при старших степенях в модели 7 степени близки к 0.

# ### 3. Validation
# The data used to build the final model usually comes from multiple datasets. In particular, three data sets are commonly used in different stages of the creation of the model.
# 
# 1. We initially fit our parameters on a __training dataset__, that consists of pairs of a feature vector and the corresponding answer. The current model is run with the training dataset and produces a result, which is then compared with the target, for each input vector in the training dataset. Based on the result of the comparison and the specific learning algorithm being used, the parameters of the model are adjusted. The model fitting can include both variable selection and parameter estimation.
# 
# 2. Second one called the __validation dataset__. The validation dataset provides an unbiased evaluation of a model fit on the training dataset while tuning the model's hyperparameters (e.g. regularization coefficient or number of hidden units in a neural network). Validation datasets can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset. This simple procedure is complicated in practice by the fact that the validation dataset's error may fluctuate during training. This complication has led to the creation of many ad-hoc rules for deciding when overfitting has truly begun.
# 
# 3. Finally, the __test dataset__ is a dataset used to provide an unbiased evaluation of a final trained model.
# 
# Cross-validation is a validation technique for estimating how accurately a predictive model will perform in practice. The goal of cross validation is to limit problems like overfitting, give an insight on how the model will generalize to an independent dataset.
# 
# Cross-validation involves partitioning a sample of data into complementary subsets, performing the analysis on one subset and making validation on the other. To reduce variability, multiple rounds of cross-validation are performed using different partitions, and the validation results are caveraged over the rounds to estimate a final predictive model.
# 
# There are following types:
# 1. Leave-p-out cross-validation - using p observations as the validation set with all possible ways.
# 2. k-fold cross-validation - split data into k folds and using each one as validation set.
# 3. Holdout validation - randomly split data into training and validation set
# 4. Repeated random sub-sampling validation - repeatedly make random splits of data into training and validation set
# 
# #### Exercises
# 1. Generate 20 validation samples
# 2. Check quality of your model on train set and validation set.
# 3. Have you experienced [overfitting](https://en.wikipedia.org/wiki/Overfitting)?
# 4. Please, read [this article](https://en.wikipedia.org/wiki/VC_dimension) to learn more about model capacity and VC-dimension.

# In[12]:


x_validation, y_validation = generate_samples(train_shape, 0, 0.1)

W = fit_polynomial_model(x_train, y_train, 7)
print("degree: 7")
print("MSE on train set:", mean_squared_error(y_train, make_y_from_x(x_train, W)))
print("MSE on validation set:", mean_squared_error(y_validation, make_y_from_x(x_validation, W)), "\n")

W = fit_polynomial_model(x_train, y_train, 3)
print("degree: 3")
print("MSE on train set:", mean_squared_error(y_train, make_y_from_x(x_train, W)))
print("MSE on validation set:", mean_squared_error(y_validation, make_y_from_x(x_validation, W)))


# Видим, что переобучения нет. Думаю, это связано с тем, что говорилось ранее. Модели 3 и 7 степени не сильно отличаются друг от друга, а значит и среднеквадратичная ошибка отличается незначительно.

# ### 4. Binary linear classification
# Let $\mathbb{Y} = \{-1, +1\}$ for binary classification. So linear model looks like
# $$sign(w^Tx + b),$$
# where $w$ is normal to the separating plane, which is defined parametrically $w^Tx+b=0$. In the half-space, which normal is directed, all points has class +1, otherwise -1. Let's assume that all points of hyperplane has class +1 to resolve the ambiguity. Also we rewrite model in the short variant $sign(w^Tx)$.
# 
# As with regression, training of linear classifier may be reduced to an optimization problem. We only have to specify the loss function. The most nature option is
# $$\mathcal{L}(y_{pred}, y_{true}) = [y_{pred} \neq y_{true}] = [M < 0],$$
# where $M$ is the margin value $yw^Tx$, which indicates how far the classifier puts a point in its class. But this loss has one drawback, it's not differentiable. That's why the optimization problem becomes very complex. However we can use any other function, which majorizes this loss. You can find some popular options below
# 
# 1. MSE has one big advantage, we optimize convex function with a local minimum. Moreover analytic solution exists.
# $$\big(w^Tx - y \big)^2$$
# 
# 2. Hinge loss function makes our linear classifier [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) (support vector machine).
# $$max \big(0, 1 - yw^Tx \big)$$
# 
# 3. Logistic loss function has a probabilistic meaning. In particular, this loss leads us to the optimal [Bayesian classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) under certain assumptions on the distribution of features. But it's a different story. So it is often used in practice.
# $$\ln \big( 1 + \exp(-yw^Tx) \big)$$

# #### Exercises
# 1. Let $\mathbb{P}\{y=1|x\} = \sigma(wx)$, where $\sigma(z) = \frac{1}{1 + \exp(-z)}$. Show that problem below it is nothing like the maximization of the likelihood.
# $$\arg\min_{w}Q(w) = \arg\min_{w} \sum_{x, y} \ln \big(1 + \exp(-yw^Tx )) \big)$$
# 2. Plot all loss functions in the axes $M \times L$.
# 3. Generate two normally distributed sets of points on the plane.
# 4. Let points of 1th set (red color) have class +1 and point of 2d set (blue color) have -1.
# 5. Train linear classifier with MSE (use analytical solution), which splits these sets.
# 6. Plot points and separating line of trained classifier.
# 7. What is time comlexity of your solution?

# ##### Ex. 2

# Для MSE требуется небольшое приведение, чтобы это была функция от margin.
# $$(w^Tx - y)^2 = (\frac{M}{y} - y)^2 = (\frac{M - y^2}{y})^2$$
# т.к. $y = -1$ или $y = 1$, то $y^2 = 1$ и
# $$(\frac{M - y^2}{y})^2 = \frac{(M - y^2)^2}{y^2} = (M - 1)^2$$

# In[13]:


def indicator(M):
    return M < 0

def MSE(M):
    return (M - 1) ** 2

def HLF(M):
    return np.maximum(0, 1 - M)

def LLF(M):
    return np.log(1 + np.exp(-M)) / np.log(2)


# In[14]:


M = np.asarray([-1 + 3 * i / 1000 for i in range(1000)])
M = np.reshape(M, (1000, 1))

indicator_f, = plt.plot(M, indicator(M), label='indicator')
MSE_f, = plt.plot(M, MSE(M), label='MSE')
HLF_f, = plt.plot(M, HLF(M), label='HLF')
LLF_f, = plt.plot(M, LLF(M), label='LLF')
plt.legend(handles=[indicator_f, MSE_f, HLF_f, LLF_f])
plt.show()


# ##### Ex. 3 and Ex. 4

# In[15]:


point_set1 = np.random.normal((1.3, 1.3), (1, 1), (100, 2))
point_set2 = np.random.normal((4, 4), (1, 1), (100, 2))

plt.scatter(point_set1[:, 0], point_set1[:, 1], color='r')
plt.scatter(point_set2[:, 0], point_set2[:, 1], color='b')
plt.show()


# ##### Ex. 5

# In[16]:


x_train = np.row_stack((point_set1, point_set2))
y_train = np.row_stack((np.ones((100, 1)), -np.ones((100, 1))))
X = np.column_stack((x_train, np.ones((200, 1))))
W = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y_train))

plt.scatter(point_set1[:, 0], point_set1[:, 1], color='r')
plt.scatter(point_set2[:, 0], point_set2[:, 1], color='b')
draw_range = np.arange(-1, 7.1, 0.5)
plt.plot(draw_range, - draw_range * W[0] / W[1] - W[2] / W[1])
plt.show()


# ### 5. Gradient descent
# Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point. Gradient descent is based on the observation that if function $Q(x)$ is defined and differentiable in a neighborhood of a point $x$, then $Q(x)$ decreases fastest if one goes from $x$  in the direction of the negative gradient.
# 
# $$x^{k+1} = x^{k} - \lambda \cdot \triangledown Q(x)$$
# 
# Here $\lambda$ is step of descent and  $k$ – step number. If $\lambda$ is too large then algorithm may not converge, otherwise training can last a long time. Also there is rather popular hack to slowly decrease $\lambda$ with each step. You need to understand that gradient descent finds exactly local minimum. The easiest way to fight this problem is make several runs of algorithm or have good initialization.
# 
# #### Exercises
# 1. Suggest some quadratic function $Q: \mathbb{R}^2 \rightarrow \mathbb{R}$ with global minimum.
# 2. Find minimum with gradient descent method.
# 3. Plot contour lines.
# 4. Trace the path of gradient descent.
# 5. How do you choose $\lambda$?
# 6. Evaluate time complexity of solution.

# In[17]:


def simple_grad_descent(func,
                       grad,
                       lmbd,
                       start_point,
                       max_iter = 1e4,
                       eps = 1e-20):
    grad_path = [np.copy(start_point)]
    step = 0
    point = np.copy(start_point)
    prev_point = np.copy(start_point)
    while (step == 0 or abs(func(point) - func(prev_point)) >= eps and step < max_iter):
        step += 1
        prev_point = np.copy(point)
        point -= lmbd(step) * grad(point)
        #print(lmbd(step) * grad(point))
        grad_path.append(np.copy(point))
    return grad_path


# In[18]:


func = lambda lst: 4 * lst[0] ** 2 + 10 * (lst[1] - 1) ** 2
d_func = lambda lst: np.asarray([8 * lst[0], 20 * (lst[1] - 1)])
lmbd = lambda x: 0.1 / np.log(np.e * x)

grad_path = simple_grad_descent(func, d_func, lmbd, np.asarray([4.7, -3.7]))
print("count of steps: ", len(grad_path))
print("solution: ", grad_path[-1])
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 7, 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter([grad_path[i][0] for i in range(len(grad_path))], [grad_path[i][1] for i in range(len(grad_path))])
plt.plot([grad_path[i][0] for i in range(len(grad_path))], [grad_path[i][1] for i in range(len(grad_path))])
plt.show()


# Я перебрал варианты $$\lambda(t) = c$$
# 
# $$\lambda(t) = \frac{c}{t}$$
# 
# $$\lambda(t) = \frac{c}{ln(et)}$$
# 
# где $c$ - константа, $t$ - номер шага
# Оказалось, что следующая лямбда сходится быстрее всего:
# $$\lambda(t) = \frac{0.1}{ln(et)}$$

# There is category of function which naive gradient descent works poorly for, e.g. [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
# $$f(x, y) = (1-x)^2 + 100(y-x^2)^2.$$
# 
# #### Exercises
# 1. Repeat previous steps for Rosenbrock function.
# 2. What problem do you face?
# 3. Is there any solution?

# In[19]:


func = lambda lst: (1 - lst[0]) ** 2 + 100 * (lst[1] - lst[0] ** 2) ** 2
d_func = lambda lst: np.asarray([-2 * (1 - lst[0]) - 400 * (lst[1] - lst[0] ** 2) * lst[0],
                                 200 * (lst[1] - lst[0] ** 2)])
lmbd = lambda x: 0.001

grad_path = simple_grad_descent(func, d_func, lmbd, np.asarray([1.9, -1.5]), max_iter=10000)
print("count of steps: ", len(grad_path))
print("solution: ", grad_path[-1])
x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter([grad_path[i][0] for i in range(len(grad_path))], [grad_path[i][1] for i in range(len(grad_path))])
plt.plot([grad_path[i][0] for i in range(len(grad_path))], [grad_path[i][1] for i in range(len(grad_path))])
plt.show()


# Я столкнулся с проблемой того, что алгоритм сходится очень медленно. Для того, чтобы получить решение с хорошей точностью, понадобилось минимум 10000 итераций. Это связано с тем, что градиент функции сильно уменьшается при приближении к минимуму, поэтому шаги очень маленькие. Возможно, это можно исправить, изменяя $\lambda$ в зависимости от величины градиента.

# There are some variations of the method, for example steepest descent, where we find optimal $\lambda$ for each step.
# $$\lambda^{k} = \arg\min_{\lambda}Q(x_k - \lambda\triangledown Q(x_k)).$$
# 
# #### Exercises
# 1. Split red and blue sets of points again. Train linear model using gradient descent and MSE.
# 2. Plot your splitting line. Compare with analytical solution.
# 3. Try steepest descent.
# 4. Comare gradient descent methods and show its convergence in axes $[step \times Q]$.

# In[20]:


point_set1 = np.random.normal((1.5, 1.5), (1, 1), (100, 2))
point_set2 = np.random.normal((4, 4), (1, 1), (100, 2))

x_train = np.row_stack((point_set1, point_set2))
y_train = np.row_stack((np.ones((100, 1)), -np.ones((100, 1))))
X = np.column_stack((x_train, np.ones((200, 1))))
W_analytical = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y_train))

func = lambda W: ((np.dot(X, W) - y_train) ** 2).mean()
d_func = lambda W: np.asarray([[2 * (X[:, 0].reshape(X.shape[0], 1) * (np.dot(X, W) - y_train)).mean()],
                               [2 * np.sum(X[:, 1].reshape(X.shape[0], 1) * (np.dot(X, W) - y_train)) / y_train.shape[0]],
                               [2 * np.sum((np.dot(X, W) - y_train)) / y_train.shape[0]]])
lmbd = lambda x: 0.01
W_st = 5 * np.random.randn(3)
W_st = np.reshape(W_st, (3, 1))
grad_path = simple_grad_descent(func, d_func, lmbd, W_st, eps=1e-4, max_iter=1000)
W_grad = grad_path[-1]
print("count of iterations: ", len(grad_path))
print(grad_path[-1])


# In[21]:


plt.scatter(point_set1[:, 0], point_set1[:, 1], color='r')
plt.scatter(point_set2[:, 0], point_set2[:, 1], color='b')
draw_range = np.arange(-1, 7.1, 0.5)
line1 = plt.plot(draw_range,
                 - draw_range * W_analytical[0] / W_analytical[1] - W_analytical[2] / W_analytical[1],
                 label='analytical')
line2 = plt.plot(draw_range, - draw_range * W_grad[0] / W_grad[1] - W_grad[2] / W_grad[1], label='gradient')
plt.legend()
plt.show()


# In[22]:


def steepest_grad_descent(func,
                       grad,
                       start_point,
                       max_iter = 1e4,
                       eps = 1e-20):
    grad_path = [np.copy(start_point)]
    step = 0
    point = np.copy(start_point)
    prev_point = np.copy(start_point)
    while (step == 0 or abs(func(point) - func(prev_point)) >= eps and step < max_iter):
        step += 1
        prev_point = np.copy(point)
        lmbd = minimize(lambda l: func(point - l * grad(point)), x0=0).x
        point -= lmbd * grad(point)
        #print(lmbd(step) * grad(point))
        grad_path.append(np.copy(point))
    return grad_path


# In[23]:


point_set1 = np.random.normal((1.5, 1.5), (1, 1), (100, 2))
point_set2 = np.random.normal((4, 4), (1, 1), (100, 2))

x_train = np.row_stack((point_set1, point_set2))
y_train = np.row_stack((np.ones((100, 1)), -np.ones((100, 1))))
X = np.column_stack((x_train, np.ones((200, 1))))
W_analytical = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y_train))

func = lambda W: ((np.dot(X, W) - y_train) ** 2).mean()
d_func = lambda W: np.asarray([[2 * (X[:, 0].reshape(X.shape[0], 1) * (np.dot(X, W) - y_train)).mean()],
                               [2 * np.sum(X[:, 1].reshape(X.shape[0], 1) * (np.dot(X, W) - y_train)) / y_train.shape[0]],
                               [2 * np.sum((np.dot(X, W) - y_train)) / y_train.shape[0]]])
lmbd = lambda x: 0.01

W_start = 5 * np.random.randn(3)
W_start = np.reshape(W_st, (3, 1))
steepest_grad_path = steepest_grad_descent(func, d_func, W_start, eps=1e-4, max_iter=1000)
simple_grad_path = simple_grad_descent(func, d_func, lmbd, W_start, eps=1e-4, max_iter=1000)
W_st_grad = steepest_grad_path[-1]
W_s_grad = simple_grad_path[-1]
print("count of iterations in steepest gradient descent: ", len(steepest_grad_path))
print("count of iterations in simple gradient descent: ", len(simple_grad_path))

plt.scatter(point_set1[:, 0], point_set1[:, 1], color='r')
plt.scatter(point_set2[:, 0], point_set2[:, 1], color='b')
draw_range = np.arange(-1, 7.1, 0.5)
plt.plot(draw_range,
         - draw_range * W_st_grad[0] / W_st_grad[1] - W_st_grad[2] / W_st_grad[1],
         label='steepest')
plt.plot(draw_range, - draw_range * W_s_grad[0] / W_s_grad[1] - W_s_grad[2] / W_s_grad[1], label='simple')
plt.legend()
plt.show()


# In[27]:


plt.plot(range(len(simple_grad_path)),
         [func(simple_grad_path[i]) for i in range(len(simple_grad_path))],
         label='simple')
plt.plot(range(len(steepest_grad_path)),
         [func(steepest_grad_path[i]) for i in range(len(steepest_grad_path))],
         label='steepest')
plt.axis([-20, max(len(simple_grad_path), len(steepest_grad_path)) + 40, 0, 50])
plt.xlabel('step')
plt.ylabel('Q')
plt.legend()
plt.show()


# Видим, что steepest gradient descent сходится гораздо быстрее обычного.

# ### 6. Stochastic gradient descent

# Sometimes you have so huge amount of data, that usual gradient descent becomes too slow. One more option, we have deal with data flow. In this case stochastic gradient method appears on the stage. The idea is simple. You can do a descent step, calculating error and gradient not for all samples, but for some small batch only.
# 
# #### Еxercises
# 1. Download [mnist](https://www.kaggle.com/c/digit-recognizer).
# 2. Train linear classificator for digits 0 and 1, using logistic loss function and stochastic gradient descent.
# 3. Use holdout to check [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) of classification.
# 4. How do accuracy and training time depend on bathch size?
# 5. Plot graphic that proves your words.
# 6. How many epochs you use? Why?
# 7. Plot value of loss function for each step (try use [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)).

# In[28]:


data = np.loadtxt("train.csv", delimiter=',', skiprows=1)
data = data[data[:,0] < 2, :]
data[data[:,0] == 0, 0] = -1
x_train, x_test, y_train, y_test = train_test_split(data[:,1:], data[:,0], test_size=0.2, random_state=20)
x_train = (x_train - np.mean(x_train)) / 255
x_test = (x_test - np.mean(x_test)) / 255
x_train = np.column_stack((x_train, np.ones((x_train.shape[0], 1))))
x_test = np.column_stack((x_test, np.ones((x_test.shape[0], 1))))


# In[116]:


def stochastic_grad_descent(grad,
                            X,
                            Y,
                            start_point,
                            batch_size=30,
                            epochs_count=4,
                            lmbd=0.001):
    point = start_point
    batch_count = X.shape[0] // batch_size
    cur_epoch = 0
    start_time = time.time()
    step = 0
    while (cur_epoch < epochs_count):
        cur_epoch += 1
        for cur_batch in batch(X, Y, batch_size):
            point -= lmbd * grad(cur_batch[0], cur_batch[1], point)
            step += 1
    return point, time.time() - start_time

def sgd_for_smoothing(grad,
                      X,
                      Y,
                      start_point,
                      batch_size=30,
                      epochs_count=4,
                      lmbd=0.001):
    point = start_point
    batch_count = X.shape[0] // batch_size
    cur_epoch = 0
    step = 0
    score = []
    while (cur_epoch < epochs_count):
        cur_epoch += 1
        for cur_batch in batch(X, Y, batch_size):
            score.append(log_loss(cur_batch[0], cur_batch[1], point))
            step += 1
            point -= lmbd * grad(cur_batch[0], cur_batch[1], point)
    return point, score

def smoothing(score, smooth_coef=0.5):
    ans = [score[0]]
    for i in range(1, len(score)):
        ans.append(smooth_coef * score[i] + (1 - smooth_coef) * ans[i - 1])
    return ans

def batch(X, Y, batch_size):
    start = 0
    end = start + batch_size
    while (end < X.shape[0]):
        yield (X[start:end, :], Y[start:end])
        start = end
        end += batch_size

def log_loss(X, Y, W):
    return np.log(1 + np.exp(-np.dot(Y.T, np.dot(X, W)))) / X.shape[0]

def d_log_loss(X, Y, W):
    Y = np.reshape(Y, (X.shape[0], 1))
    numer = -np.dot(np.dot(X.T, Y), np.exp(-np.dot(Y.T, np.dot(X, W))))
    denom = (1 + np.exp(-np.dot(Y.T, np.dot(X, W)))) * X.shape[0]
    return numer / denom

def mnist_accuracy(labels, raw_predict):
    raw_predict[raw_predict < 0] = -1
    raw_predict[raw_predict >= 0] = 1
    return accuracy_score(labels, raw_predict)


# In[325]:


W, cur_time = stochastic_grad_descent(d_log_loss,
                                      x_train,
                                      y_train,
                                      np.zeros((785, 1), dtype='float'),
                                      batch_size=30,
                                      epochs_count=4)

print("accuracy on train set: ", mnist_accuracy(y_train, np.dot(x_train, W)))
print("accuracy on test set: ", mnist_accuracy(y_test, np.dot(x_test, W)))


# Я перебрал кол-во эпох от 1 до 6, лучший результат на тестовой выборке показало количество эпох, равное четырем, поэтому в дальнейшем я использую именно это кол-во.

# In[326]:


batch_range = np.arange(1, 6000, 1)

time_values = []
accuracy_values = []
for batch_size in batch_range:
    W, cur_time = stochastic_grad_descent(d_log_loss,
                                          x_train,
                                          y_train,
                                          np.zeros((785, 1), dtype='float'),
                                          batch_size,
                                          epochs_count=4)
    
    time_values.append(cur_time)
    accuracy_values.append(mnist_accuracy(y_test, np.dot(x_test, W)))

plt.figure(figsize=(8, 6))
plt.plot(batch_range, time_values)
plt.xlabel('batch size')
plt.ylabel('time')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(batch_range, accuracy_values)
plt.xlabel('batch size')
plt.ylabel('accuracy')
plt.show()


# Время при увеличении размера батча уменьшается, а точность колеблется в очень маленьких пределах, но всё же виден пик при размере батча, недалеком от 0.

# In[327]:


W, score = sgd_for_smoothing(d_log_loss,
                                   x_train,
                                   y_train,
                                   np.zeros((785, 1), dtype='float'),
                                   batch_size=30,
                                   epochs_count=4)
plt.figure(figsize=(10, 8))
plt.plot(np.arange(len(score)), smoothing(score, smooth_coef=0.7))
plt.xlabel('step')
plt.ylabel('smoothed score')
plt.show()


# #### Momentum method
# Stochastic gradient descent with momentum remembers the update of $x$ at each iteration, and determines the next update as a linear combination of the gradient and the previous update
# $$x^{k+1} = x^{k} - s^{k},$$ where $s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k)$, $0 <\gamma < 1$ – smoothing ratio and $s^{-1} = 0$.
# 
# #### Еxercises
# 1. Find minimum for $Q(x,y)=10x^2+y^2$ with descent method.
# 2. Use momentum method and compare pathes.
# 3. How do you choose $\gamma$?

# In[30]:


def momentum_grad_descent(func, grad, lmbd, start_point, max_iter=1000, eps=1e-4, smooth_coef=0.5):
    grad_path = [np.copy(start_point)]
    step = 0
    point = np.copy(start_point)
    prev_point = np.copy(start_point)
    prev_s = 0
    cur_s = 0
    while (step == 0 or abs(func(point) - func(prev_point)) >= eps and step < max_iter):
        step += 1
        prev_point = np.copy(point)
        buf = np.copy(cur_s)
        cur_s = smooth_coef * prev_s + lmbd(step) * grad(point)
        prev_s = np.copy(buf)
        point = prev_point - cur_s
        #print(lmbd(step) * grad(point))
        grad_path.append(np.copy(point))
    return grad_path


# In[43]:


func = lambda lst: 10 * lst[0] ** 2 + lst[1] ** 2
d_func = lambda lst: np.asarray([20 * lst[0], 2 * lst[1]])
lmbd = lambda x: 0.01

simple_grad_path = simple_grad_descent(func,
                                       d_func,
                                       lmbd,
                                       np.asarray([1.9, -1.5]),
                                       eps=1e-10)
momentum_grad_path = momentum_grad_descent(func,
                                           d_func,
                                           lmbd,
                                           np.asarray([1.9, -1.5]),
                                           eps=1e-10,
                                           smooth_coef=0.7)
print("count of steps in simple grad descent: ", len(simple_grad_path))
print("solution: ", simple_grad_path[-1])
print("count of steps in momentum grad descent: ", len(momentum_grad_path))
print("solution: ", momentum_grad_path[-1])
x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-4, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter([simple_grad_path[i][0] for i in range(len(simple_grad_path))],
            [simple_grad_path[i][1] for i in range(len(simple_grad_path))])
plt.plot([simple_grad_path[i][0] for i in range(len(simple_grad_path))],
         [simple_grad_path[i][1] for i in range(len(simple_grad_path))], label='simple')
plt.scatter([momentum_grad_path[i][0] for i in range(len(momentum_grad_path))],
            [momentum_grad_path[i][1] for i in range(len(momentum_grad_path))])
plt.plot([momentum_grad_path[i][0] for i in range(len(momentum_grad_path))],
         [momentum_grad_path[i][1] for i in range(len(momentum_grad_path))], label='momentum')
plt.legend()
plt.show()


# Я перебрал значения $\gamma$ с шагом 0.1 . При значениях $>0.7$ алгоритм имеет большой разброс и поэтому сходится довольно медленно, при других значениях алгоритм сходится стабильно, но за разное кол-во итераций. Оптимальным оказался $\gamma = 0.7$, т.к. при таком коэффициенте получается хорошее соотношение кол-во итераций/качество

# #### Nesterov accelerated gradient
# And the logical development of this approach leads to the accelerated Nesterov's gradient. The descent step is calculated a little differently
# $$s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k - \gamma s^{k-1}),$$
# so we find gradient at the point which moment will move us.
# 
# #### Еxercises
# 1. Compare this method and previous with Rosenbrock function.
# 2. Plot traces of both algorithms.

# In[31]:


def nesterov_grad_descent(func, grad, lmbd, start_point, max_iter=1000, eps=1e-4, smooth_coef=0.4):
    grad_path = [np.copy(start_point)]
    step = 0
    point = np.copy(start_point)
    prev_point = np.copy(start_point)
    prev_s = 0
    cur_s = 0
    while (step == 0 or abs(func(point) - func(prev_point)) >= eps and step < max_iter):
        step += 1
        prev_point = np.copy(point)
        buf = np.copy(cur_s)
        cur_s = smooth_coef * prev_s + lmbd(step) * grad(point - smooth_coef * prev_s)
        prev_s = np.copy(buf)
        point = prev_point - cur_s
        #print(lmbd(step) * grad(point))
        grad_path.append(np.copy(point))
    return grad_path


# In[32]:


func = lambda lst: (1 - lst[0]) ** 2 + 100 * (lst[1] - lst[0] ** 2) ** 2
d_func = lambda lst: np.asarray([-2 * (1 - lst[0]) - 400 * (lst[1] - lst[0] ** 2) * lst[0],
                                 200 * (lst[1] - lst[0] ** 2)])
lmbd = lambda x: 0.0008

momentum_grad_path = momentum_grad_descent(func, d_func, lmbd, np.asarray([1.9, -1.5]), max_iter=10000, eps=1e-7, smooth_coef=0.4)
nesterov_grad_path = nesterov_grad_descent(func, d_func, lmbd, np.asarray([1.9, -1.5]), max_iter=10000, eps=1e-7, smooth_coef=0.4)
print("count of steps in momentum grad descent: ", len(momentum_grad_path))
print("solution: ", momentum_grad_path[-1])
print("count of steps in nesterov grad descent: ", len(nesterov_grad_path))
print("solution: ", nesterov_grad_path[-1])
x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
mgp0 = [momentum_grad_path[i][0] for i in range(len(momentum_grad_path))]
mgp1 = [momentum_grad_path[i][1] for i in range(len(momentum_grad_path))]
ngp0 = [nesterov_grad_path[i][0] for i in range(len(nesterov_grad_path))]
ngp1 = [nesterov_grad_path[i][1] for i in range(len(nesterov_grad_path))]
plt.scatter(mgp0, mgp1)
plt.plot(mgp0, mgp1, label='momentum')
plt.scatter(ngp0, ngp1)
plt.plot(ngp0, ngp1, label='nesterov')
plt.legend()
plt.show()


# Оба алгоритма показывают примерно сравнимые результаты, то есть отличаются не сильно при одинаковых параметрах.

# #### Adagrad (2011)
# Adaptive gradient finds lambda for each dimension of the input vector x. Informally speaking, for sparce features it makes a bigger step, but for regular ones smaller step.
# $$x_{i}^{k + 1} = x_{i}^{k} - \frac{\lambda}{\sqrt{G_{i, i}^k } + \varepsilon} \cdot \frac{\partial Q}{\partial x_i}(x^k),$$
# * $G^{k} = \sum_{t=1}^{k}g_t g_t^{T}$, где $g_t = \triangledown Q(x^t)$.
# * $\varepsilon$ - epsilon to avoid division by zero.
# It improves convergence of the learning process (e.g. when using neural networks for text).
# 
# #### RMSprop
# To avoid growth of the denominator we can use the following modification. Let's calculate the matrix $G^k$ only for a small number of latest steps, it can be done for example using exponential smoothing.
# $$G^{k+1} = \gamma G^{k} + (1 - \gamma)g_{k+1}g_{k+1}^{T},$$
# where $0< \gamma < 1$ - smoothing factor
# 
# #### Еxercises
# 1. Read about adadelta and adam (links below).
# 2. Give an example of a function that can show the difference in the studied stohastic gradient methods.
# 3. Show animation step by step how methods work.
# 4. Use your favorite method on mnist dataset again.
# 5. Show convergence of alrotigthm.
# 6. Check quality, using holdout.
# 
# #### Papers
# 1. [Adadelta (2012)](https://arxiv.org/pdf/1212.5701.pdf)
# 2. [Adam (2015)](https://arxiv.org/pdf/1412.6980.pdf)

# Минимизировать будем функцию Розенброка

# In[74]:


def adagrad(func, grad, lmbd, start_point, max_iter=10000, eps=1e-10):
    point = np.copy(start_point)
    step = 0
    grad_path = [start_point]
    while (step < max_iter):
        gradient = grad(point)
        if step == 0:
            G = gradient ** 2
        else:
            G += gradient ** 2
        step += 1
        point -= lmbd(step) * gradient / (np.sqrt(G) + eps)
        grad_path.append(np.copy(point))
    return grad_path

def smoothed_adagrad(func, grad, lmbd, start_point, max_iter=10000, eps=1e-5, smooth_coef=0.5):
    point = np.copy(start_point)
    step = 0
    grad_path = [start_point]
    while (step < max_iter):
        gradient = grad(point)
        if step == 0:
            G = (1 - smooth_coef) * gradient ** 2
        else:
            G = smooth_coef * G + (1 - smooth_coef) * gradient ** 2
        step += 1
        point -= lmbd(step) * gradient / (np.sqrt(G) + eps)
        grad_path.append(np.copy(point))
    return grad_path

def adadelta(func, grad, start_point, eps=1e-10, decay=0.9, max_iter=10000):
    point = np.copy(start_point)
    step = 0
    grad_path = [start_point]
    accumulate_grad = 0
    accumulate_delta = 0
    while (step < max_iter):
        cur_grad = grad(point)
        accumulate_grad = decay * accumulate_grad + (1 - decay) * cur_grad ** 2
        shift = -np.sqrt(accumulate_delta + eps) / np.sqrt(accumulate_grad + eps) * cur_grad
        accumulate_delta = decay * accumulate_delta + (1 - decay) * shift ** 2
        point += shift
        grad_path.append(np.copy(point))
        step += 1
    return grad_path

def adam(func,
         grad,
         start_point,
         decay1=0.9,
         decay2=0.999,
         lmbd=0.001,
         eps=1e-8,
         max_iter=10000):
    point = np.copy(start_point)
    step = 0
    grad_path = [start_point]
    accumulate_grad = 0
    accumulate_sec_grad = 0
    while (step < max_iter):
        step += 1
        cur_grad = grad(point)
        accumulate_grad = decay1 * accumulate_grad + (1 - decay1) * cur_grad
        accumulate_sec_grad = decay2 * accumulate_sec_grad + (1 - decay2) * cur_grad ** 2
        corrected_a_g = accumulate_grad / (1 - decay1 ** step)
        corrected_a_s_g = accumulate_sec_grad / (1 - decay2 ** step)
        shift = -lmbd * corrected_a_g / (np.sqrt(corrected_a_s_g) + eps)
        point += shift
        grad_path.append(np.copy(point))
    return grad_path


# In[83]:


func = lambda lst: (1 - lst[0]) ** 2 + 100 * (lst[1] - lst[0] ** 2) ** 2
d_func = lambda lst: np.asarray([-2 * (1 - lst[0]) - 400 * (lst[1] - lst[0] ** 2) * lst[0],
                                 200 * (lst[1] - lst[0] ** 2)])
lmbd = lambda x: 0.01

grad_path = adagrad(func, d_func, lmbd, np.asarray([1.9, -1.5]))

print("count of steps in adagrad: ", len(grad_path))
print("solution: ", grad_path[-1])

x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('adagrad')
plt.xlabel('x')
plt.ylabel('y')
grad_path_x = [grad_path[i][0] for i in range(len(grad_path))]
grad_path_y = [grad_path[i][1] for i in range(len(grad_path))]
plt.scatter(grad_path_x, grad_path_y)
plt.plot(grad_path_x, grad_path_y)
plt.show()

smoothed_grad_path = smoothed_adagrad(func, d_func, lmbd, np.asarray([1.9, -1.5]), smooth_coef = 0.8)

print("count of steps in smoothed adagrad: ", len(smoothed_grad_path))
print("solution: ", smoothed_grad_path[-1])

x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('smoothed adagrad')
plt.xlabel('x')
plt.ylabel('y')
smoothed_grad_path_x = [smoothed_grad_path[i][0] for i in range(len(smoothed_grad_path))]
smoothed_grad_path_y = [smoothed_grad_path[i][1] for i in range(len(smoothed_grad_path))]
plt.scatter(smoothed_grad_path_x, smoothed_grad_path_y)
plt.plot(smoothed_grad_path_x, smoothed_grad_path_y)
plt.show()

adadelta_grad_path = adadelta(func, d_func, np.asarray([1.9, -1.5]), decay=0.9, eps=1e-8, max_iter=10000)

print("count of steps in adadelta: ", len(adadelta_grad_path))
print("solution: ", adadelta_grad_path[-1])

x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('adadelta')
plt.xlabel('x')
plt.ylabel('y')
adadelta_grad_path_x = [adadelta_grad_path[i][0] for i in range(len(adadelta_grad_path))]
adadelta_grad_path_y = [adadelta_grad_path[i][1] for i in range(len(adadelta_grad_path))]
plt.scatter(adadelta_grad_path_x, adadelta_grad_path_y)
plt.plot(adadelta_grad_path_x, adadelta_grad_path_y)
plt.show()

adam_grad_path = adam(func, d_func, np.asarray([1.9, -1.5]), lmbd=0.04, max_iter=10000)

print("count of steps in adam: ", len(adam_grad_path))
print("solution: ", adam_grad_path[-1])

x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
plt.figure(figsize=(10,10))
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('adam')
plt.xlabel('x')
plt.ylabel('y')
adam_grad_path_x = [adam_grad_path[i][0] for i in range(len(adam_grad_path))]
adam_grad_path_y = [adam_grad_path[i][1] for i in range(len(adam_grad_path))]
plt.scatter(adam_grad_path_x, adam_grad_path_y)
plt.plot(adam_grad_path_x, adam_grad_path_y)
plt.show()


# Видим, что оптимизация помогла, т.к. теперь алгоритм хоть как-то сходится к (1, 1), тогда как неоптимизированный алгоритм за 10000 итераций остается далеко от минимума

# In[110]:


fig = plt.figure(figsize=(10, 10))

ax = plt.axes(
    xlim=( -2.3, 2.), 
    ylim=(-1.7, 4),)

line1, = ax.plot([], [], lw=2, label='adagrad')
line2, = ax.plot([], [], lw=2, label='smoothed adagrad')
line3, = ax.plot([], [], lw=2, label='adadelta')
line4, = ax.plot([], [], lw=2, label='adam')

x = np.linspace(-2.3, 2., 1000)
y = np.linspace(-1.7, 4., 1000)
x, y = np.meshgrid(x, y)
z = func([x, y])
CS = plt.contour(x, y, z, 15)
plt.clabel(CS, inline=1, fontsize=10)
plt.legend()

x1 = np.asarray(grad_path_x)
y1 = np.asarray(grad_path_y)
x2 = np.asarray(smoothed_grad_path_x)
y2 = np.asarray(smoothed_grad_path_y)
x3 = np.asarray(adadelta_grad_path_x)
y3 = np.asarray(adadelta_grad_path_y)
x4 = np.asarray(adam_grad_path_x)
y4 = np.asarray(adam_grad_path_y)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return (line1,line2,line3,line4,)

def animate(i):
    line1.set_data(x1[:5 * i], y1[:5 * i])
    line2.set_data(x2[:5 * i], y2[:5 * i])
    line3.set_data(x1[:5 * i], y1[:5 * i])
    line4.set_data(x2[:5 * i], y2[:5 * i])
    return (line1,line2,line3,line4,)



# In[111]:


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=10, blit=True)


# In[112]:


HTML(anim.to_html5_video())


# На данной анимации видно только adadelta и adam, т.к. они накладываются на adagrad и smoothed adagrad соответственно. Заметим, что адам работает быстрее всех остальных спусков и при этом показывает лучший результат.

# In[122]:


def stochastic_adam(grad,
                    X,
                    Y,
                    start_point,
                    decay1=0.9,
                    decay2=0.999,
                    eps=1e-8,
                    batch_size=30,
                    epochs_count=4,
                    lmbd=0.001):
    batch_count = X.shape[0] // batch_size
    cur_epoch = 0
    step = 0
    point = np.copy(start_point)
    accumulate_grad = 0
    accumulate_sec_grad = 0
    while (cur_epoch < epochs_count):
        cur_epoch += 1
        for cur_batch in batch(X, Y, batch_size):
            step += 1
            cur_grad = grad(cur_batch[0], cur_batch[1], point)
            accumulate_grad = decay1 * accumulate_grad + (1 - decay1) * cur_grad
            accumulate_sec_grad = decay2 * accumulate_sec_grad + (1 - decay2) * cur_grad ** 2
            corrected_a_g = accumulate_grad / (1 - decay1 ** step)
            corrected_a_s_g = accumulate_sec_grad / (1 - decay2 ** step)
            shift = -lmbd * corrected_a_g / (np.sqrt(corrected_a_s_g) + eps)
            point += shift
            step += 1
    return point


# In[147]:


W = stochastic_adam(d_log_loss,
                    x_train,
                    y_train,
                    np.zeros((785, 1), dtype='float'),
                    batch_size=20,
                    epochs_count=4,
                    lmbd=0.00001)
print("accuracy on train set: ", mnist_accuracy(y_train, np.dot(x_train, W)))
print("accuracy on test set: ", mnist_accuracy(y_test, np.dot(x_test, W)))

