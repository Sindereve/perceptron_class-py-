import numpy as np 

class Perceptron(object):
    """ the base this class is perceptron

    Parametrs
    ------------------------

    eta : float 
        Speed learn (0.0 < eta <= 1.0)
    
    n_iter : int
        Number of iterations
    
    random_state : int 
        Random start value in w

    Attributs
    ------------------------

    w_ : linear array
        weight
    
    errors_: list
        numbers of error class in each iteration

    """

    #create class 
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
        """ training

        Parametrs
        ------------------------

        X : array, form = [n_examples, n_features]
            n_examles - number of sample
            n_features - number of feutures :) 
        
        y: array, form = [n_examples]
            n_examles - number of sample

        return
        ------------------------
        self : object
        """
        #random input in errors and weight
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ //_// """
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
