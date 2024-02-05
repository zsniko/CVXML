import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time 

class SVM:
    
    def __init__(self, X_train, Y_train):
        '''
        Initialization of the SVM classifier. Define variables and constraints.
        Arg  X_train, Y_train: training data.
        '''
        # Define variables
        self.a = cp.Variable(X_train.shape[1]) # 2-dimensional data
        self.b = cp.Variable()                 # scalar
        self.u = cp.Variable(X_train.shape[0]) # slack variable
        self.v = cp.Variable(Y_train.shape[0]) # slack variable

        # Define constraints
        self.constraints = [
            self.a @ X_train.T - self.b >= 1 - self.u,
            self.a @ Y_train.T - self.b <= -1 + self.v,
            self.u >= 0,
            self.v >= 0
        ]
    
    def fit(self, gamma=0.1):
        '''
        Function to train the SVM classifier based on training sets.
        Arg  gamma: hyperparameter which controls the trade-off between 
                    the number of misclassified points and the width of the slab.
        '''
        self.gamma = gamma
        # Define the objective function
        objective = cp.Minimize(cp.norm(self.a) + gamma * (cp.sum(self.u) + cp.sum(self.v)))

        # Define and solve the convex optimization problem
        problem = cp.Problem(objective, self.constraints)
        problem.solve()

    def plot_svm(self, set_1, set_2, colorX='tab:blue', colorY='tab:red', mode='Training', shade=True, deactivate_plot=False, slab=True, data_pts=True):
        '''
        Function to plot the data points along with the trained SVM classifier with slab.
        Arg   set_1, set_2: The two sets of points to plot (training or test data).
              colorX, colorY: set the colors of the two sets to plot.
              mode: define whether it is training or testing data (for figure title).    
        '''

        # Plot the separating hyperplane
        # Define the range of x values
        X_set = np.concatenate((set_1, set_2))
        plt.xlim([X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5])
        plt.ylim([X_set[:, 1].min() - 1, X_set[:, 1].max() + 1])
        x = np.linspace(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5)
        y = (-self.a.value[0] * x + self.b.value) / self.a.value[1]
        plt.plot(x, y, color='blueviolet', linewidth=2.5, linestyle='-', label='f')
        
        if slab:
            y_upper = -(self.a.value[0]*x - (self.b.value + 1))/self.a.value[1]
            y_lower = -(self.a.value[0]*x - (self.b.value - 1))/self.a.value[1]
            plt.plot(x, y_upper, color='black', linestyle='--')
            plt.plot(x, y_lower, color='black', linestyle='--')
        
        if shade:
            # Use two different colors to shade the area to which the two classes belong.
            # Generate grid of points and Set axis range
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            Z = self.a.value[0] * xx + self.a.value[1] * yy - self.b.value
            Z[Z<=0] = -1; Z[Z>0] = 1
            from matplotlib.colors import ListedColormap
            light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])
            plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)
        
        if slab: # N.B.: this is only for exercise 8
            # Plot the slab.
            plt.fill_between(x, y_upper, y_lower, color='gray', alpha=0.5, label='slab')

        if data_pts: # N.B.: this is only for exercise 8
            # Plot the data points.
            plt.scatter(set_1[:, 0], set_1[:, 1], color=colorX, label='X (+1)')
            plt.scatter(set_2[:, 0], set_2[:, 1], color=colorY, label='Y ( - 1)')

        # Add axis labels and legend
        plt.title('Classification results on '+mode+' data: gamma='+str(self.gamma), fontsize=15)
        plt.xlabel('x1', fontsize=15)
        plt.ylabel('x2', fontsize=15)
        plt.legend(loc='best', fontsize=12)
        if not deactivate_plot:
            plt.show()
    
    def predict(self, x):
        '''
        Function to predict which class a single data point x belongs to
        Arg     x: the data point to classify
        Output  the class predicted (+1 or -1)
        '''
        return np.sign(np.dot(self.a.value, x.T) - self.b.value).astype(int)
    
    def predict_net(self, x):
        return (np.dot(self.a.value, x.T) - self.b.value).astype(int)

    def predict_all(self, X_test, Y_test):
        '''
        Function to make all predictions based on test data 
        (for performance evaluation purposes, not for predicting the class of a point).
        Arg     X_test, Y_test: the two sets of test points to evaluate.
        Output  pred, true: predicted labels and true labels of the test data.
        '''
        pred_X = np.sign(np.dot(self.a.value, X_test.T) - self.b.value).astype(int)
        pred_Y = np.sign(np.dot(self.a.value, Y_test.T) - self.b.value).astype(int)
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        pred = np.concatenate((pred_X, pred_Y))
        true = np.concatenate((true_X, true_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        
        return pred, true
    
    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        Arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        # for i in range(len(true)):
        #     result[true[i]][pred[i]] += 1 

        # Use numpy instead of for loop to accelerate code
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat
    
    def tune(self, lower_b=0.01, upper_b=5, pts=50, X_test=None, Y_test=None):
        '''
        Function to perform hyperparameter tuning based on the specified test set and the accuracy metric.
        Arg     lower_b: the minimum gamma value to test
                higher_b: the maximum gamma value to test
                pts: the number of points in between
                X_test: test set, class +1
                Y_test: test set, class -1
        Output  train_time: hyperparameter tuning time
                best_gamma: the first gamma that gives the highest validation accuracy
                best_acc: the highest test accuracy obtained
                gamma_range[best_gamma_index]: the best gammas in the specified range
        '''
        # Define the values for tuning.
        gamma_range = np.linspace(lower_b, upper_b, pts)
        # Initialize variables to keep track of the best gamma and the highest accuracy.
        best_gamma = 0
        best_acc = 0
        best_gamma_index = []

        # Train the SVM on the training set by varying gamma
        start = time.time()
        for i, g in enumerate(gamma_range):
            # Train
            self.fit(gamma=g)
            # Evaluate the accuracy metric of the SVM model on the test set.
            pred_vals, true_vals = self.predict_all(X_test, Y_test)
            confmat = self.compute_confusion_matrix(true=true_vals, pred=pred_vals, show=False)
            acc = confmat.trace() / confmat.sum()
            # Keep track of the best results and update the best gamma and accuracy.
            # N.B. Don't use one single >= condition otherwise the first gamma will always be appended even if it doesn't achieve highest acc.
            if acc > best_acc:
                best_gamma = g   
                best_acc = acc 
                best_gamma_index = [i]
            elif acc == best_acc:
                best_gamma_index.append(i)
        end = time.time()
        tuning_time = end - start 
        return tuning_time, best_gamma, round(best_acc*100,2), gamma_range[best_gamma_index] 
    
class model_evaluator:
    
    def __init__(self, confusion_matrix):
        # Initialize with the calculated confusion matrix. 
        self.confmat = confusion_matrix

    def display_confusion_matrix(self, mode='test', classifier_name='SVM'):
        '''
        Function to show confusion matrix using only the matplotlib library.
            TN FP
            FN TP
        Arg:  mode: str, uses different color for displaying the confusion matrix on training / test set.
        '''
        # Display the confusion matrix. Use different colors for test / training set.
        plt.imshow(self.confmat, cmap=plt.cm.Blues) if mode=='test' else plt.imshow(self.confmat, cmap='OrRd')
        plt.colorbar()
        # Add labels to the plot.
        plt.xticks([0,1], ['-1', '+1'])
        plt.yticks([0,1], ['-1', '+1'])
        plt.title('Confusion Matrix of '+classifier_name+' Classifier', fontsize=15)
        plt.xlabel('Predicted class', fontsize=15)
        plt.ylabel('True class', fontsize=15)
        plt.grid(False)
        # Set the threshold value for font color to enhance visibility.
        threshold = (self.confmat.max() + self.confmat.min())/2
        # Add text to the cells
        for i in range(self.confmat.shape[0]):
            for j in range(self.confmat.shape[1]):
                # choose the right text color according to the background color.
                color = 'white' if self.confmat[i,j] > threshold else 'black'
                plt.text(j, i, self.confmat[i,j], horizontalalignment='center', verticalalignment='center', color=color, fontsize=15)
        # Show the plot
        plt.show()
    
    def eval(self, metric='accuracy', display=True):
        """
        Evaluate the performance of a classifier based on a confusion matrix and a given metric.
        Args:
            confmat: 2x2 array, the confusion matrix of the classifier
            mode: str, the metric to use, can be 'accuracy', 'precision', 'recall', or 'f1'
            display: bool, defines whether we display the results
        Output:
            The value of the selected metric.
        """
        # detect if the user entered a valid metric.
        if metric not in {'accuracy', 'precision', 'recall', 'f1'}:
            raise ValueError('Unknown metric: {}'.format(metric))
        # flatten the matrix into an array.
        tn, fp, fn, tp = self.confmat.ravel()
        # compute the result based on the selected metric.
        if metric == 'accuracy':
            result = (tp + tn) / (tp + fp + fn + tn)
        elif metric == 'precision':
            result = tp / (tp + fp)
        elif metric == 'recall':
            result = tp / (tp + fn)
        elif metric == 'f1':
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            result = 2 * precision * recall / (precision + recall)
        if display:
            print('{}: {:.2f}%'.format(metric.capitalize(), result * 100))
            
        return result
    
class logistic_modeling:
    
    def __init__(self):
        # Define variables.
        self.a = cp.Variable(2)
        self.b = cp.Variable()

    def fit(self, X1_train, X2_train):
        '''
        Function to train the classifier.
        Arg     X1_train: training data belonging to class +1 
                X2_train: training data belonging to class -1 
        '''
        # Define the negative log-likelihood function.
        objective = cp.sum(-self.a @ X1_train.T + self.b) + cp.sum(cp.logistic(self.a @ X1_train.T - self.b)) \
        + cp.sum(cp.logistic((self.a @ X2_train.T) - self.b))
        # Define and solve the convex optimization problem.
        problem = cp.Problem(objective=cp.Minimize(objective), constraints=None)
        problem.solve()
    
    def predict(self, x):
        '''
        Function to predict which class a single data point x belongs to
        Arg     x: the data point to classify
        Output  the class predicted (+1 or -1)
        '''
        return np.sign(np.dot(self.a.value, x.T) - self.b.value).astype(int)
    
    def predict_all(self, X_test, Y_test):
        pred_X = np.sign(np.dot(self.a.value, X_test.T) - self.b.value).astype(int)
        pred_Y = np.sign(np.dot(self.a.value, Y_test.T) - self.b.value).astype(int)
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        pred = np.concatenate((pred_X, pred_Y))
        true = np.concatenate((true_X, true_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        return pred, true
    
    def plot_logistic(self, set_1, set_2, colorX='tab:blue', colorY='tab:red', mode='Training', shade=True):
        '''
        Function to plot the data points along with the trained SVM classifier with slab.
        Arg   set_1, set_2: The two sets of points to plot (training or test data).
              colorX, colorY: set the colors of the two sets to plot.
              mode: define whether it is training or testing data (for figure title).    
        '''
        # Define the range of x values
        X_set = np.concatenate((set_1, set_2))
        plt.xlim([X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5])
        plt.ylim([X_set[:, 1].min() - 1, X_set[:, 1].max() + 1])
        x = np.linspace(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5)

        # Compute the hyperplane with slab
        y = (-self.a.value[0]/self.a.value[1]) * x + (self.b.value/self.a.value[1])
        y_up = y + 1/self.a.value[1]
        y_down = y - 1/self.a.value[1]
        # Plot the separation plane
        plt.plot(x, y, color='indigo', label='f')
        plt.plot(x, y_up, color='k', linestyle='--', linewidth=0.5)
        plt.plot(x, y_down, color='k', linestyle='--', linewidth=0.5)
        
        if shade:
            # Use two different colors to shade the area to which the two classes belong.
            # Generate grid of points and Set axis range
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            Z = self.a.value[0] * xx + self.a.value[1] * yy - self.b.value
            from matplotlib.colors import ListedColormap
            light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])
            Z[Z<=0] = -1; Z[Z>0] = 1
            plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)

        # Plot the slab.
        plt.fill_between(x, y_up, y_down, color='gray', alpha=0.5)
        
        # Plot the data points.
        plt.scatter(set_1[:, 0], set_1[:, 1], color=colorX, label='X (+1)')
        plt.scatter(set_2[:, 0], set_2[:, 1], color=colorY, label='Y ( - 1)')

        # Add axis labels and legend
        plt.title('Approximate linear discrimination via logistic modeling: '+mode+' set', fontsize=12)
        plt.xlabel('x1', fontsize=15)
        plt.ylabel('x2', fontsize=15)
        plt.legend(loc='best', fontsize=12)
        plt.show()

    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat
    
class quad_discrim():

    def __init__(self, X_train, Y_train, slack_var=3.62):
        self.X_train = X_train
        self.Y_train = Y_train
        self.slack_var = slack_var
        # Define variables and parameters
        self.P = cp.Variable((X_train.shape[1],X_train.shape[1]), symmetric=True)
        self.q = cp.Variable((X_train.shape[1]))
        self.r = cp.Variable()
    
    def fit(self):
        # Define objective function and constraints
        obj = cp.Minimize(0)
        constraints = [cp.diag(self.X_train @ self.P @ self.X_train.T) + self.q @ self.X_train.T + self.r >= 1 - self.slack_var,
                    cp.diag(self.Y_train @ self.P @ self.Y_train.T) + self.q @ self.Y_train.T + self.r <= -1 + self.slack_var,
                    self.P <= - np.eye(2)
                    ]
        # Solve problem
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return self.P.value, self.q.value, self.r.value
    
    def plot_quad(self, set_1, set_2, colorX='tab:blue', colorY='tab:red', mode='Training'):
        from matplotlib.colors import ListedColormap
        light_cmap = ListedColormap(['#F5E5C9','#A2DCDE'])
        X_set = np.concatenate((set_1, set_2))
        x_lin = np.linspace(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1)
        y_lin = np.linspace(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1)
        xx, yy = np.meshgrid(x_lin, y_lin)
        X = np.vstack((xx.ravel(), yy.ravel())).T
        # Compute the value of f(x) for each point on the meshgrid
        f = -(np.diag(X @ self.P.value @ X.T) + self.q.value @ X.T + self.r.value)
        
        # Plot the points and the decision boundary
        plt.contour(xx, yy, f.reshape(xx.shape), levels=[0], colors='blueviolet')
        plt.contourf(xx, yy, f.reshape(xx.shape), levels=[-50,0,50], cmap=light_cmap)
        plt.scatter(set_1[:,0], set_1[:,1], color=colorX, label='X (+1)')
        plt.scatter(set_2[:,0], set_2[:,1], color=colorY, label='Y ( - 1)')
        plt.title('Quadratic discrimination on '+mode+' data', fontsize=15)
        plt.xlabel('x1', fontsize=15)
        plt.ylabel('x2', fontsize=15)
        plt.legend(loc='best', fontsize=12)
        plt.show()
    
    def predict(self, x):
        return -np.sign(x @ self.P.value @ x.T + self.q.value @ x.T + self.r.value)
    
    def predict_all(self, X_test, Y_test):
        # Take opposite sign for matching our convention of class +1 and -1.
        pred_X = -np.sign(np.diag(X_test @ self.P.value @ X_test.T) + self.q.value @ X_test.T + self.r.value).astype(int)
        pred_Y = -np.sign(np.diag(Y_test @ self.P.value @ Y_test.T) + self.q.value @ Y_test.T + self.r.value).astype(int)
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        true = np.concatenate((true_X, true_Y))
        pred = np.concatenate((pred_X, pred_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        return pred, true

    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat

class poly_discrim():

    def __init__(self, X_train, Y_train, order):

        # First transpose the data matrices to make data samples as columns.
        self.X_train = np.transpose(X_train)
        self.Y_train = np.transpose(Y_train)

        self.order = order
        if self.order not in {2, 3, 4, 5}:
            raise ValueError('Unsupported polynomial order!')
        
        N = self.X_train.shape[1] # N is the number of data samples in class +1 
        M = self.Y_train.shape[1] # M is the number of data samples in class -1

        # Construct Vandermonde-style monomial matrices
        col_dict = {2:6, 3:10, 4:15, 5:21}
        col = col_dict.get(self.order, None) # retrive the corresponding value if self.order is valid.
        if self.order == 5:
            p1 = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]).reshape(col, 1)
            p2 = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]).reshape(col, 1) - p1
        elif self.order == 4:
            p1 = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]).reshape(col, 1)
            p2 = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]).reshape(col, 1) - p1
        elif self.order == 3:
            p1 = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]).reshape(col, 1)
            p2 = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]).reshape(col, 1) - p1
        elif self.order == 2:
            p1 = np.array([0, 0, 1, 0, 1, 2]).reshape(col, 1)
            p2 = np.array([0, 1, 1, 2, 2, 2]).reshape(col, 1) - p1

        npoly = len(p1)
        op = np.ones((npoly, 1))
        # Construct the data matrix after transformation
        Xr0 = np.repeat(self.X_train[0,:].reshape(1, N), repeats=col, axis=0)
        Xr1 = np.repeat(self.X_train[1,:].reshape(1, N), repeats=col, axis=0)
        p1xr = np.repeat(p1, repeats=N, axis=1)
        p2xr = np.repeat(p2, repeats=N, axis=1)
        self.monX = np.multiply(np.power(Xr0, p1xr), np.power(Xr1, p2xr))
        Yr0 = np.repeat(self.Y_train[0,:].reshape(1, M), repeats=col, axis=0)
        Yr1 = np.repeat(self.Y_train[1,:].reshape(1, M), repeats=col, axis=0)
        p1yr = np.repeat(p1, repeats=M, axis=1)
        p2yr = np.repeat(p2, repeats=M, axis=1)
        self.monY = np.multiply(np.power(Yr0, p1yr), np.power(Yr1, p2yr))

        # Define optimization variables.
        self.a = cp.Variable((npoly,1))
        self.t = cp.Variable()

    def fit(self, reg=0):
        '''
        Function to train the classifier.
        Arg:    reg: regularization term.
        '''
        # Define constraints.
        constr = [self.a.T @ self.monX <= self.t,
                  self.a.T @ self.monY >= -self.t,
                  cp.norm(self.a) <= 1
                  ]
        # Define the optimization problem.
        prob = cp.Problem(objective=cp.Minimize(self.t + reg*cp.norm(self.a)), constraints=constr)
        prob.solve()
        return prob.status
    
    def fit_2(self, reg=0):
        # Define constraints.
        constr = [self.a.T @ self.monX <= reg*self.t,
                  self.a.T @ self.monY >= -reg*self.t,
                  cp.norm(self.a) <= 1
                  ]
        # Define the optimization problem.
        prob = cp.Problem(objective=cp.Minimize(0), constraints=constr)
        prob.solve()
    
    def predict(self, x):
        '''
        Function to predict which class a single data point x belongs to
        (Instead of writing the function again, use the plot function to simply code.)
        Arg     x: the data point to classify
        Output  the class predicted (+1 or -1)
        '''
        return np.sign(-self.get_decision_boundary(x1=x[0], x2=x[1]))
    
    def predict_net(self, x):
        return -self.get_decision_boundary(x1=x[0], x2=x[1])
    
    def predict_all(self, X_test, Y_test):
        '''
        Function to predict all test points using the trained classifier.
        Arg:    X_test: test set which contains class +1
                Y_test: test set which contains class -1
        Output: pred: predicted values using test set
                true: the true labels of test set
        '''
        # Take opposite sign for matching our convention of class +1 and -1.
        pred_X = np.sign(-self.get_decision_boundary(x1=X_test[:, 0], x2=X_test[:, 1])).astype(int)
        pred_Y = np.sign(-self.get_decision_boundary(x1=Y_test[:, 0], x2=Y_test[:, 1])).astype(int)
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        true = np.concatenate((true_X, true_Y))
        pred = np.concatenate((pred_X, pred_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        return pred, true
    
    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat
    
    def get_decision_boundary(self, x1, x2):
        '''
        Function to fetch the decision boundary given data point
        Arg:    x1, x2: components of a data point x
        Output: class decision of the point.
        '''
        if self.order == 4: # DEGREE 4
            return (self.a.value[0] + self.a.value[1]*x2 + self.a.value[2]*x1 + self.a.value[3]*x2**2 + self.a.value[4]*x1*x2 + self.a.value[5]*x1**2 
                    + self.a.value[6]*x2**3 + self.a.value[7]*x2**2*x1 + self.a.value[8]*x2*x1**2 + self.a.value[9]*x1**3 
                    + self.a.value[10]*x2**4 + self.a.value[11]*x2**3*x1 + self.a.value[12]*x1**2*x2**2 + self.a.value[13]*x2*x1**3 + self.a.value[14]*x1**4)
        elif self.order == 3: # DEGREE 3
            return (self.a.value[0] + self.a.value[1]*x2 + self.a.value[2]*x1 + self.a.value[3]*x2**2 + self.a.value[4]*x1*x2 + self.a.value[5]*x1**2 
            + self.a.value[6]*x2**3 + self.a.value[7]*x2**2*x1 + self.a.value[8]*x2*x1**2 + self.a.value[9]*x1**3)
        elif self.order == 2: # DEGREE 2 
            return (self.a.value[0] + self.a.value[1]*x2 + self.a.value[2]*x1 + self.a.value[3]*x2**2 + self.a.value[4]*x1*x2 + self.a.value[5]*x1**2)
        elif self.order == 5: # DEGREE 5
            return (self.a.value[0] + self.a.value[1]*x2 + self.a.value[2]*x1 + self.a.value[3]*x2**2 + self.a.value[4]*x1*x2 + self.a.value[5]*x1**2 
            + self.a.value[6]*x2**3 + self.a.value[7]*x2**2*x1 + self.a.value[8]*x2*x1**2 + self.a.value[9]*x1**3 
            + self.a.value[10]*x2**4 + self.a.value[11]*x2**3*x1 + self.a.value[12]*x2**2*x1**2 + self.a.value[13]*x2*x1**3 + self.a.value[14]*x1**4 
            + self.a.value[15]*x2**5 + self.a.value[16]*x2**4*x1 + self.a.value[17]*x2**3*x1**2 + self.a.value[18]*x2**2*x1**3 + self.a.value[19]*x2*x1**4 + self.a.value[20]*x1**5)

    def plot_decision_boundary(self, X, Y, color='training'):
        '''
        Function to plot the data points along with the trained SVM classifier with slab.
        Arg   X, Y: The two sets of points to plot (training or test data).
              color: Differentiate colors for training and test points (also used for figure title).    
        '''
        from matplotlib.colors import ListedColormap
        light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])
        # Generate a grid of points to plot the decision boundary
        X_concat = np.concatenate((X, Y), axis=1)
        x1_min, x1_max = X_concat[0, :].min() - 0.5, X_concat[0, :].max() + 0.5
        x2_min, x2_max = X_concat[1, :].min() - 0.5, X_concat[1, :].max() + 0.5
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 2000), np.linspace(x2_min, x2_max, 2000))
        zz = -self.get_decision_boundary(x1=xx1.ravel(), x2=xx2.ravel())
        zz = zz.reshape(xx1.shape)
        # Plot the decision boundary
        if color not in {'training', 'Training', 'test', 'Test'}:
            raise ValueError('Unknown color: {}'.format(color))
        if color == 'training':
            plt.contour(xx1, xx2, zz, levels=[0], colors='tab:olive')
            plt.contourf(xx1, xx2, zz, levels=[-100,0,100], cmap=light_cmap)
            plt.scatter(X[0, :], X[1, :], color='tab:blue', label='X (+1)')
            plt.scatter(Y[0, :], Y[1, :], color='tab:red', label='Y ( - 1)')
        elif color == 'test':
            plt.contour(xx1, xx2, zz, levels=[0], colors='tab:pink')
            plt.contourf(xx1, xx2, zz, levels=[-100,0,100], cmap=light_cmap)
            plt.scatter(X[0, :], X[1, :], color='tab:green', label='X (+1)')
            plt.scatter(Y[0, :], Y[1, :], color='tab:orange', label='Y ( - 1)')
        plt.title('Polynomial discrimination order '+str(self.order)+' on '+color+' data', fontsize=11)
        plt.xlabel('x1', fontsize=13)
        plt.ylabel('x2', fontsize=13)
        plt.legend(loc='best', fontsize=12)
        plt.show()
        plt.show()

# Define the nonlinear transformation phi
def phi(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])

class logistic_classifier:
    
    def __init__(self, X, Y, nonlinear=False):
        
        self.nonlinear = nonlinear

        if nonlinear:
            # Define a nonlinear transformation on data.
            def phi(x):
                return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])
            self.X_train = np.apply_along_axis(phi, axis=1, arr=X)
            self.Y_train = np.apply_along_axis(phi, axis=1, arr=Y)
            self.a = cp.Variable(self.X_train.shape[1]) # Match the dimension, in this case 3-dimensional.
        else:
            self.X_train = X
            self.Y_train = Y
            self.a = cp.Variable(2) # We know that the original data is in 2D.
        self.b = cp.Variable() # Scalar


    def fit(self):
        '''
        Function to train the logistic classifier.
        No arguments: Classifier already intialzied. After initialization, call this function to start training. 
        '''
        # Define the negative log-likelihood function.
        objective = cp.sum(-self.a @ self.X_train.T + self.b) + cp.sum(cp.logistic(self.a @ self.X_train.T - self.b)) \
        + cp.sum(cp.logistic((self.a @ self.Y_train.T) - self.b))
        # Define and solve the convex optimization problem.
        problem = cp.Problem(objective=cp.Minimize(objective), constraints=None)
        problem.solve()
    
    def predict(self, x):
        '''
        Function to predict which class a single data point x belongs to
        Arg     x: the data point to classify
        Output  the class predicted (+1 or -1)
        '''
        return np.sign(np.dot(self.a.value, x.T) - self.b.value).astype(int)
    
    def predict_all(self, X_test, Y_test):
        '''
        Function to predict all test points using the trained classifier.
        Arg:    X_test: test set which contains class +1
                Y_test: test set which contains class -1
        Output: pred: predicted values using test set
                true: the true labels of test set
        '''

        # Make sure to transform also the test set into the feature space !
        if self.nonlinear:
            X_test = np.apply_along_axis(phi, axis=1, arr=X_test)
            Y_test = np.apply_along_axis(phi, axis=1, arr=Y_test)

        pred_X = np.sign(np.dot(self.a.value, X_test.T) - self.b.value).astype(int)
        pred_Y = np.sign(np.dot(self.a.value, Y_test.T) - self.b.value).astype(int)
      
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        pred = np.concatenate((pred_X, pred_Y))
        true = np.concatenate((true_X, true_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        return pred, true
    
    def plot_logistic(self, set_1, set_2, colorX='tab:blue', colorY='tab:red', mode='Training', shade=True):
        '''
        Function to plot the data points along with the trained SVM classifier with slab.
        Arg   set_1, set_2: The two sets of points to plot (training or test data).
              colorX, colorY: set the colors of the two sets to plot.
              mode: define whether it is training or test data (just for figure title).    
        '''

        # Define axis range
        X_set = np.concatenate((set_1, set_2))
        plt.xlim([X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5])
        plt.ylim([X_set[:, 1].min() - 1, X_set[:, 1].max() + 1])
        x = np.linspace(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5)

        # If we did not apply nonlinear transformation,
        if not self.nonlinear:
            # Compute the hyperplane with slab.
            y = (-self.a.value[0]/self.a.value[1]) * x + (self.b.value/self.a.value[1])
            y_up = y + 1/self.a.value[1]
            y_down = y - 1/self.a.value[1]
            # Plot the separation plane.
            plt.plot(x, y, color='indigo', label='f')
            plt.plot(x, y_up, color='k', linestyle='--', linewidth=0.5)
            plt.plot(x, y_down, color='k', linestyle='--', linewidth=0.5)

        # If we applied the nonlinear transformation,
        else: 
            # Compute and plot the separating hyperplane in the original data space.
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            Z = self.a.value[0]*xx**2 + self.a.value[1]*np.sqrt(2)*xx*yy + self.a.value[2]*yy**2 - self.b.value
            contour = plt.contour(xx, yy, Z, levels=[0])
            h1,_ = contour.legend_elements()
            legend_f = plt.legend([h1[0], ],['f'], loc='upper left', fontsize=12)
        
        # Just to give the two classes a color to enhance visualization. Has nothing to do with the algorithm.
        if shade:
            from matplotlib.colors import ListedColormap
            light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])
            # Use two different colors to shade the area to which the two classes belong.
            # Generate grid of points and Set axis range
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            if self.nonlinear:
                Z = self.a.value[0]*xx**2 + self.a.value[1]*np.sqrt(2)*xx*yy + self.a.value[2]*yy**2 - self.b.value
                Z[Z<=0] = -1; Z[Z>0] = 1
                plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)
            else: 
                Z = self.a.value[0] * xx + self.a.value[1] * yy - self.b.value
                Z[Z<=0] = -1; Z[Z>0] = 1
                plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)
                # Plot the slab.
                plt.fill_between(x, y_up, y_down, color='gray', alpha=0.5)  
        
        # Plot the data points.
        plt.scatter(set_1[:, 0], set_1[:, 1], color=colorX, label='X (+1)')
        plt.scatter(set_2[:, 0], set_2[:, 1], color=colorY, label='Y ( - 1)')

        # Add axis labels and legend
        plt.title('Logistic classifier: '+mode+' set', fontsize=12)
        plt.xlabel('x1', fontsize=15)
        plt.ylabel('x2', fontsize=15)
        plt.legend(loc='best', fontsize=12)
        # plt.contour does not support the argument 'label'! add the legend separately
        if self.nonlinear:
            plt.gca().add_artist(legend_f)
        plt.show()

    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat

class SVM_v2:
    
    def __init__(self, X_train, Y_train, nonlinear=False):
        '''
        Initialization of the SVM classifier. Define variables and constraints.
        Arg  X_train, Y_train: training data.
             nonlinear: choose whether activating nonlinear transformation on data
        '''
        self.nonlinear = nonlinear # save as instance variable for other functions e.g. plot_svm().

        if self.nonlinear:
            # It is up to us to define the nonlinear transformation here.
            def phi(x):
                return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])
            self.X_train = np.apply_along_axis(phi, axis=1, arr=X_train)
            self.Y_train = np.apply_along_axis(phi, axis=1, arr=Y_train)
        else:
            self.X_train = X_train 
            self.Y_train = Y_train 

        # Define variables
        self.a = cp.Variable(self.X_train.shape[1]) # 2-dimensional data
        self.b = cp.Variable()                      # scalar
        self.u = cp.Variable(self.X_train.shape[0]) # slack variable
        self.v = cp.Variable(self.Y_train.shape[0]) # slack variable
            

        # Define constraints
        self.constraints = [
            self.a @ self.X_train.T - self.b >= 1 - self.u,
            self.a @ self.Y_train.T - self.b <= -1 + self.v,
            self.u >= 0,
            self.v >= 0
        ]
    
    def fit(self, gamma=0.1):
        '''
        Function to train the SVM classifier based on training sets.
        Arg  gamma: hyperparameter which controls the trade-off between 
                    the number of misclassified points and the width of the slab.
        '''
        self.gamma = gamma
        # Define the objective function
        objective = cp.Minimize(cp.norm(self.a) + gamma * (cp.sum(self.u) + cp.sum(self.v)))

        # Define and solve the convex optimization problem
        problem = cp.Problem(objective, self.constraints)
        problem.solve()
    
    def tune(self, lower_b=0.01, upper_b=5, pts=50, X_test=None, Y_test=None):
        '''
        Function to perform hyperparameter tuning based on the specified test set and the accuracy metric.
        Arg     lower_b: the minimum gamma value to test
                higher_b: the maximum gamma value to test
                pts: the number of points in between
                X_test: test set, class +1
                Y_test: test set, class -1
        Output  train_time: hyperparameter tuning time
                best_gamma: the first gamma that gives the highest validation accuracy
                best_acc: the highest test accuracy obtained
                gamma_range[best_gamma_index]: the best gammas in the specified range
        '''
        # Define the values for tuning.
        gamma_range = np.linspace(lower_b, upper_b, pts)
        # Initialize variables to keep track of the best gamma and the highest accuracy.
        best_gamma = 0
        best_acc = 0
        best_gamma_index = []

        # Train the SVM on the training set by varying gamma
        start = time.time()
        for i, g in enumerate(gamma_range):
            # Train
            self.fit(gamma=g)
            # Evaluate the accuracy metric of the SVM model on the test set.
            pred_vals, true_vals = self.predict_all(X_test, Y_test)
            confmat = self.compute_confusion_matrix(true=true_vals, pred=pred_vals, show=False)
            evaluator_svm = model_evaluator(confusion_matrix=confmat)
            acc = evaluator_svm.eval('accuracy', display=False)
            # Keep track of the best results and update the best gamma and accuracy.
            # N.B. Don't use one single >= condition otherwise the first gamma will always be appended even if it doesn't achieve highest acc.
            if acc > best_acc:
                best_gamma = g   
                best_acc = acc 
                best_gamma_index = [i]
            elif acc == best_acc:
                best_gamma_index.append(i)
        end = time.time()
        tuning_time = end - start 
        return tuning_time, best_gamma, round(best_acc*100,2), gamma_range[best_gamma_index] 

    def plot_svm(self, set_1, set_2, colorX='tab:blue', colorY='tab:red', mode='Training', shade=True, disp_slab=True):
        '''
        Function to plot the data points along with the trained SVM classifier with slab.
        Arg   set_1, set_2: The two sets of points to plot (training or test data).
              colorX, colorY: set the colors of the two sets to plot.
              mode: define whether it is training or testing data (for figure title).    
        '''

        # Define the range of x values
        X_set = np.concatenate((set_1, set_2))
        plt.xlim([X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5])
        plt.ylim([X_set[:, 1].min() - 1, X_set[:, 1].max() + 1])
        x = np.linspace(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5)

        if not self.nonlinear:
            y = (-self.a.value[0] * x + self.b.value) / self.a.value[1]
            y_upper = -(self.a.value[0]*x - (self.b.value + 1))/self.a.value[1]
            y_lower = -(self.a.value[0]*x - (self.b.value - 1))/self.a.value[1]
            plt.plot(x, y, color='blueviolet', linewidth=2.5, linestyle='-', label='f')
            plt.plot(x, y_upper, color='black', linestyle='--')
            plt.plot(x, y_lower, color='black', linestyle='--')
        else: 
            # Compute and plot the separating hyperplane in the original data space.
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            Z = self.a.value[0]*xx**2 + self.a.value[1]*np.sqrt(2)*xx*yy + self.a.value[2]*yy**2 - self.b.value
            contour = plt.contour(xx, yy, Z, levels=[0], colors='blueviolet')
            if disp_slab:
                contour_lower = plt.contour(xx, yy, Z, levels=[-1], colors='slategrey', linestyles='dashed')
                contour_upper = plt.contour(xx, yy, Z, levels=[1], colors='slategrey', linestyles='dashed')

            h1,_ = contour.legend_elements()
            legend_f = plt.legend([h1[0], ],['f'], loc='upper left', fontsize=12)

        # Just to give the two classes a color to enhance visualization. Has nothing to do with the algorithm.
        if shade:
            from matplotlib.colors import ListedColormap
            light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])
            # Use two different colors to shade the area to which the two classes belong.
            # Generate grid of points and Set axis range
            h = 0.01
            xx, yy = np.meshgrid(np.arange(X_set[:, 0].min() - 0.5, X_set[:, 0].max() + 0.5, h), np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, h))
            if self.nonlinear:
                Z = self.a.value[0]*xx**2 + self.a.value[1]*np.sqrt(2)*xx*yy + self.a.value[2]*yy**2 - self.b.value
                contour_slab = Z.copy()
                Z[Z<=0] = -1; Z[Z>0] = 1
                plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)
                # Plot the slab.
                if disp_slab:
                    plt.contourf(xx, yy, contour_slab,levels=[-1,1], alpha=0.35, colors='slategray')

            else: 
                Z = self.a.value[0] * xx + self.a.value[1] * yy - self.b.value
                Z[Z<=0] = -1; Z[Z>0] = 1
                plt.contourf(xx, yy, Z, levels=[-1,0,1], cmap=light_cmap)
                # Plot the slab.
                if disp_slab:
                    plt.fill_between(x, y_upper, y_lower, color='gray', alpha=0.5, label='slab')  

        # Plot the data points.
        plt.scatter(set_1[:, 0], set_1[:, 1], color=colorX, label='X (+1)')
        plt.scatter(set_2[:, 0], set_2[:, 1], color=colorY, label='Y ( - 1)')

        # Add axis labels and legend
        plt.title('Classification results on '+mode+' data: gamma='+str(self.gamma), fontsize=15)
        plt.xlabel('x1', fontsize=15)
        plt.ylabel('x2', fontsize=15)
        plt.legend(loc='best', fontsize=12)
        # plt.contour does not support the argument 'label'! add the legend separately
        if self.nonlinear:
            plt.gca().add_artist(legend_f)
        plt.show()
    
    def predict(self, x):
        '''
        Function to predict which class a single data point x belongs to
        Arg     x: the data point to classify
        Output  the class predicted (+1 or -1)
        '''
        return np.sign( np.dot(self.a.value, x.T) - self.b.value).astype(int)

    def predict_all(self, X_test, Y_test):
        '''
        Function to make all predictions based on test data 
        (for performance evaluation purposes, not for predicting the class of a point).
        Arg     X_test, Y_test: the two sets of test points to evaluate.
        Output  pred, true: predicted labels and true labels of the test data.
        '''
        # Make sure to transform also the test set into the feature space !
        if self.nonlinear:
            X_test = np.apply_along_axis(phi, axis=1, arr=X_test)
            Y_test = np.apply_along_axis(phi, axis=1, arr=Y_test)
            
        pred_X = np.sign(np.dot(self.a.value, X_test.T) - self.b.value).astype(int)
        pred_Y = np.sign(np.dot(self.a.value, Y_test.T) - self.b.value).astype(int)
        true_X = np.ones(X_test.shape[0]).astype(int)
        true_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two sets
        pred = np.concatenate((pred_X, pred_Y))
        true = np.concatenate((true_X, true_Y))
        # Convert '-1' to '0' for computing confusion matrix in a later stage
        pred[pred == -1] = 0
        true[true == -1] = 0
        
        return pred, true
    
    def compute_confusion_matrix(self, true, pred, show=True):
        '''
        Function to calcualte the confusion matrix based on predictions and true labels of test data.
        Arg     true, pred: test data's true and predicted labels.
        output  the confusion matrix.
        '''
        num_cls = len(np.unique(true))
        result = np.zeros((num_cls,num_cls))
        # Use numpy instead of for loop to accelerate code
        np.add.at(result, (true, pred), 1)
        confmat = result.astype('int')
        if show:
            print('The Confusion Matrix: \n', confmat)
            diagonal_sum = confmat.trace()
            sum_of_all_elements = confmat.sum()
            accuracy = diagonal_sum / sum_of_all_elements
            print('Accuracy = {:0.2f}%'.format(accuracy*100))
        return confmat

class SVM_v3:
    def __init__(self, X_train, Y_train, X_test, Y_test, C=10, features=2, sigma_sq=0.1, kernel=None, degree=3, gamma=1.0):
        
        # Copy and save dataset as instance variables for plotting decision boundary later.
        self.X_train_copy = X_train
        self.Y_train_copy = Y_train 
        self.X_test_copy = X_test 
        self.Y_test_copy = Y_test 

        true_X = np.ones(X_train.shape[0]).astype(int)
        true_Y = -np.ones(Y_train.shape[0]).astype(int)
        # Concatenante two training sets
        self.X = np.concatenate((X_train, Y_train))
        self.y = np.concatenate((true_X, true_Y))

        test_X = np.ones(X_test.shape[0]).astype(int)
        test_Y = -np.ones(Y_test.shape[0]).astype(int)
        # Concatenante two test sets
        self.X_test = np.concatenate((X_test, Y_test))
        self.y_test = np.concatenate((test_X, test_Y))

        self.C = C
        self.features = features
        self.sigma_sq = sigma_sq
        self.kernel = kernel
        self.a = np.zeros(features)
        self.b = 0.
        self.degree=degree
        self.gamma=gamma
            
    def rbf_kernel(self, x1, x2):
        sq_norm = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-sq_norm / (2 * self.sigma_sq))
    
    def polynomial_kernel(self, x1, x):
        K = (1 + self.gamma * np.dot(x1, x.T)) ** self.degree
        return K

    def fit(self):
        y = self.y.copy()
        x = self.X.copy()
        self.initial = x.copy()
        
        if self.kernel == "rbf":
            x = self.rbf_kernel(x, x)
        elif self.kernel == "poly":
            x = self.polynomial_kernel(x, x)  

        # Use CVXPY to formulate and solve the optimization problem.
        a = cp.Variable(x.shape[0], nonneg=True)
        cost = cp.sum(cp.maximum(0, 1 - cp.multiply(y, x @ a))) + self.C*1/2 * cp.sum_squares(a) 
        #print(y.shape, x.shape, alpha.shape)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        self.a = (a.value * y) @ x
        self.b = np.mean(y - x @ self.a)

    def evaluate(self, mode=None):
        # Detect if the user entered a valid mode.
        if mode not in {'training', 'Training', 'test', 'Test'}:
            raise ValueError('Unknown evaluation mode.')
        # Evaluate based on mode.
        if (mode == 'training') or (mode == 'Training'):
            pred = self.predict(self.X)
            pred = np.where(pred == -1, 0, 1)
            diff = np.abs(np.where(self.y == -1, 0, 1) - pred)
            return ((len(diff) - sum(diff)) / len(diff))
        elif (mode == 'test') or (mode == 'Test'):
            pred = self.predict(self.X_test)
            pred = np.where(pred == -1, 0, 1)
            diff = np.abs(np.where(self.y_test == -1, 0, 1) - pred)
            return ((len(diff) - sum(diff)) / len(diff))

    def predict(self, x):
        # Choose the specified kernel to make predictions.
        if self.kernel == "rbf":
            x = self.rbf_kernel(x, self.initial)
        elif self.kernel == "poly":
            x = self.polynomial_kernel(x, self.initial)
        return np.where(np.dot(x, self.a) + self.b > 0, 1, -1)
    
    def plot_decision_boundary(self, mode=None):
        # Function to plot the final decision boundary, just like before.
        from matplotlib.colors import ListedColormap
        light_cmap = ListedColormap(['#F5E5C9', '#A2DCDE'])

        if mode not in {'training', 'Training', 'test', 'Test'}:
            raise ValueError('Unknown evaluation mode.')
        if (mode == 'training') or (mode == 'Training'):
            X = self.X.copy()
            y = self.y.copy()
            X_data = self.X_train_copy.copy()
            Y_data = self.Y_train_copy.copy()
        elif (mode == 'test') or (mode == 'Test'):
            X = self.X_test.copy()
            y = self.y_test.copy()
            X_data = self.X_test_copy.copy()
            Y_data = self.Y_test_copy.copy()
        # Create grid.
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                            np.arange(x2_min, x2_max, 0.1))
        # Evaluate the SVM model at each point in the grid.
        Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        # Plot the points and the decision boundary.
        plt.contour(xx1, xx2, Z, levels=[0], colors='blueviolet')
        plt.contourf(xx1, xx2, Z, cmap=light_cmap)
        # Chosse the right data points and different colors to plot.
        if (mode == 'training') or (mode == 'Training'): 
            plt.scatter(X_data[:, 0], X_data[:, 1], color='tab:blue', label='X (+1)')
            plt.scatter(Y_data[:, 0], Y_data[:, 1], color='tab:red', label='Y ( - 1)')
        elif (mode == 'test') or (mode == 'Test'): 
            plt.scatter(X_data[:, 0], X_data[:, 1], color='tab:green', label='X (+1)')
            plt.scatter(Y_data[:, 0], Y_data[:, 1], color='tab:orange', label='Y ( - 1)')
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc='best')
        plt.show()



