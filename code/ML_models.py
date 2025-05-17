import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso #Regressors
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDClassifier #Classifiers
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RepeatedKFold#Model evaluation
from sklearn.model_selection import cross_val_score

import time



# Since the goal is to predict housing characteristics (such as pricing), 
# on labelled data, the use of Supervised Learning (classification and regression)
# seems more appropriate. Unsupervised Learning may be of use if
# "tendencies" (from an unlabelled data perspective) become a topic 
# of interest.


class SupervisedLearning:
    def __init__(self):
        pass
    
    class Regression():
        def __init__(self):
            pass
            
        #Lasso Regression
        def lassoRegression(cls, X_train, X_test, y_train):
            # 'alpha' controls the regularization term (the penalization factor) 
            # if it's too large for the values at hand, the weight of penalized
            # coefficients will drop to 0 very quickly, resulting in sparsity and
            # the prediction being equal to the 'intercept' for every value, which 
            # is the mean of the dependent variable
            # (y= 0*X + mean_of_y)
            lasso_model = Lasso(alpha=0.002)             
            lasso_model.fit(X_train, y_train)
    
            #
            # Evaluate the model
            #
            # RepeatedKFold with 10 folds, trained/repeated 10 times
            cv_splitting_strategy = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42) # decide the cross-validation splitting strategy
            
            # the n_jobs = -1 means using all processors to prallelize the
            # training of a split with compating another's score
            #
            # scoring = 'some_metric' -> other metrics available at:
            # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            # Average MAE of the splits for each repetition #
            scores = cross_val_score(lasso_model, X_train, y_train, 
                                     scoring='neg_mean_absolute_error', 
                                     cv=cv_splitting_strategy, n_jobs=-1) 
            scores = np.absolute(scores) 
            print(f"MAE: {np.mean(scores)}, std:{np.std(scores)}")
        
            predicted_values = lasso_model.predict(X_test) 
            
            return predicted_values

        
        def randomForestsRegressor(cls, X_train, X_test, y_train):
            # n_estimators: number of models/trees that will be created
            # (with different samples and features) to ve averaged out in the end
            #
            # max_features: maximum number of features it can consider in each
            # tree model(estimator)
            #         - default: 1.0 (n_features);
            #         - 'sqrt' (=sqrt(n_features)) (or sqrt(x_train_#columns))
            #
            # n_jobs = Number of jobs to run in parallel ('fit', 'predict',
            #  'decision_path' and 'apply' are all parallelized over the trees)
            #      -(no noticeable difference for the current data size and
            #      learning parameters.)
            random_forests_r_model = RandomForestRegressor(n_estimators=100,
                                                           criterion='squared_error',
                                                           max_features='sqrt',
                                                           n_jobs=-1,
                                                           random_state=42)
            random_forests_r_model.fit(X_train, y_train)
            predicted_values = random_forests_r_model.predict(X_test)
            
            #p#print(f"score:\n{random_forests_r_model.score(predicted_values, y_test.to_numpy().reshape(-1,1))}")             
            return predicted_values
            
    

    
    class Classification():
        def __init__(self):
            pass


        def stochasticGradientDescentClassifier(cls, X_train, X_test, y_train):
            # loss: loss function. 
            #     
            #          - 'hinge' (default) -> (linear SVM classification);
            #          - 'log_loss' -> logistic regression, probabilistic classifier;
            #          - etc...
            #
            # penalty: How the feature-weight shifts happen during the learning
            #          process; regularization term. 
            #              - default: 'l2';
            #              - 'l1';
            #              - 'None';
            #              -  etc..
            # alpha: controls the regularization term (multiplier of the penalty).
            #        Also affects the learning rate (when it is set to optimal).
            #
            # max_iter: maximum number of epochs (passes over the training data).
            # n_jobs: parallelization. Number of CPU cores to use during the OVA
            #         (one versus all) computation for multiclass (>2 classes)
            #         classification problems.
            #
            # average: average SGD weights across all updates (epoch average(?))
            #          for each feature, stored in the '.coef_' attribute.
            sgd_c_model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001,
                                        max_iter=1000, n_jobs=-1, random_state=42,
                                        average=False) 
            sgd_c_model.fit(X_train,y_train)
            predicted_values = sgd_c_model.predict(X_test)
            
            return predicted_values



        def multiLayerPerceptronClassifier(cls, X_train, X_test, y_train):
            # hidden_layer_sizes: number of layers and neurons per layer.
            #     - default: (100,)  -> 100 neurons, 1 layer
            #
            # solver: solver for weight optimization.
            #     - default: 'adam' (recommended for large datasets);
            #     - 'lbfgs' (recommended for small datasets, converges faster)
            #
            # alpha: multiplier of the (L2) regularization term.
            #
            # (only for solver='sgd') learning_rate: learning rate plan for each
            #                                        weight update
            #     - default: 'constant' -> const. learning rate, given by 
            #                              learning_rate_init;
            #     - 'invscaling' -> learning rate decreases at each step                    
            #     - 'adaptive' -> learning rate is constant while the training
            #                     loss decreases. Each two consecutive epochs that the loss 
            #                     fails to decrease by  at least "tol"(tolerance), the 
            #                     learning rate decreases by a factor of 5.                
            #
            # (only for solver='sgd'|'adam') learning_rate_init: controls the
            #                                                    step-size in
            #                                                    updating the
            #                                                    weights in 
            #                                                    each epoch.
            #     - default=0.001
            #
            #  max_iter: maximum number of: epochs (for stochastic the solvers
            #                               'sgd' or 'adam').
            #               "      "    of: gradient steps (for 'lbfgs')
            #                           The solver learns until convergence or
            #                           until it has reached this number of
            #                           gradient steps or epochs.
            #
            # verbose: whether to print progress messages. 
            #     - default: False
            #
            # (only for solver='sgd'|'adam') n_iter_no_change: maximum number of
            #                                                  epochs to not meet
            #                                                  the tolerance
            #                                                  improvement (in
            #                                                  the loss function).
            #     - default: 10
            #
            # (only for solver='lbfgs') max_fun: maximum number of loss function
            #                                    calls before the algorithm 
            #                                    stops (competes with max_iter
            #                                    and tol convergence).
            #     - default: 15000
            mlp_c_model = MLPClassifier(solver='adam', alpha=0.0001,
                                        learning_rate='constant',  
                                        learning_rate_init=0.00028,
                                        max_iter=2000, random_state=42,
                                        verbose=False, max_fun=15000)
            mlp_c_model.fit(X_train, y_train)
            predicted_values = mlp_c_model.predict(X_test)

            return predicted_values

