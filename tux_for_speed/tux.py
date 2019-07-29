import os

from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

import pandas as pd
from random import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class TuxML:
    
    def __init__(self, dataset=None, resultsPath=None, perf="vmlinux", graphPath=None, nbFolds = 10, minSampleSize=2, maxSampleSize=None, paceSampleSize=None, nb_bins=50, hyperparams=None, columns_to_drop=None, nb_yes=1, algo="rf", semaphore=None):
        
        self.dataset = dataset
        
        #Create the folder for results if it does not exist:
        if not os.path.exists(resultsPath):
            try:
                os.makedirs(resultsPath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                    
        if not graphPath == None:
            #Create the folder for graphs if it does not exist:
            if not os.path.exists(graphPath):
                try:
                    os.makedirs(graphPath)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        
        self.resultsPath = resultsPath
        self.perf = perf
        self.graphPath = graphPath
        self.nbFolds = nbFolds
        self.nb_bins = nb_bins
        self.minSampleSize = minSampleSize
        if maxSampleSize == None and not dataset is None:
            self.maxSampleSize = int(self.dataset.shape[0] * 0.9)
        elif not maxSampleSize == None:
            self.maxSampleSize = maxSampleSize
            
        if paceSampleSize == None:
            self.paceSampleSize = int(self.maxSampleSize/self.nb_bins)
        else:
            self.paceSampleSize = paceSampleSize
            
            
        self.columns_to_drop = columns_to_drop
        self.nb_yes = nb_yes
        self.semaphore = semaphore
        
        self.saveFile = None
        self.dfResults = None
        
        if int(self.minSampleSize) < 2:
            raise Exception('minSampleSize cannot be less than 2')
            
        self.algo = algo
        
        if self.algo == "rf":
            self.hyperparams =  {
                "criterion":"mse",
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.,
                "max_features":"auto",
                "random_state":None,
                "max_leaf_nodes":None,
                "min_impurity_decrease":1e-7,
                "min_impurity_split":0,
                "n_estimators":100,
                "n_jobs":-1
            }
        elif self.algo == "gb":
            self.hyperparams = {
                "criterion":"friedman_mse",
                "loss":"ls",
                "learning_rate":0.1,
                "n_estimators":100,
                "subsample":1.0,
                "max_features":None,
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.,
                "max_leaf_nodes":None,
                "random_state":None,
                "min_impurity_decrease":1e-7,
                "presort":False
            }
        elif self.algo == "dt":
            self.hyperparams = {
                "criterion":"mse",
                "splitter":"best",
                "max_depth":None,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "min_weight_fraction_leaf":0.,
                "max_features":None,
                "random_state":None,
                "max_leaf_nodes":None,
                "min_impurity_decrease":1e-7,
                "presort":False
            }
        else:
            raise("wrong algo "+self.algo)
        
        if not hyperparams == None:
            for k,v in hyperparams.items():
                if k in self.hyperparams:
                    self.hyperparams[k] = v
                    
                    
                    
    def _runRF(self, train_size):
        df = self.dataset
        
        dfErrors = pd.DataFrame()
        dfImportance = pd.DataFrame()
        col = self.dataset.drop(columns=self.columns_to_drop, errors="ignore").columns
        
        for i in range(0,self.nbFolds):
            print("Fold",i)
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.drop(columns=self.columns_to_drop, errors="ignore"), self.dataset[self.perf], train_size=train_size)
            
            # Give the hyperparams to build the model
            reg = ensemble.RandomForestRegressor(**self.hyperparams)

            # Train the random forest
            reg.fit(X_train, y_train)

            # Prediction and scoring
            y_pred = reg.predict(X_test)
            
            dfErrors = dfErrors.append([pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})], ignore_index=True)
            
            dfImportance = dfImportance.append([pd.Series(reg.feature_importances_, index=col.values)])
            
        return dfErrors["% error"].describe(), dfImportance.mean()
                    
                    
    def _runGB(self, train_size):
        df = self.dataset
        
        dfErrors = pd.DataFrame()
        dfImportance = pd.DataFrame()
        col = self.dataset.drop(columns=self.columns_to_drop, errors="ignore").columns
        
        for i in range(0,self.nbFolds):
            print("Fold",i)
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.drop(columns=self.columns_to_drop, errors="ignore"), self.dataset[self.perf], train_size=train_size)
            
            # Give the hyperparams to build the model
            reg = ensemble.GradientBoostingRegressor(**self.hyperparams)

            # Train the random forest
            reg.fit(X_train, y_train)

            # Prediction and scoring
            y_pred = reg.predict(X_test)
            
            dfErrors = dfErrors.append([pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})], ignore_index=True)
            
            dfImportance = dfImportance.append([pd.Series(reg.feature_importances_, index=col.values)])
            
        return dfErrors["% error"].describe(), dfImportance.mean()
          
    
    def _runDT(self, train_size):
        df = self.dataset
        
        dfErrors = pd.DataFrame()
        dfImportance = pd.DataFrame()
        col = self.dataset.drop(columns=self.columns_to_drop, errors="ignore").columns
        
        for i in range(0,self.nbFolds):
            print("Fold",i)
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.drop(columns=self.columns_to_drop, errors="ignore"), self.dataset[self.perf], train_size=train_size)
            
            # Give the hyperparams to build the model
            reg = tree.DecisionTreeRegressor(**self.hyperparams)

            # Train the random forest
            reg.fit(X_train, y_train)

            # Prediction and scoring
            y_pred = reg.predict(X_test)
            
            dfErrors = dfErrors.append([pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})], ignore_index=True)
            
            dfImportance = dfImportance.append([pd.Series(reg.feature_importances_, index=col.values)])
            
        return dfErrors["% error"].describe(), dfImportance.mean()
    
    
    def _save_results(self, results, train_size, random_id):
        results = pd.concat([results,pd.Series(self.hyperparams),pd.Series({"random_id":random_id,"train_size":train_size,"nb_yes":self.nb_yes,"algo":self.algo,"perf":self.perf})])
        if os.path.isfile(self.resultsPath+'/results.csv'):
            line = ",".join([str(i) for i in results.values])
            with open(self.resultsPath+'/results.csv',"a") as f:
                f.write(line+"\n")
                f.close()
        else:
            pd.DataFrame([results]).to_csv(self.resultsPath+'/results.csv', index=False)
    
    def _save_feature_importance(self, feature_importance):
        if os.path.isfile(self.resultsPath+'/feature_importance.csv'):
            line = ",".join([str(i) for i in feature_importance.values])
            with open(self.resultsPath+'/feature_importance.csv',"a") as f:
                f.write(line+"\n")
                f.close()
        else:
            pd.DataFrame([feature_importance]).to_csv(self.resultsPath+'/feature_importance.csv', index=False)
    
    def start(self):
        if not self.semaphore is None:
            self.semaphore.acquire()
        random_id = random()
        for train_size in range(self.minSampleSize, self.maxSampleSize, self.paceSampleSize):
            print("Train size",train_size)
            if self.algo == "rf":
                results, feature_importance = self._runRF(train_size)
            if self.algo == "gb":
                results, feature_importance = self._runGB(train_size)
            if self.algo == "dt":
                results, feature_importance = self._runDT(train_size)
            
            self._save_results(results, train_size, random_id)
            self._save_feature_importance(feature_importance)
        if not self.semaphore is None:
            self.semaphore.release()