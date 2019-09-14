import numpy as np, pandas as pd
import pickle

from collections import Counter, defaultdict
from tqdm import tqdm_notebook

from sklearn.metrics import mean_squared_error

from utils import validation
from utils.predict import PredictorPipeline
from utils.models import *
from utils.base import BaseModel



def load_logger(path):
    with open(path, 'rb+') as fin:
        logger = pickle.load(fin)
    return logger

class Logger(BaseModel):
    """ Trains models from models.py and records all the results. """
    def __init__(self, data):
        """ Construct and train LGBMRegressor.
        
        Parameters
        ----------
        data: pandas dataframe
            Initial dataframe with both train and test. 
            Needs specific columns to be present:
                group, izbir_region, izbir_town, isTrain, target.
        
        Attributes
        ----------
        models: list
            List of trained models
        score: float
            Score of the latest predictions ensemble
            
        """
        super(Logger, self).__init__()
       
        self.y = data.loc[data.isTrain, 'target'].values
        self.isTrain = data.isTrain.values
        self.groups = data['group'].values
        self.val_groups = self.groups[self.isTrain]
        self.test_groups = self.groups[~self.isTrain]
        
        self.n_models = 0
        self.models = []
        
    def get_CVModel(self, data, features, params, seed, verbose):
        """ Trains CVModel instance

        Parameters
        ----------
        data: pandas dataframe
            Initial dataframe with both train and test. 
            Needs specific columns to be present:
                group, izbir_region, izbir_town, isTrain, target.
        features: list of strings
            List with dataframe column names to be regarded as features for 
            LGBMRegressor model.
        params: dict
            Parameters of LGBMRegressor and Validator
        seed: int, optional (default=19)
            Validation seed
        verbose: bool, optional (default=False)
            Enables output

        Returns
        -------
        score: float
            CV score of trained instance
        """
        
        self.n_models += 1
        new_model = CVModel(data, features, params, seed, verbose)
        self.models.append(new_model)
        
        return new_model.score
    
    def get_Bagger(self, data, features, params, seed=19, n_seeds=3, gap=None, verbose=False):
        """ Trains Bagger instance

        Parameters
        ----------
        data: pandas dataframe
            Initial dataframe with both train and test. 
            Needs specific columns to be present:
                group, izbir_region, izbir_town, isTrain, target.
        features: list of strings
            List with dataframe column names to be regarded as features for 
            LGBMRegressor model.
        params: dict
            Parameters of LGBMRegressor and Validator
        seed: int, optional (default=19)
            Validation seed
        n_seeds: int, optional (default=3)
            Amount of LGBMRegressor models to be trained
        gap: int, optional (default=None)
            Patience value for early stopping
        verbose: bool, optional (default=False)
            Enables output

        Returns
        -------
        score: float
            OOF score of trained instance
        """
        
        self.n_models += 1
        new_model = Bagger(data, features, params, seed, n_seeds, gap, verbose)
        self.models.append(new_model)
        
        return new_model.score
    
    def get_valBagger(self, data, features_dict, params, n_seeds=10, n_inner=3, gap=None, inner_gap=None,
                     subsample_func=lambda x: list(x.values()), random_state=19, verbose=False):
        """ Trains valBagger instance

        Parameters
        ----------
        data: pandas dataframe
            Initial dataframe with both train and test. 
            Needs specific columns to be present:
                group, izbir_region, izbir_town, isTrain, target.
        features_dict: dict of lists
            Dictionary with lists of non-repeating features.
        params: dict
            Parameters of LGBMRegressor and Validator and predictr
        n_seeds: int, optional (default=10)
            Amount of Baggers to be trained
        n_inner: int, optional (default=3)
            Amount of CVModels inside each Bagger to be trained
        gap: int, optional (default=None)
            Patience value for early stopping valBagger
        inner gap: int, optional (default=None)
            Patience value for early stopping Baggers inside valBagger
        subsample_func: function, optional (default=lambda x: list(x.values()))
            Function used to sample features from features_dict
        random_state: int (default=19)
            Random state for subsample function
        verbose: bool, optional (default=False)
            Enables output

        Returns
        -------
        score: float
            OOF score of trained instance
        """
        
        self.n_models += 1
        new_model = valBagger(data, features_dict, params, n_seeds, n_inner, gap, inner_gap,
                             subsample_func, random_state, verbose)
        self.models.append(new_model)
        
        return new_model.score
    
    def get_KNN(self, data, k=5, verbose=False):
        
        self.n_models += 1
        new_model = KNN(data, k, verbose)
        self.models.append(new_model)
        
        return new_model.score
    
    def set_predictor(self, predictors, mode='CV', verbose=False):
        """ Applies prediction ensembling with given predictors

        Parameters
        ----------
        predictors: list of dicts
            Each dict contains name of prediction transformation and information about regularization coefficient
        mode: str, optional (default='CV')
            Predictor mode

        Returns
        -------
        score: float
            OOF score of adjusted predictions
        """
        
        val_preds = [model.get_preds() for model in self.models]
        test_preds = [model.get_test_preds() for model in self.models]

        if mode == 'CV':
        
            self.predictor = CVPredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
                
        elif mode == 'RegCV':
        
            self.predictor = RegCVPredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
                
        else:
        
            self.predictor = PredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
        
        self.val_preds = self.predictor.get_preds()
        self.test_preds = self.predictor.get_test_preds()
        self.score = self.predictor.score
        
        return self.score
    
    def save(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self, fout)
            
        return path
    
    def features(self):
        self.importances = Counter()
        for model in self.models:
            self.importances.update(model.importances)
            
        return self.importances.most_common()
  
    
    def get_importances(self): 
        return self.importances.most_common()
