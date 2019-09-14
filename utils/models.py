import numpy as np, pandas as pd
import pickle, time

from math import sin, cos, sqrt, atan2, radians
from collections import Counter, defaultdict
from tqdm import tqdm_notebook
from IPython.display import clear_output

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors  
from lightgbm import LGBMRegressor

from utils import validation
from utils.predict import PredictorPipeline, CVPredictorPipeline, RegCVPredictorPipeline
from utils.base import BaseModel

R = 6373.0 # радиус земли в километрах

def distance(x, y):
    lat_a, long_a, lat_b, long_b = map(radians, [*x,*y])  
    dlon = long_b - long_a
    dlat = lat_b - lat_a
    a = sin(0.5 * dlat)**2 + cos(lat_a) * cos(lat_b) * sin(0.5 * dlon)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

    
class CVModel(BaseModel):
    """ Implementation of cross-validated LGBMRegressor training. """
    def __init__(self, data, features, params, seed=19, verbose=False):
        """ Construct and train LGBMRegressor.
        
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
        
        Attributes
        ----------
        params: dict
            Parameters of LGBMRegressor and Validator
        seed: int
            Validation seed
        features: list
            List of features
        importances: list
            List of feature importances
        score: float
            CV score
        """
        super(CVModel, self).__init__()
        
        X = data.loc[data.isTrain, features].reset_index(drop=True)
        y = data.loc[data.isTrain, 'target'].reset_index(drop=True)
        test = data.loc[~data.isTrain, features].reset_index(drop=True)
        
        self.params = params.copy()
        self.seed = seed
        self.features = features
        
        gbm = LGBMRegressor(**self.params['model'])
        
        if self.params['validation']['mode'] == 'hierarchy':
            
            labels = [data.loc[data.isTrain, 'group'], 
                      data.loc[data.isTrain, 'izbir_region'].values, 
                      data.loc[data.isTrain, 'izbir_town'].values]
            
            validator = validation.hierarchyValidator(labels, real_ids=None, 
                            n_splits = self.params['validation']['n_splits'], 
                            n_intervals = self.params['validation']['n_intervals'])
            
            splits = validator.split(random_state=self.seed)
            
        elif self.params['validation']['mode'] == 'class':
            
            labels = data.loc[data.isTrain, 'group'].values
            validator = validation.classValidator(labels, self.params['validation']['n_splits'])
            splits = validator.split(random_state=self.seed, mode='standard')
            
        elif self.params['validation']['mode'] == 'reg':
            
            validator = validation.regValidator(y, self.params['validation']['n_intervals'],
                                                self.params['validation']['n_splits'])
            splits = validator.split(random_state=self.seed, mode='standard')
            
        elif self.params['validation']['mode'] == 'classreg':
            
            validator = validation.classregValidator(data.loc[data.isTrain, 'group'].values, y, 
                                                     self.params['validation']['n_intervals'],
                                                     self.params['validation']['n_splits'])
            splits = validator.split(random_state=self.seed, mode='standard')
            
        elif self.params['validation']['mode'] == 'adversarial':
            
            validator = validation.classregValidator(data.loc[data.isTrain, 'group'].values,
                                                    data.loc[data.isTrain, 'test_prob'].values,
                                                    self.params['validation']['n_intervals'],
                                                    self.params['validation']['n_splits'])
            splits = validator.split(random_state=self.seed, mode='standard')
        
        self.val_preds = np.zeros(X.shape[0])
        self.test_preds = []
        self.importances = []
        
        if verbose:
            times = []
            start = time.time()
            print('=================================')
            print('   fold   |   score  |    time ')
            print('=================================')
            scores = []
            
        for idx, split in enumerate(splits):
            
            tr, te = split['train'], split['val']
            train_X, train_y = X.loc[tr, ], y[tr]
            test_X, test_y = X.loc[te, ], y[te]
            
            gbm.fit(train_X, train_y, eval_set=(test_X, test_y), early_stopping_rounds=200, verbose=False)
            
            val_pred = gbm.predict(test_X)
            self.val_preds[te] = val_pred
            self.test_preds.append(gbm.predict(test))
            self.importances.append(gbm.feature_importances_)
            
            if verbose:
                times.append(time.time() - start)
                score = mean_squared_error(val_pred, test_y)
                scores.append(score)
                print('{:6d}    | {:0.6f} | {:0.4f}s '.format(idx + 1, round(score, 6), times[-1]))
            
        self.score = mean_squared_error(self.val_preds, y)
        self.importances = Counter(dict(zip(self.features, np.sum(self.importances, axis=0))))
        self.test_preds = np.mean(self.test_preds, axis=0)
        
        if verbose:
            print('---------------------------------')
            print(' CV score | {:0.6f} | {:0.4f}s '.format(self.score, time.time() - start))
            print(' VAL mean | {:0.6f} | {:0.4f}s '.format(np.mean(scores), np.mean(times)))
            print(' VAL std  | {:0.6f} | {:0.4f}s '.format(np.std(scores), np.std(times)))
            
        return None

    def get_importances(self): 
        return self.importances.most_common()
    
class Bagger(BaseModel):
    """ Implementation of early stopping bagger of LGBMRegressor models trained with the same validation seed. """
    def __init__(self, data, features, params, seed=19, n_seeds=3, gap=None, verbose=False):
        """ Construct and train Bagger.
        
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
        
        Attributes
        ----------
        params: dict
            Parameters of LGBMRegressor and Validator
        seed: int
            Validation seed
        features: list
            List of features
        gap: int
            Gap parameter value
        models: list
            List of trained CVModel instances
        importances: list
            List of feature importances
        score: float
            Bagged CV score (score of averaged OOF preds)
        """
        super(Bagger, self).__init__()
        
        self.params = params.copy()
        self.seed = seed
        self.features = features.copy()
        self.gap = gap
        
        self.models = []
        
        y = data.loc[data.isTrain, 'target'].values
        
        if verbose:
            times = []
            start = time.time()
            print('=================================')
            print('   seed   |   score  |   time ')
            print('=================================')
            scores = []
            
        if gap is None:
            
            for bagging_seed in range(n_seeds):
                
                self.params['model']['seed'] = bagging_seed
                model = CVModel(data, features, self.params, seed=seed)
                self.models.append(model)
                

                if verbose:
                    times.append(time.time() - start)
                    score = mean_squared_error(np.mean([model.get_preds() for model in self.models], axis=0), y)
                    scores.append(score)
                    print('{:6d}    | {:0.6f} | {:0.4f}s '.format(bagging_seed, round(score, 6), times[-1]))
        else:
            
            best_loss = 1.
            patience = 0
            bagging_seed = 0
            while patience < gap:
                
                self.params['model']['seed'] = bagging_seed
                model = CVModel(data, features, self.params, seed=seed)
                self.models.append(model)
                
                self.val_preds = np.mean([model.get_preds() for model in self.models])
                curr_loss = mean_squared_error(self.val_preds, y)
                
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    patience = 0
                else:
                    patience += 1
                    
                if verbose:
                    times.append(time.time() - start)
                    scores.append(curr_loss)
                    print('{:6d}    | {:0.6f} | {:0.4f}s '.format(bagging_seed, round(scores[-1], 6), times[-1]))
                    
                bagging_seed += 1
                
            for i in range(gap):
                self.models.pop()
                
        self.val_preds = np.mean([model.get_preds() for model in self.models], axis=0)
        self.test_preds = np.mean([model.get_test_preds() for model in self.models], axis=0)
        self.score = mean_squared_error(self.val_preds, y)
 
        if verbose:
            print('---------------------------------')
            print('   score  | {:0.6f} | {:0.4f}s '.format(self.score, time.time() - start))
            
        self.importances = Counter()
        for model in self.models:
            self.importances.update(model.importances)
            
        return None
    
    def get_importances(self): return self.importances.most_common()
    
    
    
    
class valBagger(BaseModel):
    """ Implementation of stochastic validation bagger with adaptive prediction ensembling using LGBMRegressor as base model. """
    
    def __init__(self, data, features_dict, params, n_seeds=10, n_inner=3, gap=None, inner_gap=None, 
                 subsample_func=lambda x: list(x.values()), random_state=19, verbose=False):
        """ Construct and train valBagger.
        
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
        
        Attributes
        ----------
        params: dict
            Parameters of LGBMRegressor and Validator
        features: dict
            Dictionary of features
        n_seeds: int
            n_seeds from parameters
        gap: int
            Gap parameter value
        inner_gap: int
            Bagger gap value
        baggers: list
            List of trained Bagger instances
        importances: list
            List of feature importances
        score: float
            valBagger score (score of OOF preds ensembled with Predictor)
        """
        
        super(valBagger, self).__init__()
        
        self.params = params.copy()
        self.features = features_dict.copy()
        self.n_seeds = n_seeds
        self.n_inner = n_inner
        self.gap = gap
        self.inner_gap = inner_gap
        
        self.baggers = []
        
        self.y = data.loc[data.isTrain, 'target'].values
        self.val_groups = data.loc[data.isTrain, 'group'].values
        self.test_groups = data.loc[~data.isTrain, 'group'].values
        
        if verbose:
            scores = []
            self.times = []
            start = time.time()
        
        r = np.random.RandomState(random_state)
        
        try:
        
            if gap is None:
                for seed in range(self.n_seeds):
                    
                    curr_features = subsample_func(features_dict, r)
                    bagger = Bagger(data, curr_features, self.params, seed, n_inner, inner_gap)
                    self.baggers.append(bagger)
                    
                    if verbose:
                        self.times.append(time.time() - start)
                        
                        if seed == 0:
                            score = mean_squared_error(self.baggers[0].get_preds(), self.y)
                        else:
                            score = self.set_predictor()
                        scores.append(score)
                        
                        if params['predict']['predictors'] == ['kkt']:
                            if seed == 0.:
                                print_scores(scores, self.times, weights=[1.])
                            else:
                                print_scores(scores, self.times, self.predictor.predictors[0].lambdas)
                        else:
                            print_scores(scores, self.times)
                            
                self.score = scores[-1]
                    
            else:
                best_loss = 1.
                patience = 0
                seed = 0
                while patience < gap:
                    curr_features = subsample_func(features_dict, r)
                    bagger = Bagger(data, curr_features, self.params, seed, n_inner, inner_gap)
                    self.baggers.append(bagger)
                
                    if best_loss == 1.:
                        curr_loss = mean_squared_error(self.baggers[0].get_preds(), self.y)
                    else:
                        curr_loss = self.set_predictor()
                    
                    seed += 1
                    
                    if verbose:
                        self.times.append(time.time() - start)
                        scores.append(curr_loss)
                        if params['predict']['predictors'] == ['kkt']:
                            if best_loss == 1.:
                                print_scores(scores, self.times, weights=[1.])
                            else:
                                print_scores(scores, self.times, self.predictor.predictors[0].lambdas)
                        else:
                            print_scores(scores, self.times)
                    
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        patience = 0
                    else:
                        patience += 1
            
                for i in range(gap):
                    self.baggers.pop()
            
                self.set_predictor()
                self.score = best_loss
                self.n_seeds = seed + 1
                
        except KeyboardInterrupt:
            self.score = scores[-1]

        if verbose:
            if params['predict']['predictors'] == ['kkt']:
                print_scores(scores, self.times, self.predictor.predictors[0].lambdas, cv_score=self.score)
            else:
                print_scores(scores, self.times, cv_score = self.score)
   
        self.importances = Counter()
        for bagger in self.baggers:
            self.importances.update(bagger.importances)
        
        return None
        
        
    def set_predictor(self, predictors=None, verbose=False):
        val_preds = [model.get_preds() for model in self.baggers]
        test_preds = [model.get_test_preds() for model in self.baggers]
        
        if predictors is None:
            predictors = self.params['predict']['predictors']
            
  
        if self.params['predict']['mode'] == 'CV':
        
            self.predictor = CVPredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
                
        elif self.params['predict']['mode'] == 'RegCV':
        
            self.predictor = RegCVPredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
                
        else:
        
            self.predictor = PredictorPipeline(predictors, val_preds=val_preds, 
                test_preds=test_preds, y=self.y, val_groups=self.val_groups, test_groups=self.test_groups, verbose=verbose)
                
        self.val_preds = self.predictor.get_preds()
        self.test_preds = self.predictor.get_test_preds()
        
        return self.predictor.score
    
    def get_importances(self): 
        return self.importances.most_common()
        
def print_scores(scores, times, weights=None, cv_score=None):
    n = len(scores)

    clear_output()
    if weights is None:
        print('==================================')
        print('  seed  |    score     |   time   ')
        print('==================================')

        for i in range(n):
            print('{:6d}  |  {:0.8f}  |  {:4.4f}'.format(i + 1, scores[i], times[i]))

        if cv_score is not None:
            print('----------------------------------')
            print('   CV   |  {:0.8f}  |  {:4.4f}  '.format(cv_score, times[-1]))
            print('==================================')
            print()
    else:
        print('==============================================')
        print('  seed  |    score     |    time    |  weight ')
        print('==============================================')

        for i in range(n):
            print('{:6d}  |  {:0.8f}  |  {:0.4f}   |  {:0.4f} '.format(i + 1, scores[i], times[i], weights[i]))

        if cv_score is not None:
            print('----------------------------------------------')
            print('   CV   |  {:0.8f}  |  {:0.4f}   |  {:0.4f} '.format(cv_score, times[-1], sum(weights)))
            print('==============================================')
            print()

    return None

    
    
class KNN(BaseModel):
    """ KNN wrapper. """
    def __init__(self, data, k=5, verbose=False):
    
        super(KNN, self).__init__()
        self.k = k
        
        groups = data['group'].unique()
        
        self.preds = np.zeros(data.shape[0])
        
        if verbose:
            print('=======================')
            print('   group  |    score  ')
            print('=======================')
        for group in groups:
            
            mask = data.isTrain & (data['group'] == group)
            neigh = NearestNeighbors(metric=distance)
            neigh.fit(data.loc[mask, ['lat', 'long']].values)
            distances, indexes = neigh.kneighbors(data.loc[data.group == group, ['lat', 'long']].values, 
                                                  np.minimum(mask.sum(), 100), return_distance=True)
            preds = []
            for idx, (dists, ids) in enumerate(zip(distances, indexes)):
                curr_ids = ids[ids != idx]
                preds += [data[mask].reset_index().loc[curr_ids[:self.k], 'target'].mean()]  
                
            self.preds[data.group == group] = np.array(preds)
        
            if verbose:
                score = mean_squared_error(self.preds[mask], data.loc[mask, 'target'].values)
                print('  {:>7.2f} | {:0.8f} '.format(group, round(score, 8)))
                
        self.val_preds = self.preds[data.isTrain]
        self.test_preds = self.preds[~data.isTrain]
        
        self.score = mean_squared_error(self.val_preds, data.loc[data.isTrain, 'target'].values)
        if verbose:
            print('-----------------------')
            print('    CV    | {:0.8f} '.format(self.cv_score))
            print('=======================')
            print()
        
        return None
    
    def get_importances(self): 
        return []
