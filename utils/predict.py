import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import mean_squared_error
from utils.base import BaseModel

import datetime

class Predictor(BaseModel):
    def __init__(self, val_preds, test_preds, y):
    
        super(Predictor, self).__init__()
        
        self.old_val_preds = val_preds
        self.old_test_preds = test_preds
        self.y = y
        
        self.score = None
        
        return None
        
        
### KKT Predictor ###
def get_lambda(a, b, y, regcoef=0., regcoef2=0.): 
    val = np.minimum(1, ((a - b).dot(y - b) + regcoef * (1 - 1. / (2. + regcoef2))) / (((a - b)**2).sum() + regcoef))
    val = np.maximum(0, val)
    return val

def get_inner_lambdas(preds, y, regcoef=0., regcoef2=0.):
    
    lmbd = get_lambda(preds[0], preds[1], y, regcoef=regcoef, regcoef2=regcoef2)
    if len(preds) == 2:
        return [lmbd]
    else:
        new_preds = [lmbd * preds[0] + (1 - lmbd) * preds[1]] + preds[2:]
        if regcoef != 0.:
            return [lmbd] + list(get_inner_lambdas(new_preds, y, regcoef=regcoef, regcoef2=regcoef2 + 1.))
        else:
            return [lmbd] + list(get_inner_lambdas(new_preds, y))
    
def get_lambdas(preds, y, regcoef=0.):
    if len(preds) == 1:
        return [1.]
    else:
        inner_lambdas = np.array(get_inner_lambdas(preds, y, regcoef=regcoef))
        lambdas = [np.product(inner_lambdas)]
        for i in range(len(inner_lambdas) - 1):
            lambdas.append((1 - inner_lambdas[i]) * np.product(inner_lambdas[i + 1:]))
        lambdas.append(1 - inner_lambdas[-1])
        return lambdas

class KKTPredictor(Predictor):
    """ Finds coefficients for predictions based on OOF preds. """
    def __init__(self, val_preds, test_preds, y, regcoef=0., verbose=False):
        
        super(KKTPredictor, self).__init__(val_preds, test_preds, y)
        
        self.n_preds = len(val_preds)
        self.lambdas = get_lambdas(self.old_val_preds, self.y, regcoef=regcoef)
        
        self.val_preds = np.sum([self.old_val_preds[i] * self.lambdas[i] for i in range(self.n_preds)], axis=0)
        self.test_preds = np.sum([self.old_test_preds[i] * self.lambdas[i] for i in range(self.n_preds)], axis=0)
        self.score = mean_squared_error(self.val_preds, self.y)
        
        if verbose:
            print('AVG score: {:0.8f}'.format(mean_squared_error(np.mean(self.old_val_preds, axis=0), self.y)), end=', ')
            print('KKT score: {:0.8f}'.format(self.score))
            print('n_models: {}, nonzero coefs: {}'.format(self.n_preds, (np.array(self.lambdas) > 0).sum()))
            
        return None
        
### Linear Predictor ###
def get_linear(a, y, regcoef=0.): 
    return (a.dot(y) + regcoef) / (a.dot(a) + regcoef)

class LinearPredictor(Predictor):
    """ Finds lambda such that lamdas * pred is better than just pred. """
    
    def __init__(self, val_preds, test_preds, y, regcoef=0., verbose=False):
        
        super(LinearPredictor, self).__init__(val_preds, test_preds, y)
        
        self.coef = get_linear(self.old_val_preds, y, regcoef=regcoef)
        
        self.val_preds = self.old_val_preds * self.coef
        self.test_preds = self.old_test_preds * self.coef
        self.score = mean_squared_error(self.val_preds, y)
        
        if verbose:
            print('Coef: {:0.4f}'.format(self.coef))
            print('Old score: {:0.8f}, new score: {:0.8f}'.format(mean_squared_error(self.old_val_preds, y), self.score))
            
        return None
        
        
### Affine Predictor ###
def get_affine(a, y, regcoef=[0., 0.]):
    a_a_regcoef = a.dot(a) + regcoef[1]
    y_a_regcoef = y.dot(a) + regcoef[1]
    a_sum = a.sum()
    
    bias = (y.sum() * a_a_regcoef - y_a_regcoef * a_sum) \
            / (a_a_regcoef * (len(a) + regcoef[0]) - a_sum**2)
        
    coef = (y.sum() - bias * (len(a) + regcoef[0]) ) / a.sum()
    return bias, coef


class AffinePredictor(Predictor):
    """ Finds bias and lambda such that bias + lambda * pred is better than just pred. """
    def __init__(self, val_preds, test_preds, y, regcoef=[0., 0.], verbose=False):
        super(AffinePredictor, self).__init__(val_preds, test_preds, y)
        self.bias, self.coef = get_affine(self.old_val_preds, y, regcoef)
        
        self.val_preds = self.bias + self.coef * self.old_val_preds
        self.test_preds = self.bias + self.coef * self.old_test_preds
        self.score = mean_squared_error(self.val_preds, y)

        if verbose:
            print('Bias: {:0.4f}, coef: {:0.4f}'.format(self.bias, self.coef))
            print('Old score: {:0.8f}, new score: {:0.8f}'.format(mean_squared_error(self.old_val_preds, y), self.score))
            
        return None
        
        
### GroupPredictor ###
class GroupPredictor(Predictor):
    """ Base class for groupwise prediction ensembles. """
    def __init__(self, val_preds, test_preds, y, val_groups, test_groups):
        super(GroupPredictor, self).__init__(val_preds, test_preds, y)
        
        self.val_groups = val_groups
        self.test_groups = test_groups
        
        cnt = Counter(val_groups)
        self.groups = [el[0] for el in cnt.most_common()]
        self.group_ration = {group: cnt[group] / len(val_groups) for group in self.groups}
        
        return None
        
        
### KKTGroupPredictor ### 
class KKTGroupPredictor(GroupPredictor):
    """ Groupwise prediction ensembling. """
    def __init__(self, val_preds, test_preds, y, val_groups, test_groups, regcoef=0., verbose=False):
        super(KKTGroupPredictor, self).__init__(val_preds, test_preds, y, val_groups, test_groups)
        
        self.predictors = dict()
        self.val_preds = np.zeros(len(val_preds[0]))
        self.test_preds = np.zeros(len(test_preds[0]))
        
        if verbose:
            print('Avg score:', round(mean_squared_error(np.mean(self.old_val_preds, axis=0), y), 8))
            print()
        
        for group in self.groups:
            
            val_mask = self.val_groups == group
            test_mask = self.test_groups == group
            
            if verbose:
                print('Group:', group)
                
            predictor = KKTPredictor([pred[val_mask] for pred in self.old_val_preds], 
                                     [pred[test_mask] for pred in self.old_test_preds], 
                                     self.y[val_mask], regcoef=regcoef, verbose=verbose)
            
            if verbose:
                print()
            
            self.predictors[group] = predictor
            
            self.val_preds[val_mask] = predictor.get_preds()
            self.test_preds[test_mask] = predictor.get_test_preds()
                     
        self.score = mean_squared_error(self.val_preds, self.y)
        
        if verbose:
            print('KKT score:', round(self.score, 8))
            
        return None
        
        
### LinearGroupPredictor ### 
class LinearGroupPredictor(GroupPredictor):
    """ Finds best groupwise linear transformation for prediction. """
    def __init__(self, val_preds, test_preds, y, val_groups, test_groups, regcoef=0., verbose=False):
        super(LinearGroupPredictor, self).__init__(val_preds, test_preds, y, val_groups, test_groups)

        self.predictors = dict()
        self.val_preds = np.zeros(len(val_preds))
        self.test_preds = np.zeros(len(test_preds))
        
        if verbose:
            print('Old score:', mean_squared_error(self.old_val_preds, y))
            print()
            
        for group in self.groups:
            
            val_mask = self.val_groups == group
            test_mask = self.test_groups == group
            
            if verbose:
                print('Group:', group)
                
            predictor = LinearPredictor(self.old_val_preds[val_mask], self.old_test_preds[test_mask], self.y[val_mask],
                                       regcoef=regcoef, verbose=verbose)
            
            if verbose:
                print()
                
            self.predictors[group] = predictor
                      
            self.val_preds[val_mask] = predictor.get_preds()
            self.test_preds[test_mask] = predictor.get_test_preds()
                     
        self.score = mean_squared_error(self.val_preds, self.y)
        
        if verbose:
            print('Linear score:', round(self.score, 8))
            
        return None
        
        
        
        
### AffineGroupPredictor ### 
class AffineGroupPredictor(GroupPredictor):
    """ Finds best groupwise affine transformation for prediction. """
    def __init__(self, val_preds, test_preds, y, val_groups, test_groups, regcoef=[0., 0.], verbose=False):
        super(AffineGroupPredictor, self).__init__(val_preds, test_preds, y, val_groups, test_groups)

            
        self.predictors = dict()
        self.val_preds = np.zeros(len(val_preds))
        self.test_preds = np.zeros(len(test_preds))
        
        if verbose:
            print('Old score:', mean_squared_error(self.old_val_preds, y))
            print()
            
        for group in self.groups:
            
            val_mask = self.val_groups == group
            test_mask = self.test_groups == group
            
            if verbose:
                print('Group:', group)
                
            predictor = AffinePredictor(self.old_val_preds[val_mask], self.old_test_preds[test_mask], self.y[val_mask],
                                       regcoef=regcoef, verbose=verbose)
            
            if verbose:
                print()
                
            self.predictors[group] = predictor
                      
            self.val_preds[val_mask] = predictor.get_preds()
            self.test_preds[test_mask] = predictor.get_test_preds()
                     
        self.score = mean_squared_error(self.val_preds, self.y)
        
        if verbose:
            print('Affine score:', round(self.score, 8))
            
        return None
  
### AvgPredictor
class AvgPredictor(Predictor):
    """ Simple averaging predictor. """
    
    def __init__(self, val_preds, test_preds, y):
        
        super(AvgPredictor, self).__init__(val_preds, test_preds, y)
        
        self.val_preds = np.mean(val_preds, axis=0)
        self.test_preds = np.mean(test_preds, axis=0)
        self.score = mean_squared_error(self.val_preds, y)
        
        return None
                
### PredictorPipeline ###

class PredictorPipeline(Predictor):
    def __init__(self, predictors, val_preds, test_preds, y, val_groups, test_groups, verbose=False):
        curr_val_preds = val_preds
        curr_test_preds = test_preds
        
        self.predictors = []
                
        for predictor in predictors:
            
            name = predictor['name']
            
            if name in {'affine', 'group_affine'}:
                regcoef = predictor.get('regcoef', [0., 0.])
            else:
                regcoef = predictor.get('regcoef', 0.)
                
            if name == 'kkt':
                curr_predictor = KKTPredictor(curr_val_preds, curr_test_preds, y, regcoef)
            elif name == 'linear':
                curr_predictor = LinearPredictor(curr_val_preds, curr_test_preds, y, regcoef)
            elif name == 'affine':
                curr_predictor = AffinePredictor(curr_val_preds, curr_test_preds, y, regcoef)
            elif name == 'group_kkt':
                curr_predictor = KKTGroupPredictor(curr_val_preds, curr_test_preds, y, val_groups, test_groups, regcoef)
            elif name == 'group_linear':
                curr_predictor = LinearGroupPredictor(curr_val_preds, curr_test_preds, y, val_groups, test_groups, regcoef)
            elif name == 'group_affine':
                curr_predictor = AffineGroupPredictor(curr_val_preds, curr_test_preds, y, val_groups, test_groups, regcoef)
            elif name == 'avg':
                curr_predictor = AvgPredictor(curr_val_preds, curr_test_preds, y)
                
            self.predictors.append(curr_predictor)
            
            curr_val_preds = curr_predictor.get_preds()
            curr_test_preds = curr_predictor.get_test_preds()
            
            if verbose:
                print(predictor, 'score:', curr_predictor.score)
                
        self.val_preds = curr_val_preds
        self.test_preds = curr_test_preds
        self.score = self.predictors[-1].score
        
        return None
        
        
### CVPredictor ###

from utils.validation import classValidator

class CVPredictorPipeline(Predictor):
    def __init__(self, predictors, val_preds, test_preds, y, val_groups, test_groups, verbose=False):
        val = classValidator(val_groups)
        splits = val.split()
        
        self.n_preds = len(val_preds)
        self.test_preds = []
        self.val_preds = np.zeros(len(val_preds[0]))
        
        for split in splits:
            
            tr, te = split['train'], split['val']
            
            train_X, train_y = [pred[tr] for pred in val_preds], y[tr]
            test_X, test_y = [pred[te] for pred in val_preds], y[te]
            tr_groups, te_groups = val_groups[tr], val_groups[te]
        
            predictor = PredictorPipeline(predictors, train_X, test_preds, train_y, 
                                          tr_groups, test_groups)
            self.test_preds.append(predictor.get_test_preds())
            
            predictor = PredictorPipeline(predictors, train_X, test_X, train_y, tr_groups, te_groups)
            new_pred = predictor.get_test_preds()
            self.val_preds[te] = new_pred
            
        self.test_preds = np.mean(self.test_preds, axis=0)
        
        self.score = mean_squared_error(self.val_preds, y)
        
        
### RegCVPredictor ###

import copy, itertools

class RegCVPredictorPipeline(Predictor):
    def __init__(self, predictors, val_preds, test_preds, y, val_groups, test_groups, tolerance=10, verbose=False):
        val = classValidator(val_groups)

        splits = val.split()
        alpha_values = np.logspace(-8, 2, 12.)
        #alpha_values = np.linspace(0., 1., tolerance)
        
        self.n_preds = len(val_preds)
        self.test_preds = []
        self.val_preds = np.zeros(len(val_preds[0]))
        

        for split_idx, split in enumerate(splits):
            
            new_predictors = copy.deepcopy(predictors)
            
            tr, te = split['train'], split['val']
            
            train_X, train_y = [pred[tr] for pred in val_preds], y[tr]
            test_X, test_y = [pred[te] for pred in val_preds], y[te]
            tr_groups, te_groups = val_groups[tr], val_groups[te]

            for idx, predictor_cfg in enumerate(new_predictors):
                if not 'regcoef' in predictor_cfg:
                    if predictor_cfg['name'] in {'kkt', 'group_kkt', 'linear', 'group_linear'}:
                        scores = []
                        for alpha in alpha_values:
                            predictor_cfg['regcoef'] = alpha
                            predictor = PredictorPipeline(
                                new_predictors[:idx] + [predictor_cfg], 
                                train_X, 
                                test_X, 
                                train_y, 
                                tr_groups, 
                                te_groups
                            )
                            pred = predictor.get_test_preds()
                            scores.append(mean_squared_error(test_y, pred))
                        best_alpha = alpha_values[np.argmin(scores)]
                        predictor_cfg['regcoef'] = best_alpha
                    else:
                        alphavec_values = list(itertools.combinations(alpha_values, 2))
                        scores = []
                        for alpha in alphavec_values:
                            predictor_cfg['regcoef'] = [alpha[0], alpha[1]]
                            predictor = PredictorPipeline(
                                new_predictors[:idx] + [predictor_cfg], 
                                train_X, 
                                test_X, 
                                train_y, 
                                tr_groups, 
                                te_groups
                            )
                            pred = predictor.get_test_preds()
                            scores.append(mean_squared_error(test_y, pred))
                        best_alpha = alphavec_values[np.argmin(scores)]
                        predictor_cfg['regcoef'] = [best_alpha[0], best_alpha[1]]
                        
                if verbose:
                    print(idx, predictor_cfg['regcoef'])
             
            predictor = PredictorPipeline(new_predictors, train_X, test_preds, train_y, 
                                          tr_groups, test_groups)
            self.test_preds.append(predictor.get_test_preds())
            predictor = PredictorPipeline(new_predictors, train_X, test_X, train_y, tr_groups, te_groups)
            new_pred = predictor.get_test_preds()
            self.val_preds[te] = new_pred
            
            
        self.test_preds = np.mean(self.test_preds, axis=0)
        
        self.score = mean_squared_error(self.val_preds, y)
        
        return None
