import numpy as np, pandas as pd
from collections import Counter, defaultdict

class Validator():
    def __init__(self, targets, n_splits=10):
        self._n = len(targets)
        self.n_splits = n_splits
        self.classes = None
        self.ratio = None
        self.class_ids = None
        
    def split(self, random_state=19, mode='standard'):
        r = np.random.RandomState(random_state)
        
        permuted_ids = dict()
        for label in self.classes:
            permuted_ids[label] = r.permutation(self.class_ids[label])

        fold_sizes = {label: int(len(self.class_ids[label]) / self.n_splits) for label in self.classes}
        
        folds = [{label: permuted_ids[label][i * fold_sizes[label]: (i + 1) * fold_sizes[label]] 
                          for label in self.classes if fold_sizes[label] != 0.} 
                            for i in range(self.n_splits)]
        
        remains = {label: permuted_ids[label][self.n_splits * fold_sizes[label]: ] for label in self.classes}
        
        fold_ids = range(len(folds))
        for label in remains:
            amount = len(remains[label])
            if amount > 0:
                chosen_ids = r.choice(fold_ids, amount, replace=False)
                for num, idx in enumerate(chosen_ids):
                    if label in folds[idx]:
                        folds[idx][label] = np.append(folds[idx][label], remains[label][num])
                    else:
                        folds[idx][label] = np.array([remains[label][num]])

        if mode == 'standard':
            merged_folds = [r.permutation(sum([list(fold[label]) for label in self.classes if label in fold], [])).tolist() 
                                for fold in folds]
        else:
            merged_folds = [r.permutation(sum([list(fold[label]) for label in self.classes if fold_sizes[label] != 0], [])).tolist() 
                                for fold in folds]

        splits = [{'train': sum([merged_folds[j] for j in list(range(self.n_splits)[:i]) + list(range(self.n_splits)[i + 1:])], []), 
                   'val': merged_folds[i]} 
                                 for i in range(self.n_splits)]
                    
        if mode =='hierarchy':
            return splits,  set([label for label in self.classes if fold_sizes[label] == 0])
        else:
            return splits
    
    def quantify(self, targets, n_intervals):
        interval_size = 100. / n_intervals
        percentiles = []
        for i in range(n_intervals):
            percentiles.append(sum(percentiles[i - 1 : i]) + interval_size)
        thresholds = np.percentile(targets, percentiles)
    
        label = 0
        thresh = thresholds[label]
        prev_idx = 0

        labels = np.zeros(len(targets))
        enum_targets = sorted(list(enumerate(targets)), key=lambda x: x[1])
        for idx, row in enumerate(enum_targets):
            if row[1] >= thresh:
                labels[[el[0] for el in enum_targets[prev_idx : idx + 1]]] = label
                prev_idx = idx + 1
                label += 1
                if label == n_intervals - 1:
                    break
                thresh = thresholds[label]
        labels[[el[0] for el in enum_targets[prev_idx: ]]] = label
        
        return labels, thresholds
    
    
class classValidator(Validator):
    def __init__(self, labels, n_splits=10):
        self.labels = labels
        super(classValidator, self).__init__(labels, n_splits)
        cnt = Counter(labels)
        self.classes = [el[0] for el in cnt.most_common()]
        self.ratio = {label: cnt[label] / self._n for label in self.classes}
        ids = np.arange(self._n)
        self.class_ids = {label: np.array([idx for idx, obj in enumerate(labels) if obj == label]) 
                          for label in self.classes}
        self.fold_sizes = {label: int(len(self.class_ids[label]) / self.n_splits) for label in self.classes}
        
        
class regValidator(classValidator):
    def __init__(self, targets, n_intervals, n_splits=10):
        new_labels, self.thresholds = self.quantify(targets, n_intervals)
        
        super(regValidator, self).__init__(new_labels, n_splits)
        
        
class multiclassValidator(classValidator):
    def __init__(self, labels, n_splits=10):
        if len(set([len(field) for field in labels])) > 1:
            raise ValueError('All the labels must have identical size')
            
        new_labels = [tuple(obj) for obj in np.vstack(labels).T]

        super(multiclassValidator, self).__init__(new_labels, n_splits)
        
        
class classregValidator(multiclassValidator):
    def __init__(self, labels, targets, n_intervals, n_splits=10):
        target_labels, self.thresholds = self.quantify(targets, n_intervals)
        
        super(classregValidator, self).__init__([labels, target_labels], n_splits)
        

class groupValidator(Validator):
    def __init__(self, labels, groups, n_intervals, n_splits=10):
        self.group_names = list(set(groups))
        
        ids = np.arange(len(labels))
        
        group_labels = []
        group_sizes = []
        self.groupset = []
        for group_name in self.group_names:
            self.groupset.append([obj_idx for obj_idx, obj_group in enumerate(groups) if obj_group == group_name])
            group_labels.append(labels[self.groupset[-1][0]])
            group_sizes.append(len(self.groupset[-1]))
            
        group_sizes = np.array(group_sizes)
        
        super(groupValidator, self).__init__(group_labels, n_splits)
        
        self._validator = classregValidator(group_labels, group_sizes, n_intervals, n_splits)
        
        
    def split(self, random_state=19):
        
        group_splits = self._validator.split(random_state=random_state)
        
        return [
            {
            'train': sum([self.groupset[group] for group in split['train']], []),
            'val': sum([self.groupset[group] for group in split['val']], [])
            } 
                for split in group_splits
        ]
        
        
class hierarchyValidator(Validator):
    def __init__(self, labels, real_ids=None, n_splits=10, level=1, n_intervals=10):
        self.n_splits = n_splits
        m = len(labels) # amount of fields
        if level == 1:
            if real_ids is not None:
                new_labels = []
                for idx, field_labels in enumerate(labels):
                    if idx in real_ids:
                        target_labels, _ = self.quantify(field_labels, n_intervals)
                        new_labels.append(target_labels)
                    else:
                        new_labels.append(field_labels)
            
                self.labels = [tuple(obj) for obj in np.vstack(new_labels).T]
            else:
                self.labels = [tuple(obj) for obj in np.vstack(labels).T]
        else:
            self.labels = labels
            
        self.multi = len(labels[0]) > 1
        
    def split(self, random_state=19):
        if self.multi:
            
            current_validator = classValidator(self.labels, self.n_splits)
            splits, remains = current_validator.split(random_state=random_state, mode='hierarchy')
        
            remain_ids = []
            remain_objects = []
            for idx, obj in enumerate(self.labels):
                if obj in remains:
                    remain_ids.append(idx)
                    remain_objects.append(obj[:-1])
            
            
            next_validator = hierarchyValidator(remain_objects, n_splits=self.n_splits, level=2)
            next_splits = next_validator.split(random_state=random_state)
            
            for i in range(self.n_splits):
                splits[i]['train'] = splits[i]['train'] + [remain_ids[idx] for idx in next_splits[i]['train']]
                splits[i]['val'] = splits[i]['val'] + [remain_ids[idx] for idx in next_splits[i]['val']]
            return splits
        
        else:
            
            val = classValidator([el[0] for el in self.labels], self.n_splits)
            splits = val.split(random_state=random_state, mode='standard')
            
            return splits
            
########################

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
            
class Adversarial:
    def __init__(self, data, features, model_params, val_seed=19, verbose=False):
        adv_data = data.copy()
        adv_data['target'] = (~adv_data.isTrain).astype(int)

        X, y = adv_data[features + ['group']], adv_data['target']

        validator = multiclassValidator([X['group'].values, y])
        splits = validator.split(random_state=val_seed)
        
        gbm = LGBMClassifier(**model_params)
        
        self.val_preds = np.zeros(X.shape[0])
        self.score = []
        self.importances = []
        self.features = features
        self.val_seed = val_seed
        
        if verbose:
            print('=====================')
            print('   fold   |   score ')
            print('=====================')
            scores = []
            
        for idx, split in enumerate(splits):
            tr, te = split['train'], split['val']
            train_X, train_y = adv_data.loc[tr, features], y[tr]
            test_X, test_y = adv_data.loc[te, features], y[te]

            gbm.fit(train_X, train_y, eval_set=(test_X, test_y), early_stopping_rounds=200, verbose=False)
            self.importances.append(gbm.feature_importances_)

            val_pred = gbm.predict_proba(test_X)[:, 1]
            self.val_preds[te] = val_pred

            if verbose:
                score = roc_auc_score(test_y, val_pred)
                scores.append(score)
                print('{:6d}    | {:0.6f} '.format(idx + 1, round(score, 6))) 

        self.cv_score = roc_auc_score(y, self.val_preds)
        
        if verbose:
            print('--------------------')
            print(' CV score | {:0.6f} '.format(self.cv_score))
            print(' VAL mean | {:0.6f} '.format(np.mean(scores)))
            print(' VAL std  | {:0.6f} '.format(np.std(scores)))
            
        self.importances = np.mean(self.importances, axis = 0)
        self.importances = Counter(dict(zip(self.features, gbm.feature_importances_)))
        
    def get_preds(self):
        return self.val_preds
    
    
    def get_importances(self):
        return self.importances.most_common()