import datetime

class BaseModel():
    """ Base class for models and logger. """
    def __init__(self):
        
        self.date = str(datetime.datetime.now())[:-7]
        self.val_preds = None
        self.test_preds = None
    
    def get_preds(self):
        return self.val_preds
     
    def get_test_preds(self):
        return self.test_preds
