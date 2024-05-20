import xgboost as xgb


class ClassifierBase:
    def __init__(self):
        pass


class ClassifierXGB(ClassifierBase):
    def __init__(self):
        super().__init__()
        self.params = {
            'max_depth': 3,
            'eta': 0.1,
            'objective': 'binary:logistic',
            # 'objective': 'multi:softmax',
            # 'num_class': 3,
        }
        self.model = None

    def train(self, data, num_boost_round=100):
        self.model = xgb.train(self.params, data, num_boost_round=num_boost_round)

    def infer(self, data):
        return self.model.predict(data)
