import xgboost as xgb


class ClassifierBase:
    def __init__(self):
        pass


class ClassifierXGB(ClassifierBase):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = None

    def train(self, data, num_boost_round=100):
        self.model = xgb.train(self.params, data, num_boost_round=num_boost_round)

    def infer(self, data):
        return self.model.predict(data)
