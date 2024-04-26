class NormalScaler:
    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0) - self.min

    def transform(self, X):
        # Add a small constant to avoid division by zero
        return (X - self.min) / (self.max + 1e-10)
        # return (X - self.min) / self.max

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max + self.min
    

class StandardScaler():
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # the standard deviation can be 0 in certain cases,
        #  which provokes 'devision-by-zero' errors; we can
        #  avoid this by adding a small amount if std==0
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean