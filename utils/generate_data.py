from sklearn import datasets


class DataGenration():

    def __init__(self, noOfPoints = 100, noise_level='low'):
        self.noOfPoints = noOfPoints
        if noise_level == 'low':
            self.noise = 20
        elif noise_level == 'medium':
            self.noise = 50
        elif noise_level == 'high':
            self.noise = 100        

    def linearClassificationData(self):
        X, y = datasets.make_regression(
            n_samples = self.noOfPoints,
            n_features = 1, 
            noise = self.noise,
            random_state = 20
        )
        Y = (y > y.mean()).astype(int)  # Convert regression target to classification labels
        return(X, Y)
    