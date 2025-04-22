import os
import numpy as np
import matplotlib.pyplot as plt


class DecisionStump:

    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = np.inf

        # search over all features and thresholds
        for feature_i in range(n_features):
            X_col = X[:, feature_i]
            thresholds = np.unique(X_col)
            for thr in thresholds:
                for polarity in (1, -1):
                    # predict: +1 everywhere, then flip where polarity * (x - thr) < 0
                    preds = np.ones(n_samples)
                    mask = polarity * (X_col - thr) < 0
                    preds[mask] = -1

                    # weighted error
                    err = np.sum(sample_weights[preds != y])

                    if err < min_error:
                        min_error = err
                        self.polarity = polarity
                        self.threshold = thr
                        self.feature_index = feature_i

    def predict(self, X):
        X_col = X[:, self.feature_index]
        preds = np.ones(X.shape[0])
        mask = self.polarity * (X_col - self.threshold) < 0
        preds[mask] = -1
        return preds


class AdaBoost:

    def __init__(self, n_classifiers = 5):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.alphas = []
        self.sample_weights = None
        self.X = None
        self.y = None
        self._round = 0

        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        else:
            # remove all images in the directory
            for file in os.listdir(self.images_dir):
                if file.endswith(".png"):
                    os.remove(os.path.join(self.images_dir, file))

    def initialize_adaboost(self, X, y):
        self.X = X
        self.y = y

        n_samples = self.y.shape[0]
        self.sample_weights = np.full(n_samples, 1 / n_samples)
        self.classifiers = []
        self.alphas = []
        self._round = 0

    def fit_step(self):
        """
        Performs one boosting round
        """

        # 1) fit stump
        stump = DecisionStump()
        stump.fit(self.X, self.y, self.sample_weights)

        # 2) compute error & alpha
        preds = stump.predict(self.X)
        # error = sum w_i [h(x_i) != y_i]
        err = np.sum(self.sample_weights[preds != self.y])
        err = np.clip(err, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)

        # 3) update weights
        self.sample_weights *= np.exp(-alpha * self.y * preds)
        self.sample_weights /= np.sum(self.sample_weights)

        self.classifiers.append(stump)
        self.alphas.append(alpha)
        self._round += 1

        return stump, alpha, self.sample_weights.copy()

    def predict(self, X):
        """
        Ensemble prediction: sign( sum_t α_t h_t(x) ).
        """
        clf_preds = np.array([clf.predict(X) for clf in self.classifiers])
        # weighted sum over classifiers
        weighted = np.dot(self.alphas, clf_preds)
        return np.sign(weighted)

    def plot(self, plotting=True):
        """
        Plot ensemble decision boundary and data and save figure

        Inputs:
            - plotting (bool): whether to display the plots or not
        """
        resolution = 200

        x_min, x_max = self.X[:,0].min() - 0.5, self.X[:,0].max() + 0.5
        y_min, y_max = self.X[:,1].min() - 0.5, self.X[:,1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        Z = self.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], cmap='bwr')
        sizes = (self.sample_weights / self.sample_weights.max()) * 100
        ax.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='bwr', s=sizes, edgecolor='k')
        ax.set_title(f"Round {self._round}, α={self.alphas[-1]:.2f}")

        file_path = os.path.join(self.images_dir, f"adaboost_{self._round}.png")
        fig.savefig(file_path)

        if plotting:
            plt.show()
        plt.close(fig)
