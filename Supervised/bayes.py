import numpy as np

class CustomNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        n_docs, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features)) 

        for idx, class_label in enumerate(self.classes):
            X_class = X[y == class_label]
            self.class_log_prior_[idx] = np.log(X_class.shape[0] / n_docs)
            self.feature_log_prob_[idx, :] = np.log((X_class.sum(axis=0) + self.alpha) / 
                                                    (X_class.sum() + self.alpha * n_features))

    def predict(self, X):
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes[np.argmax(log_probs, axis=1)]