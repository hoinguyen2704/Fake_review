import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict

class NaiveBayes:
    def __init__(self, alpha: float = 1.0, var_smoothing: float = 1e-9):
        """
        Args:
            alpha: Laplace smoothing parameter (default: 1.0)
            var_smoothing: Smoothing parameter cho variance (default: 1e-9)
        """
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        
        # Model parameters
        self.class_priors = None
        self.class_counts = None
        self.feature_probs = None
        self.feature_means = None
        self.feature_vars = None
        self.n_classes = None
        self.n_features = None
        self.classes = None
        self.classes_ = None  # Tương thích với sklearn
        # Training info
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        """
        
        Args:
            X: Training data (n_samples, n_features) - có thể là sparse matrix hoặc dense array
            y: Target labels (n_samples,)
        
        Returns:
            self: Trained model
        """
        # Xử lý X - có thể là sparse matrix từ sklearn
        if hasattr(X, 'toarray'):
            # Nếu là sparse matrix, chuyển về dense array
            X = X.toarray()
        else:
            X = np.asarray(X)
        
        # Xử lý y - có thể là list hoặc array
        if hasattr(y, 'shape'):
            y = np.asarray(y)
        else:
            # Nếu y không có shape (ví dụ: list), chuyển về array
            y = np.asarray(y)
        
        # Kiểm tra shape an toàn hơn
        if not hasattr(X, 'shape'):
            raise ValueError("X không có thuộc tính shape")
        
        if not hasattr(y, 'shape'):
            raise ValueError("y không có thuộc tính shape")
        
        # Xử lý trường hợp X có shape không đúng
        if len(X.shape) == 0:
            raise ValueError("X có shape rỗng")
        elif len(X.shape) == 1:
            # Nếu X là 1D, reshape thành (n_samples, 1)
            X = X.reshape(-1, 1)
        elif len(X.shape) > 2:
            raise ValueError(f"X phải là 1D hoặc 2D array, nhận được: {X.shape}")
        
        if len(y.shape) == 0:
            raise ValueError("y có shape rỗng")
        elif len(y.shape) > 1:
            raise ValueError(f"y phải là 1D array, nhận được: {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X và y phải có cùng số lượng samples. X: {X.shape[0]}, y: {y.shape[0]}")
        
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.classes_ = self.classes  # Tương thích với sklearn
        self.n_classes = len(self.classes)
        
        # Tính class priors và counts
        self._compute_class_priors(y)
        
        # Tính feature probabilities cho mỗi class
        self._compute_feature_probabilities(X, y)
        
        self.is_fitted = True
        return self
    
    def _compute_class_priors(self, y: np.ndarray):
        """Tính prior probability cho mỗi class"""
        self.class_counts = {}
        self.class_priors = {}
        
        for class_label in self.classes:
            count = np.sum(y == class_label)
            self.class_counts[class_label] = count
            self.class_priors[class_label] = (count + self.alpha) / (self.n_samples + self.alpha * self.n_classes)
    
    def _compute_feature_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Tính feature probabilities cho mỗi class"""
        self.feature_probs = {}
        self.feature_means = {}
        self.feature_vars = {}
        
        for class_label in self.classes:
            # Lấy data cho class này
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Tính mean và variance cho mỗi feature
            self.feature_means[class_label] = np.mean(X_class, axis=0)
            
            # Tính variance với xử lý đặc biệt cho single sample
            if len(X_class) == 1:
                # Nếu chỉ có 1 sample, variance = 0 + smoothing
                self.feature_vars[class_label] = np.full(self.n_features, self.var_smoothing)
            else:
                # Nếu có nhiều samples, tính variance bình thường
                self.feature_vars[class_label] = np.var(X_class, axis=0, ddof=1)
                # Smoothing variance
                self.feature_vars[class_label] += self.var_smoothing
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class labels
        
        Args:
            X: Input data (n_samples, n_features) - có thể là sparse matrix hoặc dense array
        
        Returns:
            predictions: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được fit")
        
        # Xử lý X - có thể là sparse matrix từ sklearn
        if hasattr(X, 'toarray'):
            # Nếu là sparse matrix, chuyển về dense array
            X = X.toarray()
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for sample in X:
            pred = self._predict_single(sample)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray) -> Any:
        """Dự đoán cho một sample"""
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.classes:
            # Tính log probability để tránh underflow
            log_prob = np.log(self.class_priors[class_label])
            
            # Tính log likelihood cho mỗi feature
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                mean = self.feature_means[class_label][feature_idx]
                var = self.feature_vars[class_label][feature_idx]
                
                # Gaussian probability density
                log_likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((feature_value - mean) ** 2) / var
                log_prob += log_likelihood
            
            if log_prob > best_score:
                best_score = log_prob
                best_class = class_label
        
        return best_class
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class probabilities
        
        Args:
            X: Input data (n_samples, n_features) - có thể là sparse matrix hoặc dense array
        
        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được fit")
        
        # Xử lý X - có thể là sparse matrix từ sklearn
        if hasattr(X, 'toarray'):
            # Nếu là sparse matrix, chuyển về dense array
            X = X.toarray()
        else:
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        probabilities = []
        for sample in X:
            probs = self._predict_proba_single(sample)
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Tính probability cho một sample"""
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        class_scores = {}
        
        # Tính score cho mỗi class
        for class_label in self.classes:
            log_prob = np.log(self.class_priors[class_label])
            
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                mean = self.feature_means[class_label][feature_idx]
                var = self.feature_vars[class_label][feature_idx]
                
                log_likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((feature_value - mean) ** 2) / var
                log_prob += log_likelihood
            
            class_scores[class_label] = log_prob
        
        # Chuyển về probability
        scores = np.array([class_scores[c] for c in self.classes])
        # Tránh overflow bằng cách trừ max score
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Tính accuracy score
        
        Args:
            X: Test data - có thể là sparse matrix hoặc dense array
            y: True labels
        
        Returns:
            accuracy: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> Dict[str, Any]:
        """Lấy model parameters"""
        return {
            'alpha': self.alpha,
            'var_smoothing': self.var_smoothing,
            'n_classes': self.n_classes,
            'n_features': self.n_features,
            'classes': self.classes,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params) -> 'NaiveBayes':
        """Set model parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class GaussianNaiveBayes(NaiveBayes):
    """Gaussian Naive Bayes"""
    
    def __init__(self, var_smoothing: float = 1e-9):
        super().__init__(alpha=0.0, var_smoothing=var_smoothing)


class MultinomialNaiveBayes(NaiveBayes):
    """Multinomial Naive Bayes cho discrete data"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha, var_smoothing=0.0)
    
    def _compute_feature_probabilities(self, X: np.ndarray, y: np.ndarray):
        #2. Ước lượng tham số (feature probabilities)
        """Tính feature probabilities cho multinomial NB"""
        self.feature_probs = {}
        
        for class_label in self.classes:
            # Lấy data cho class này
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Tính count cho mỗi feature
            feature_counts = np.sum(X_class, axis=0)
            total_count = np.sum(feature_counts)
            
            # Laplace smoothing - sử dụng alpha cho mỗi feature
            smoothed_probs = (feature_counts + self.alpha) / (total_count + self.alpha * self.n_features)
            self.feature_probs[class_label] = smoothed_probs
            
    
    def _predict_single(self, x: np.ndarray) -> Any:
        """Dự đoán cho multinomial NB"""
        #3. Tính xác suất cho một sample (predict)
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.classes:
            # Tính log probability
            log_prob = np.log(self.class_priors[class_label])
            
            # Tính log likelihood cho mỗi feature
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                if feature_value > 0:  # Chỉ xét features có value > 0
                    prob = self.feature_probs[class_label][feature_idx]
                    log_prob += feature_value * np.log(prob)
                # Khi feature_value = 0, bỏ qua (không thêm gì vào log_prob)
            
            if log_prob > best_score:
                best_score = log_prob
                best_class = class_label
        
        return best_class
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Tính probability cho multinomial NB"""
        #4. Chuẩn hóa về phân phối xác suất (predict_proba)
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        class_scores = {}
        
        # Tính score cho mỗi class
        for class_label in self.classes:
            log_prob = np.log(self.class_priors[class_label])
            
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                if feature_value > 0:  # Chỉ xét features có value > 0
                    prob = self.feature_probs[class_label][feature_idx]
                    log_prob += feature_value * np.log(prob)
                # Khi feature_value = 0, bỏ qua (không thêm gì vào log_prob)
            
            class_scores[class_label] = log_prob
        
        # Chuyển về probability
        scores = np.array([class_scores[c] for c in self.classes])
        # Tránh overflow bằng cách trừ max score
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities


class BernoulliNaiveBayes(NaiveBayes):
    """Bernoulli Naive Bayes cho binary data"""
    
    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        super().__init__(alpha=alpha, var_smoothing=0.0)
        self.binarize = binarize
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        """Fit Bernoulli NB với binarization"""
        # Xử lý X - có thể là sparse matrix từ sklearn
        if hasattr(X, 'toarray'):
            # Nếu là sparse matrix, chuyển về dense array
            X = X.toarray()
        else:
            X = np.asarray(X)
        
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        return super().fit(X, y)
    
    def _compute_feature_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Tính feature probabilities cho Bernoulli NB"""
        self.feature_probs = {}
        
        for class_label in self.classes:
            # Lấy data cho class này
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Tính probability cho mỗi feature (P(feature=1|class))
            feature_probs = np.mean(X_class, axis=0)
            
            # Laplace smoothing
            smoothed_probs = (feature_probs + self.alpha) / (1 + 2 * self.alpha)
            self.feature_probs[class_label] = smoothed_probs
    
    def _predict_single(self, x: np.ndarray) -> Any:
        """Dự đoán cho Bernoulli NB"""
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        if self.binarize is not None:
            x = (x > self.binarize).astype(float)
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.classes:
            # Tính log probability
            log_prob = np.log(self.class_priors[class_label])
            
            # Tính log likelihood cho mỗi feature
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                prob = self.feature_probs[class_label][feature_idx]
                
                if feature_value == 1:
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(1 - prob)
            
            if log_prob > best_score:
                best_score = log_prob
                best_class = class_label
        
        return best_class
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Tính probability cho Bernoulli NB"""
        # Xử lý x - có thể là sparse matrix từ sklearn
        if hasattr(x, 'toarray'):
            x = x.toarray().flatten()
        else:
            x = np.asarray(x).flatten()
        
        if self.binarize is not None:
            x = (x > self.binarize).astype(float)
        
        class_scores = {}
        
        # Tính score cho mỗi class
        for class_label in self.classes:
            log_prob = np.log(self.class_priors[class_label])
            
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                prob = self.feature_probs[class_label][feature_idx]
                
                if feature_value == 1:
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(1 - prob)
            
            class_scores[class_label] = log_prob
        
        # Chuyển về probability
        scores = np.array([class_scores[c] for c in self.classes])
        # Tránh overflow bằng cách trừ max score
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
