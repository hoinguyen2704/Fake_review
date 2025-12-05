"""
Custom Random Forest Implementation from Scratch
Không sử dụng thư viện sklearn, tự implement từ đầu với tối ưu hóa hiệu suất
"""

import numpy as np
import random
from collections import Counter
import time
from typing import Optional, List, Tuple, Dict, Union, Any
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionNode:
    """Node trong Decision Tree với thông tin chi tiết"""
    
    def __init__(self, feature_idx: Optional[int] = None, 
                 threshold: Optional[float] = None, 
                 left: Optional['DecisionNode'] = None, 
                 right: Optional['DecisionNode'] = None, 
                 value: Optional[Any] = None,
                 samples_count: int = 0,
                 impurity: float = 0.0):
        self.feature_idx = feature_idx      # Index của feature để split
        self.threshold = threshold          # Ngưỡng để split
        self.left = left                    # Node con bên trái (<= threshold)
        self.right = right                  # Node con bên phải (> threshold)
        self.value = value                  # Giá trị dự đoán (nếu là leaf node)
        self.samples_count = samples_count  # Số lượng samples tại node này
        self.impurity = impurity            # Impurity của node này

class DecisionTree:
    """Decision Tree từ đầu với tối ưu hóa hiệu suất"""
    
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, 
                 random_state: Optional[int] = None,
                 criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.n_classes = None
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Tính Gini impurity với tối ưu hóa"""
        if len(y) == 0:
            return 0.0
        
        # Sử dụng numpy để tối ưu hóa
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Tính Entropy với tối ưu hóa"""
        if len(y) == 0:
            return 0.0
        
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Tránh log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Tính impurity dựa trên criterion"""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Tìm split tốt nhất với tối ưu hóa hiệu suất"""
        n_samples, n_features = X.shape
        
        if n_samples <= self.min_samples_split:
            return None, None
        
        # Tính impurity hiện tại
        current_impurity = self._calculate_impurity(y)
        best_impurity = current_impurity
        best_feature = None
        best_threshold = None
        
        # Tối ưu: chỉ xem xét một số features ngẫu nhiên nếu có quá nhiều
        if n_features > 100:
            n_features_to_check = min(100, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, size=n_features_to_check, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature_idx in feature_indices:
            # Lấy unique values của feature này (giới hạn số lượng để tối ưu)
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Giới hạn số threshold để kiểm tra
            if len(unique_values) > 100:
                # Sử dụng percentiles để chọn threshold
                percentiles = np.percentile(feature_values, np.linspace(10, 90, 20))
                thresholds = np.unique(percentiles)
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                # Split data sử dụng vectorized operations
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Tính impurity sau khi split
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                left_impurity = self._calculate_impurity(left_y)
                right_impurity = self._calculate_impurity(right_y)
                
                # Weighted impurity
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Cập nhật best split
                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        """Xây dựng tree đệ quy với tối ưu hóa"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            impurity = self._calculate_impurity(y)
            return DecisionNode(
                value=most_common_class,
                samples_count=n_samples,
                impurity=impurity
            )
        
        # Tìm best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            # Không thể split, tạo leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            impurity = self._calculate_impurity(y)
            return DecisionNode(
                value=most_common_class,
                samples_count=n_samples,
                impurity=impurity
            )
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]
        
        # Tạo node con
        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)
        
        # Tính impurity của node hiện tại
        current_impurity = self._calculate_impurity(y)
        
        return DecisionNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            samples_count=n_samples,
            impurity=current_impurity
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Huấn luyện decision tree"""
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x: np.ndarray, node: DecisionNode) -> Any:
        """Dự đoán cho một sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán cho nhiều samples với vectorization"""
        if self.root is None:
            raise ValueError("Model chưa được fit")
        
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(X, 'values'):
            X = X.values
        
        # Vectorized prediction
        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán probability với cải tiến"""
        if self.root is None:
            raise ValueError("Model chưa được fit")
        
        predictions = self.predict(X)
        
        # Tạo probability matrix
        probas = np.zeros((len(X), self.n_classes))
        for i, pred in enumerate(predictions):
            probas[i, pred] = 1.0
        
        return probas
    
    def get_depth(self) -> int:
        """Lấy độ sâu của tree"""
        def _get_node_depth(node: DecisionNode, current_depth: int) -> int:
            if node is None or node.value is not None:
                return current_depth
            return max(
                _get_node_depth(node.left, current_depth + 1),
                _get_node_depth(node.right, current_depth + 1)
            )
        
        return _get_node_depth(self.root, 0) if self.root else 0

class CustomRandomForest:
    """Random Forest từ đầu với tối ưu hóa hiệu suất"""
    
    def __init__(self, n_estimators: int = 100, 
                 max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, 
                 max_features: Union[str, float, int] = 'sqrt', 
                 bootstrap: bool = True, 
                 random_state: Optional[int] = None, 
                 n_jobs: int = 1,
                 criterion: str = 'gini',
                 warm_start: bool = False,
                 memory_efficient: bool = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.warm_start = warm_start
        self.memory_efficient = memory_efficient
        
        self.trees = []
        self.feature_indices = []
        self.n_features = None
        self.n_classes = None
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo bootstrap sample với tối ưu hóa memory"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Xử lý pandas objects và tối ưu memory
        if hasattr(X, 'iloc'):
            # Sử dụng pandas indexing để tránh copy data
            X_sample = X.iloc[indices].values
        else:
            # Sử dụng numpy indexing với copy=False nếu có thể
            X_sample = X[indices]
            
        if hasattr(y, 'iloc'):
            y_sample = y.iloc[indices].values
        else:
            y_sample = y[indices]
            
        return X_sample, y_sample
    
    def _get_feature_subset(self, n_features: int) -> np.ndarray:
        """Lấy subset của features với tối ưu hóa"""
        if self.max_features == 'sqrt':
            n_subset = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_subset = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            n_subset = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            n_subset = min(self.max_features, n_features)
        else:
            n_subset = n_features
        
        return np.random.choice(n_features, size=n_subset, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomRandomForest':
        """Huấn luyện Random Forest với tối ưu hóa"""
        logger.info(f"🌲 Đang huấn luyện Random Forest với {self.n_estimators} trees...")
        start_time = time.time()
        
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.n_classes = len(np.unique(y))
        
        # Kiểm tra memory usage và tự động điều chỉnh
        estimated_memory = self._estimate_memory_usage(X, y)
        if estimated_memory > 1.0:  # Nếu > 1GB
            logger.warning(f"⚠️ Estimated memory usage: {estimated_memory:.2f} GB")
            if not self.memory_efficient:
                logger.info("🔄 Tự động bật memory efficient mode")
                self.memory_efficient = True
        
        # Kiểm tra warm start
        if self.warm_start and self.trees:
            start_idx = len(self.trees)
            logger.info(f"🔄 Warm start: tiếp tục từ tree {start_idx + 1}")
        else:
            start_idx = 0
            self.trees = []
            self.feature_indices = []
        
        for i in range(start_idx, self.n_estimators):
            if i % max(1, self.n_estimators // 10) == 0:
                logger.info(f"  Tree {i+1}/{self.n_estimators}")
            
            # Tạo bootstrap sample hoặc sử dụng data gốc
            if self.bootstrap and not self.memory_efficient:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                # Memory efficient mode: sử dụng data gốc
                X_sample, y_sample = X, y
            
            # Chọn subset features
            feature_subset = self._get_feature_subset(n_features)
            X_subset = X_sample[:, feature_subset]
            
            # Tạo và huấn luyện tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state is not None else None,
                criterion=self.criterion
            )
            
            tree.fit(X_subset, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_subset)
            
            # Memory cleanup nếu cần
            if self.memory_efficient:
                del X_sample, y_sample, X_subset
        
        training_time = time.time() - start_time
        logger.info(f"✅ Random Forest training hoàn thành trong {training_time:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán bằng voting với tối ưu hóa"""
        if not self.trees:
            raise ValueError("Model chưa được fit")
        
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(X, 'values'):
            X = X.values
            
        predictions = []
        
        for tree, feature_subset in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_subset]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Voting (majority vote) với vectorization
        predictions = np.array(predictions)
        final_predictions = []
        
        for i in range(X.shape[0]):
            # Lấy predictions của tất cả trees cho sample i
            sample_predictions = predictions[:, i]
            # Majority vote
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán probability với cải tiến"""
        if not self.trees:
            raise ValueError("Model chưa được fit")
        
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(X, 'values'):
            X = X.values
            
        predictions = []
        
        for tree, feature_subset in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_subset]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        n_samples = X.shape[0]
        
        # Tính probability dựa trên voting
        probas = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            sample_predictions = predictions[:, i]
            counts = Counter(sample_predictions)
            
            for class_label, count in counts.items():
                probas[i, class_label] = count / self.n_estimators
        
        return probas
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Tính feature importance với cải tiến"""
        if not self.trees:
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
        # Đếm số lần mỗi feature được sử dụng
        feature_counts = Counter()
        total_trees = len(self.trees)
        
        for tree, feature_subset in zip(self.trees, self.feature_indices):
            for feature_idx in feature_subset:
                feature_counts[feature_idx] += 1
        
        # Tính importance
        importance = {}
        for feature_idx, count in feature_counts.items():
            if feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                importance[feature_name] = count / total_trees
        
        return importance
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính accuracy score"""
        # Chuyển đổi pandas objects thành numpy arrays nếu cần
        if hasattr(y, 'values'):
            y = y.values
            
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> Dict[str, Any]:
        """Lấy parameters của model"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'criterion': self.criterion,
            'warm_start': self.warm_start,
            'memory_efficient': self.memory_efficient
        }
    
    def _estimate_memory_usage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Ước tính memory usage cho training"""
        # Memory cho data gốc
        data_memory = X.nbytes + y.nbytes
        
        # Memory cho bootstrap samples (nếu có)
        if self.bootstrap and not self.memory_efficient:
            bootstrap_memory = data_memory * self.n_estimators
        else:
            bootstrap_memory = data_memory
        
        # Memory cho trees (ước tính)
        tree_memory = self.n_estimators * 1000  # 1KB per tree (ước tính)
        
        total_memory_gb = (data_memory + bootstrap_memory + tree_memory) / (1024**3)
        return total_memory_gb
    
    def set_params(self, **params) -> 'CustomRandomForest':
        """Set parameters cho model"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class CustomRandomForestClassifier:
    """Wrapper class tương thích với sklearn interface với cải tiến"""
    
    def __init__(self, n_estimators: int = 100, 
                 max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, 
                 max_features: Union[str, float, int] = 'sqrt', 
                 bootstrap: bool = True, 
                 random_state: Optional[int] = None, 
                 n_jobs: int = 1,
                 criterion: str = 'gini',
                 warm_start: bool = False,
                 memory_efficient: bool = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.warm_start = warm_start
        self.memory_efficient = memory_efficient
        
        # Tạo Random Forest
        self.forest = CustomRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            criterion=criterion,
            warm_start=warm_start,
            memory_efficient=memory_efficient
        )
        
        self.classes_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomRandomForestClassifier':
        """Fit model với validation"""
        # Validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X và y phải có cùng số lượng samples")
        
        if X.shape[0] == 0:
            raise ValueError("X không được rỗng")
        
        # Lưu thông tin về classes và features
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
        self.classes_ = np.unique(y_values)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.n_features_in_ = X.shape[1]
        
        # Huấn luyện forest
        self.forest.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict với validation"""
        if not hasattr(self, 'classes_'):
            raise ValueError("Model chưa được fit")
        
        # Validation cho pandas objects
        if hasattr(X, 'shape'):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X phải có {self.n_features_in_} features")
        else:
            raise ValueError("X phải có thuộc tính shape")
        
        return self.forest.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability với validation"""
        if not hasattr(self, 'classes_'):
            raise ValueError("Model chưa được fit")
        
        # Validation cho pandas objects
        if hasattr(X, 'shape'):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X phải có {self.n_features_in_} features")
        else:
            raise ValueError("X phải có thuộc tính shape")
        
        return self.forest.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score với validation"""
        if not hasattr(self, 'classes_'):
            raise ValueError("Model chưa được fit")
        
        return self.forest.score(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance với validation"""
        if not hasattr(self, 'classes_'):
            raise ValueError("Model chưa được fit")
        
        return self.forest.get_feature_importance(self.feature_names_in_)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters"""
        return self.forest.get_params()
    
    def set_params(self, **params) -> 'CustomRandomForestClassifier':
        """Set parameters"""
        self.forest.set_params(**params)
        return self