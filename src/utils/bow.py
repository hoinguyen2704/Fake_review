import numpy as np
import re
from typing import List, Union, Optional
import sys
sys.path.append('../config/')
from config import TEXT_PROCESSING_PARAMS

class CountVectorizer:
    """CountVectorizer tương thích với scikit-learn để xử lý văn bản"""
    
    def __init__(self, 
                 stop_words: Optional[Union[str, List[str], set]] = None,
                 max_features: Optional[int] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 vocabulary: Optional[Union[dict, List[str]]] = None,
                 binary: bool = False,
                 lowercase: bool = True,
                 strip_accents: bool = True,
                 analyzer: str = 'word',
                 token_pattern: str = r"(?u)\b\w\w+\b",
                 ngram_range: tuple = (1, 1),
                 max_document_frequency: Optional[Union[int, float]] = None,
                 min_document_frequency: Optional[Union[int, float]] = None):
        """
        Khởi tạo CountVectorizer
        
        Args:
            stop_words: Stop words để loại bỏ
            max_features: Số features tối đa
            min_df: Tần suất tối thiểu của từ trong documents
            max_df: Tần suất tối đa của từ trong documents
            vocabulary: Từ điển từ vựng
            binary: Nếu True, chỉ đếm có/không có từ
            lowercase: Chuyển về chữ thường
            strip_accents: Loại bỏ dấu
            analyzer: Loại phân tích ('word', 'char', 'char_wb')
            token_pattern: Regex pattern để tách từ
            ngram_range: Phạm vi n-gram (min_n, max_n)
            max_document_frequency: Tần suất tối đa (deprecated, dùng max_df)
            min_document_frequency: Tần suất tối thiểu (deprecated, dùng min_df)
        """
        self.stop_words = stop_words
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = vocabulary
        self.binary = binary
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.analyzer = analyzer
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_document_frequency = max_document_frequency
        self.min_document_frequency = min_document_frequency
        
        # Các thuộc tính được tạo sau khi fit
        self.vocabulary_ = {}
        self.stop_words_ = set()
        self.feature_names_ = []
        self.fixed_vocabulary_ = False
        
        # Xử lý stop_words
        if stop_words == 'english':
            self.stop_words_ = self._get_english_stop_words()
        elif isinstance(stop_words, (list, set)):
            self.stop_words_ = set(stop_words)
        elif stop_words is None:
            self.stop_words_ = set()
        
        # Xử lý vocabulary
        if vocabulary is not None:
            self.vocabulary_ = self._validate_vocabulary(vocabulary)
            self.fixed_vocabulary_ = True
    
    def _get_english_stop_words(self) -> set:
        """Lấy stop words tiếng Anh"""
        return TEXT_PROCESSING_PARAMS['fallback_stopwords']
    
    def _validate_vocabulary(self, vocabulary: Union[dict, List[str]]) -> dict:
        """Xác thực vocabulary"""
        if isinstance(vocabulary, list):
            return {word: idx for idx, word in enumerate(vocabulary)}
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise ValueError("vocabulary phải là list hoặc dict")
    
    def _preprocess_text(self, text: str) -> str:
        """Tiền xử lý văn bản"""
        if self.lowercase:
            text = text.lower()
        
        if self.strip_accents:
            # Loại bỏ dấu tiếng Việt và các dấu khác
            text = self._remove_accents(text)
        
        return text
    
    def _remove_accents(self, text: str) -> str:
        """Loại bỏ dấu tiếng Việt và các dấu khác"""
        # Mapping dấu tiếng Việt
        accent_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        for accented, plain in accent_map.items():
            text = text.replace(accented, plain)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tách từ theo pattern"""
        if self.analyzer == 'word':
            # Sử dụng regex pattern để tách từ
            tokens = re.findall(self.token_pattern, text)
        elif self.analyzer == 'char':
            # Tách theo ký tự
            tokens = list(text)
        elif self.analyzer == 'char_wb':
            # Tách theo ký tự với word boundary
            tokens = []
            for i in range(len(text)):
                if i == 0 or text[i-1].isspace():
                    tokens.append(text[i])
                else:
                    tokens.append(text[i])
        else:
            raise ValueError(f"analyzer '{self.analyzer}' không được hỗ trợ")
        
        return tokens
    
    def _build_ngrams(self, tokens: List[str]) -> List[str]:
        """Xây dựng n-grams"""
        min_n, max_n = self.ngram_range
        
        if min_n == 1 and max_n == 1:
            return tokens
        
        ngrams = []
        for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Lọc tokens theo điều kiện"""
        filtered = []
        
        for token in tokens:
            # Loại bỏ stop words
            if token in self.stop_words_:
                continue
            
            # Kiểm tra độ dài tối thiểu
            if len(token) < TEXT_PROCESSING_PARAMS['min_word_length']:
                continue
            
            # Kiểm tra xem có phải số không
            if token.isdigit():
                continue
            
            filtered.append(token)
        
        return filtered
    
    def fit(self, raw_documents: List[str], y=None) -> 'CountVectorizer':
        """Huấn luyện vectorizer trên tập documents"""
        if self.fixed_vocabulary_:
            return self
        
        # Xử lý tất cả documents để xây dựng vocabulary
        word_counts = {}
        doc_counts = {}
        
        for doc in raw_documents:
            # Tiền xử lý
            processed_doc = self._preprocess_text(doc)
            
            # Tách từ
            tokens = self._tokenize(processed_doc)
            
            # Lọc tokens
            filtered_tokens = self._filter_tokens(tokens)
            
            # Xây dựng n-grams
            ngrams = self._build_ngrams(filtered_tokens)
            
            # Đếm từ và document frequency
            doc_words = set(ngrams)  # Sử dụng set để đếm document frequency
            for word in doc_words:
                doc_counts[word] = doc_counts.get(word, 0) + 1
            
            for word in ngrams:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Lọc theo min_df và max_df
        n_docs = len(raw_documents)
        min_df = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_df = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        
        filtered_words = []
        for word, count in word_counts.items():
            doc_freq = doc_counts[word]
            if min_df <= doc_freq <= max_df:
                filtered_words.append((word, count))
        
        # Sắp xếp theo tần suất giảm dần
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Giới hạn số features
        if self.max_features is not None:
            filtered_words = filtered_words[:self.max_features]
        
        # Xây dựng vocabulary
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        self.feature_names_ = [word for word, _ in filtered_words]
        
        return self
    
    def transform(self, raw_documents: List[str]) -> np.ndarray:
        """Chuyển đổi documents thành ma trận đặc trưng"""
        if not self.vocabulary_:
            raise ValueError("Vectorizer chưa được fit. Hãy gọi fit() trước.")
        
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        
        # Khởi tạo ma trận kết quả
        X = np.zeros((n_docs, n_features), dtype=np.int64)
        
        for doc_idx, doc in enumerate(raw_documents):
            # Tiền xử lý
            processed_doc = self._preprocess_text(doc)
            
            # Tách từ
            tokens = self._tokenize(processed_doc)
            
            # Lọc tokens
            filtered_tokens = self._filter_tokens(tokens)
            
            # Xây dựng n-grams
            ngrams = self._build_ngrams(filtered_tokens)
            
            # Đếm từ
            word_counts = {}
            for word in ngrams:
                if word in self.vocabulary_:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Điền vào ma trận
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    feature_idx = self.vocabulary_[word]
                    if self.binary:
                        X[doc_idx, feature_idx] = 1
                    else:
                        X[doc_idx, feature_idx] = count
        
        return X
    
    def fit_transform(self, raw_documents: List[str], y=None) -> np.ndarray:
        """Kết hợp fit và transform"""
        return self.fit(raw_documents).transform(raw_documents)
    
    def get_feature_names_out(self) -> List[str]:
        """Lấy tên các features (tương thích với scikit-learn)"""
        return self.feature_names_
    
    def get_feature_names(self) -> List[str]:
        """Lấy tên các features (tương thích với scikit-learn cũ)"""
        return self.feature_names_
    
    def get_stop_words(self) -> set:
        """Lấy danh sách stop words"""
        return self.stop_words_
    
    def vocabulary(self) -> dict:
        """Lấy từ điển vocabulary"""
        return self.vocabulary_
    
    def inverse_transform(self, X: np.ndarray) -> List[List[str]]:
        """Chuyển đổi ngược từ ma trận về documents"""
        if not self.vocabulary_:
            raise ValueError("Vectorizer chưa được fit.")
        
        # Tạo mapping ngược
        idx_to_word = {idx: word for word, idx in self.vocabulary_.items()}
        
        documents = []
        for row in X:
            doc_words = []
            for idx, count in enumerate(row):
                if count > 0:
                    word = idx_to_word[idx]
                    doc_words.extend([word] * count)
            documents.append(doc_words)
        
        return documents
    
    def set_params(self, **params):
        """Thiết lập tham số"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Tham số '{key}' không tồn tại")
        return self
    
    def get_params(self, deep=True):
        """Lấy tham số"""
        params = {
            'stop_words': self.stop_words,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vocabulary': self.vocabulary,
            'binary': self.binary,
            'lowercase': self.lowercase,
            'strip_accents': self.strip_accents,
            'analyzer': self.analyzer,
            'token_pattern': self.token_pattern,
            'ngram_range': self.ngram_range
        }
        return params