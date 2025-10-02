"""
ML Pipeline
===========
Nesne Yönelimli Programlama ile ML pipeline yönetimi
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report
)

from feature_engineering import FeatureEngineer
from models import PredictionInput, PredictionOutput
from logger import setup_logger
from config import settings


logger = setup_logger(__name__)


# ============================================================================
# BASE ALGORITHM CLASS
# ============================================================================

class BaseAlgorithm:
    """Tüm algoritmaların base sınıfı"""
    
    def __init__(self, parameters: Optional[Dict] = None):
        self.parameters = parameters or {}
        self.model = None
        self.is_fitted = False
    
    def get_model(self):
        """Model instance döndürür - override edilmeli"""
        raise NotImplementedError("Subclass must implement get_model()")
    
    def fit(self, X, y):
        """Model eğitimi"""
        self.model = self.get_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Tahmin"""
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmemiş")
        return self.model.predict(X)
    
    def get_name(self):
        """Algoritma adı"""
        return self.__class__.__name__


# ============================================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================================

class RandomForestAlgorithm(BaseAlgorithm):
    """Random Forest algoritması"""
    
    def get_model(self):
        default_params = {
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        params = {**default_params, **self.parameters}
        
        base_estimator = RandomForestClassifier(**params)
        return ClassifierChain(base_estimator, order=[0, 1, 2], random_state=42)


class GradientBoostingAlgorithm(BaseAlgorithm):
    """Gradient Boosting algoritması"""
    
    def get_model(self):
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        params = {**default_params, **self.parameters}
        
        base_estimator = GradientBoostingClassifier(**params)
        return ClassifierChain(base_estimator, order=[0, 1, 2], random_state=42)


class LogisticRegressionAlgorithm(BaseAlgorithm):
    """Logistic Regression algoritması"""
    
    def get_model(self):
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        params = {**default_params, **self.parameters}
        
        base_estimator = LogisticRegression(**params)
        return ClassifierChain(base_estimator, order=[0, 1, 2], random_state=42)


class NeuralNetworkAlgorithm(BaseAlgorithm):
    """Neural Network (MLP) algoritması"""
    
    def get_model(self):
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True
        }
        params = {**default_params, **self.parameters}
        
        base_estimator = MLPClassifier(**params)
        return ClassifierChain(base_estimator, order=[0, 1, 2], random_state=42)


# ============================================================================
# ALGORITHM FACTORY
# ============================================================================

class AlgorithmFactory:
    """Algorithm Factory Pattern"""
    
    _algorithms = {
        'random_forest': RandomForestAlgorithm,
        'gradient_boosting': GradientBoostingAlgorithm,
        'logistic_regression': LogisticRegressionAlgorithm,
        'neural_network': NeuralNetworkAlgorithm
    }
    
    @classmethod
    def create(cls, algorithm_name: str, parameters: Optional[Dict] = None) -> BaseAlgorithm:
        """Algoritma instance oluşturur"""
        algorithm_class = cls._algorithms.get(algorithm_name)
        
        if algorithm_class is None:
            raise ValueError(f"Desteklenmeyen algoritma: {algorithm_name}")
        
        return algorithm_class(parameters)
    
    @classmethod
    def get_available_algorithms(cls):
        """Mevcut algoritmaları döndürür"""
        return list(cls._algorithms.keys())


# ============================================================================
# ML PIPELINE
# ============================================================================

class MLPipeline:
    """
    Ana ML Pipeline sınıfı
    
    Sorumluluklar:
    - Feature engineering
    - Model eğitimi
    - Tahmin yapma
    - Performans değerlendirme
    - Model kaydetme/yükleme
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.current_algorithm: Optional[str] = None
        self.algorithm: Optional[BaseAlgorithm] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: list = []
        self.training_results: Dict = {}
        self.is_trained_flag: bool = False
        
        logger.info("ML Pipeline initialized")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    def configure(self, algorithm: str, parameters: Optional[Dict] = None) -> bool:
        """
        Algoritma yapılandırması
        
        Args:
            algorithm: Algoritma adı
            parameters: Algoritma parametreleri
        
        Returns:
            bool: Başarılı ise True
        """
        try:
            self.algorithm = AlgorithmFactory.create(algorithm, parameters)
            self.current_algorithm = algorithm
            
            logger.info(f"Algorithm configured: {algorithm}")
            if parameters:
                logger.info(f"Parameters: {parameters}")
            
            return True
        
        except Exception as e:
            logger.error(f"Configuration error: {str(e)}", exc_info=True)
            return False
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    def _prepare_data(self, df: pd.DataFrame, test_size: float = 0.25) -> Tuple:
        """
        Veriyi eğitim için hazırlar
        
        Args:
            df: Ham veri
            test_size: Test set oranı (USE_FIXED_SPLIT=True ise kullanılmaz)
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Data preparation started")
        logger.info(f"Input shape: {df.shape}")
        
        # Feature engineering
        df_transformed = self.feature_engineer.transform(df, is_training=True)
        
        # Feature kolonlarını belirle
        exclude_cols = [
            'seller_number', 'customer_number', 'main_account',
            'description', 'document_number', 'currency_code',
            'payment_type', 'transaction_type', 'amount_range',
            'trans_pay_combo', 'company_group', 'doc_number_first_digit'
        ]
        
        self.feature_columns = [
            col for col in df_transformed.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Feature count: {len(self.feature_columns)}")
        
        # Features
        X = df_transformed[self.feature_columns].copy()
        X = X.fillna(-999)
        
        # Targets - Label Encoding
        target_names = ['seller_number', 'customer_number', 'main_account']
        y_encoded = []
        
        for target in target_names:
            target_data = df[target].fillna('MISSING').astype(str)
            
            le = LabelEncoder()
            encoded = le.fit_transform(target_data)
            y_encoded.append(encoded)
            
            self.label_encoders[target] = le
            logger.info(f"{target}: {len(le.classes_)} unique classes")
        
        y = np.column_stack(y_encoded)
        
        # CASE GEREKSİNİMİ: Sabit 150/50 split
        if settings.USE_FIXED_SPLIT:
            train_size = min(settings.FIXED_TRAIN_SIZE, len(X))
            test_size_actual = min(settings.FIXED_TEST_SIZE, len(X) - train_size)
            
            logger.info("=" * 80)
            logger.info("CASE REQUİREMENT: Fixed 150/50 split kullanılıyor")
            logger.info("=" * 80)
            
            # İlk 150 kayıt train, sonraki 50 kayıt test
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:train_size + test_size_actual]
            y_train = y[:train_size]
            y_test = y[train_size:train_size + test_size_actual]
            
            logger.info(f"Fixed train set: {X_train.shape[0]} samples")
            logger.info(f"Fixed test set: {X_test.shape[0]} samples")
            
        else:
            # Dinamik split (orijinal yöntem)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
            
            logger.info(f"Train set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def train(self, data: pd.DataFrame, test_size: float = 0.25) -> Dict:
        """
        Model eğitimi
        
        Args:
            data: Eğitim verisi
            test_size: Test set oranı
        
        Returns:
            Dict: Eğitim sonuçları ve metrikler
        """
        if self.algorithm is None:
            raise ValueError("Önce configure() metodunu çağırın")
        
        logger.info("=" * 80)
        logger.info("TRAINING STARTED")
        logger.info("=" * 80)
        
        # Veriyi hazırla
        X_train, X_test, y_train, y_test = self._prepare_data(data, test_size)
        
        # Model eğitimi
        logger.info(f"Training {self.current_algorithm}...")
        self.algorithm.fit(X_train, y_train)
        
        # Değerlendirme
        train_metrics = self._evaluate(X_train, y_train, "train")
        test_metrics = self._evaluate(X_test, y_test, "test")
        
        # Sonuçları kaydet
        self.training_results = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_count": len(self.feature_columns),
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        }
        
        self.is_trained_flag = True
        
        # Model kaydet
        self._save_model()
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        
        return self.training_results
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def _evaluate(self, X, y, dataset_name: str) -> Dict:
        """
        Model değerlendirmesi
        
        Args:
            X: Features
            y: Targets
            dataset_name: "train" veya "test"
        
        Returns:
            Dict: Metrikler
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        y_pred = self.algorithm.predict(X)
        
        target_names = ['seller_number', 'customer_number', 'main_account']
        metrics = {}
        
        for idx, target_name in enumerate(target_names):
            y_true = y[:, idx]
            y_pred_single = y_pred[:, idx]
            
            acc = accuracy_score(y_true, y_pred_single)
            f1_w = f1_score(y_true, y_pred_single, average='weighted', zero_division=0)
            f1_m = f1_score(y_true, y_pred_single, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred_single, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred_single, average='weighted', zero_division=0)
            
            metrics[target_name] = {
                'accuracy': float(acc),
                'f1_weighted': float(f1_w),
                'f1_macro': float(f1_m),
                'precision': float(prec),
                'recall': float(rec)
            }
            
            logger.info(f"{target_name}: Acc={acc:.4f}, F1={f1_w:.4f}")
        
        # Overall metrics
        avg_acc = np.mean([m['accuracy'] for m in metrics.values()])
        avg_f1 = np.mean([m['f1_weighted'] for m in metrics.values()])
        
        metrics['overall'] = {
            'accuracy': float(avg_acc),
            'f1_weighted': float(avg_f1)
        }
        
        logger.info(f"Overall: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")
        
        return metrics
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Tahmin yapma
        
        Args:
            input_data: Tahmin için girdi
        
        Returns:
            PredictionOutput: Tahmin sonuçları
        """
        if not self.is_trained_flag:
            raise ValueError("Model henüz eğitilmemiş")
        
        # DataFrame'e çevir
        df = pd.DataFrame([input_data.dict()])
        
        # Feature engineering (inference mode)
        df_transformed = self.feature_engineer.transform(df, is_training=False)
        
        # Features
        X = df_transformed[self.feature_columns].copy()
        X = X.fillna(-999)
        
        # Tahmin
        y_pred = self.algorithm.predict(X)
        
        # Label'lara geri çevir
        predictions = {}
        target_names = ['seller_number', 'customer_number', 'main_account']
        
        for idx, target_name in enumerate(target_names):
            # Float'tan integer'a çevir (Neural Network gibi modeller float döndürebilir)
            encoded_value = int(np.round(y_pred[0][idx]))
            
            # Bounds check - encoded value sınıf sayısından fazla olamaz
            num_classes = len(self.label_encoders[target_name].classes_)
            if encoded_value < 0 or encoded_value >= num_classes:
                logger.warning(f"{target_name}: Invalid prediction {encoded_value}, using 0")
                encoded_value = 0
            
            decoded_value = self.label_encoders[target_name].inverse_transform([encoded_value])[0]
            
            # MISSING ise None yap
            predictions[target_name] = None if decoded_value == 'MISSING' else decoded_value
        
        return PredictionOutput(**predictions)
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def _save_model(self):
            """Modeli ve ilgili artifactları kaydeder"""
            try:
                # Algoritma adını string'e çevir (enum değilse)
                algo_name = str(self.current_algorithm)
                if 'AlgorithmType.' in algo_name:
                    algo_name = algo_name.replace('AlgorithmType.', '').lower()
                
                # Algoritma-spesifik dosyalar (model comparison için)
                model_path = os.path.join(settings.MODELS_DIR, f'model_{algo_name}.pkl')
                encoders_path = os.path.join(settings.MODELS_DIR, f'encoders_{algo_name}.pkl')
                features_path = os.path.join(settings.MODELS_DIR, f'features_{algo_name}.pkl')
                results_path = os.path.join(settings.MODELS_DIR, f'results_{algo_name}.pkl')
                
                joblib.dump(self.algorithm, model_path)
                joblib.dump(self.label_encoders, encoders_path)
                joblib.dump(self.feature_columns, features_path)
                joblib.dump(self.training_results, results_path)
                
                # Backward compatibility: genel dosyalar da kaydet (son eğitilen model için)
                general_encoders = os.path.join(settings.MODELS_DIR, 'label_encoders.pkl')
                general_features = os.path.join(settings.MODELS_DIR, 'feature_columns.pkl')
                general_results = os.path.join(settings.MODELS_DIR, 'training_results.pkl')
                
                joblib.dump(self.label_encoders, general_encoders)
                joblib.dump(self.feature_columns, general_features)
                joblib.dump(self.training_results, general_results)
                
                logger.info(f"Model saved: {model_path}")
                logger.info(f"Results saved: {results_path}")
                
            except Exception as e:
                logger.error(f"Model save error: {str(e)}", exc_info=True)
    
    def load_model(self, algorithm: str):
        """Kaydedilmiş modeli yükler"""
        try:
            model_path = os.path.join(settings.MODELS_DIR, f'model_{algorithm}.pkl')
            encoders_path = os.path.join(settings.MODELS_DIR, 'label_encoders.pkl')
            features_path = os.path.join(settings.MODELS_DIR, 'feature_columns.pkl')
            results_path = os.path.join(settings.MODELS_DIR, 'training_results.pkl')
            
            self.algorithm = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            self.feature_columns = joblib.load(features_path)
            self.training_results = joblib.load(results_path)
            
            self.current_algorithm = algorithm
            self.is_trained_flag = True
            
            logger.info(f"Model loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"Model load error: {str(e)}", exc_info=True)
            raise
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def is_trained(self) -> bool:
        """Model eğitilmiş mi kontrol eder"""
        return self.is_trained_flag
    
    def get_metrics(self) -> Dict:
        """Eğitim metriklerini döndürür"""
        if not self.is_trained_flag:
            raise ValueError("Model henüz eğitilmemiş")
        return self.training_results
    
    def get_feature_importance(self) -> Optional[Dict]:
        """Feature importance döndürür (destekliyorsa)"""
        if not self.is_trained_flag:
            return None
        
        try:
            # Sadece tree-based modeller için
            if hasattr(self.algorithm.model, 'estimators_'):
                importances = []
                for estimator in self.algorithm.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    feature_importance = dict(zip(self.feature_columns, avg_importance))
                    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return None
        
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return None