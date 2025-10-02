"""
Pydantic Models
===============
API için validation ve schema tanımları
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AlgorithmType(str, Enum):
    """
    Desteklenen ML algoritmaları
    
    - random_forest: Random Forest (Ensemble, yüksek doğruluk, overfitting'e dirençli)
    - gradient_boosting: Gradient Boosting (Güçlü, karmaşık ilişkileri yakalar)
    - logistic_regression: Logistic Regression (Basit, hızlı, yorumlanabilir)
    - neural_network: Neural Network/MLP (Derin öğrenme, karmaşık pattern'ler)
    """
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"


class PaymentType(str, Enum):
    """Ödeme tipleri"""
    S = "S"  # Satış
    H = "H"  # Alış


# ============================================================================
# CONFIG MODELS
# ============================================================================

class ConfigRequest(BaseModel):
    """
    Algoritma yapılandırma isteği
    
    Önce /algorithms endpoint'ini çağırarak mevcut algoritmaları 
    ve varsayılan parametrelerini öğrenebilirsiniz.
    """
    algorithm: AlgorithmType = Field(
        ...,
        description="Kullanılacak ML algoritması (random_forest, gradient_boosting, logistic_regression, neural_network)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Algoritma parametreleri (opsiyonel, boş bırakılırsa varsayılanlar kullanılır)"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "algorithm": "random_forest",
                    "parameters": {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "min_samples_split": 5
                    }
                },
                {
                    "algorithm": "gradient_boosting",
                    "parameters": {
                        "n_estimators": 150,
                        "learning_rate": 0.05
                    }
                },
                {
                    "algorithm": "neural_network",
                    "parameters": {
                        "hidden_layer_sizes": [100, 50, 25],
                        "activation": "relu"
                    }
                },
                {
                    "algorithm": "logistic_regression",
                    "parameters": None
                }
            ]
        }


class ConfigResponse(BaseModel):
    """Yapılandırma yanıtı"""
    status: str
    message: str
    algorithm: str
    parameters: Optional[Dict[str, Any]] = None


# ============================================================================
# TRAINING MODELS
# ============================================================================

class TrainRequest(BaseModel):
    """Model eğitim isteği"""
    test_size: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="Test set oranı (0.1 - 0.5 arası)"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Reproducibility için random seed"
    )
    
    @validator('test_size')
    def validate_test_size(cls, v):
        """Test size validasyonu"""
        if not 0.1 <= v <= 0.5:
            raise ValueError('test_size 0.1 ile 0.5 arasında olmalıdır')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_size": 0.25,
                "random_state": 42
            }
        }


class TargetMetrics(BaseModel):
    """Tek bir hedef değişken için metrikler"""
    accuracy: float = Field(..., ge=0, le=1)
    f1_weighted: float = Field(..., ge=0, le=1)
    f1_macro: float = Field(..., ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)


class TrainResponse(BaseModel):
    """Eğitim yanıtı"""
    status: str
    message: str
    algorithm: str
    results: Dict[str, Any]


# ============================================================================
# PREDICTION MODELS
# ============================================================================

class PredictionInput(BaseModel):
    """Tahmin için girdi verisi"""
    company_code: int = Field(
        ...,
        description="Şirket kodu",
        ge=1000,
        le=9999
    )
    document_number: str = Field(
        ...,
        description="Belge numarası",
        min_length=1,
        max_length=50
    )
    description: str = Field(
        ...,
        description="İşlem açıklaması",
        min_length=1,
        max_length=500
    )
    payment_type: PaymentType = Field(
        ...,
        description="Ödeme tipi (S: Satış, H: Alış)"
    )
    amount: float = Field(
        ...,
        description="İşlem tutarı",
        gt=0
    )
    currency_code: str = Field(
        default="TRY",
        description="Para birimi",
        min_length=3,
        max_length=3
    )
    transaction_type: str = Field(
        ...,
        description="İşlem tipi (NTRF, NEFT, NCHK, vb.)",
        min_length=1,
        max_length=10
    )
    
    @validator('company_code')
    def validate_company_code(cls, v):
        """Company code validasyonu"""
        if v < 1000 or v > 9999:
            raise ValueError('company_code 1000 ile 9999 arasında olmalıdır')
        return v
    
    @validator('amount')
    def validate_amount(cls, v):
        """Amount validasyonu"""
        if v <= 0:
            raise ValueError('amount pozitif olmalıdır')
        return round(v, 2)
    
    @validator('currency_code')
    def validate_currency(cls, v):
        """Currency code validasyonu"""
        v = v.upper()
        if len(v) != 3:
            raise ValueError('currency_code 3 karakter olmalıdır')
        return v
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        """Transaction type validasyonu"""
        v = v.upper()
        valid_types = ['NTRF', 'NEFT', 'NCHK', 'NCHG', 'NTAX']
        if v not in valid_types:
            raise ValueError(f'transaction_type şunlardan biri olmalıdır: {", ".join(valid_types)}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_code": 1504,
                "document_number": "4200004825",
                "description": "EF7789421YORUMAJANSPAZARLAMAITHALATVET-FATURABEDEL",
                "payment_type": "H",
                "amount": 400000.00,
                "currency_code": "TRY",
                "transaction_type": "NEFT"
            }
        }


class PredictionOutput(BaseModel):
    """Tahmin çıktısı"""
    seller_number: Optional[str] = None
    customer_number: Optional[str] = None
    main_account: Optional[str] = None


class PredictionResponse(BaseModel):
    """Tahmin yanıtı"""
    status: str
    predictions: PredictionOutput
    algorithm_used: str
    confidence: Optional[Dict[str, float]] = None


# ============================================================================
# BULK PREDICTION MODELS
# ============================================================================

class BulkPredictionInput(BaseModel):
    """Toplu tahmin isteği"""
    transactions: List[PredictionInput] = Field(
        ...,
        description="Tahmin yapılacak işlemler listesi",
        min_items=1,
        max_items=1000
    )
    
    @validator('transactions')
    def validate_transactions(cls, v):
        """Transaction listesi validasyonu"""
        if len(v) > 1000:
            raise ValueError('Maksimum 1000 işlem gönderilebilir')
        return v


class BulkPredictionResponse(BaseModel):
    """Toplu tahmin yanıtı"""
    status: str
    count: int
    predictions: List[PredictionOutput]
    algorithm_used: str


# ============================================================================
# METRICS MODELS
# ============================================================================

class MetricsResponse(BaseModel):
    """Performans metrikleri yanıtı"""
    status: str
    algorithm: str
    metrics: Dict[str, Any]


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı"""
    status: str
    service: str
    version: str
    data_loaded: bool
    model_trained: bool
    current_algorithm: Optional[str] = None
    timestamp: str


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    status: str = "error"
    message: str
    error_code: Optional[int] = None
    detail: Optional[str] = None
    timestamp: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Validation error",
                "error_code": 422,
                "detail": "company_code must be between 1000 and 9999"
            }
        }


# ============================================================================
# FILE UPLOAD MODELS
# ============================================================================

class DataInfo(BaseModel):
    """Yüklenen veri bilgileri"""
    filename: str
    rows: int
    columns: int
    has_targets: bool
    columns_list: List[str]


class UploadResponse(BaseModel):
    """Dosya yükleme yanıtı"""
    status: str
    message: str
    data_info: DataInfo