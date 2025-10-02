"""
Configuration Management
========================
Uygulama ayarları ve yapılandırma yönetimi
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Uygulama ayarları
    
    Environment variable'lardan veya .env dosyasından okunur
    """
    
    # Application
    APP_NAME: str = Field(default="ML Pipeline API", description="Uygulama adı")
    APP_VERSION: str = Field(default="1.0.0", description="Uygulama versiyonu")
    DEBUG: bool = Field(default=False, description="Debug modu")
    
    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Uvicorn worker sayısı")
    
    # Directories
    BASE_DIR: str = Field(default=os.getcwd(), description="Ana dizin")
    DATA_DIR: str = Field(default="data", description="Veri dizini")
    MODELS_DIR: str = Field(default="models", description="Model dizini")
    LOGS_DIR: str = Field(default="logs", description="Log dizini")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Log seviyesi")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log formatı"
    )
    LOG_FILE: str = Field(default="app.log", description="Log dosya adı")
    
    # ML Settings
    DEFAULT_TEST_SIZE: float = Field(default=0.25, ge=0.1, le=0.5)
    DEFAULT_RANDOM_STATE: int = Field(default=42)
    MAX_BULK_PREDICTIONS: int = Field(default=1000, description="Maksimum toplu tahmin sayısı")
    
    # File Upload
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024, description="Maksimum dosya boyutu (bytes)")
    ALLOWED_EXTENSIONS: list = Field(default=[".xlsx", ".xls"], description="İzin verilen dosya uzantıları")
    
    # Model Algorithms
    DEFAULT_ALGORITHM: str = Field(default="random_forest", description="Varsayılan algoritma")
    AVAILABLE_ALGORITHMS: list = Field(
        default=["random_forest", "gradient_boosting", "logistic_regression", "neural_network"],
        description="Mevcut algoritmalar"
    )
    
    # Algorithm Parameters
    RF_N_ESTIMATORS: int = Field(default=150, description="Random Forest - estimator sayısı")
    RF_MAX_DEPTH: int = Field(default=12, description="Random Forest - maksimum derinlik")
    RF_MIN_SAMPLES_SPLIT: int = Field(default=5, description="Random Forest - minimum split")
    
    GB_N_ESTIMATORS: int = Field(default=100, description="Gradient Boosting - estimator sayısı")
    GB_MAX_DEPTH: int = Field(default=5, description="Gradient Boosting - maksimum derinlik")
    GB_LEARNING_RATE: float = Field(default=0.1, description="Gradient Boosting - learning rate")
    
    LR_MAX_ITER: int = Field(default=1000, description="Logistic Regression - maksimum iterasyon")
    
    NN_HIDDEN_LAYERS: tuple = Field(default=(100, 50), description="Neural Network - hidden layer boyutları")
    NN_MAX_ITER: int = Field(default=500, description="Neural Network - maksimum iterasyon")
    NN_ACTIVATION: str = Field(default="relu", description="Neural Network - aktivasyon fonksiyonu")
    
    # Data Split Settings (Case Requirement)
    USE_FIXED_SPLIT: bool = Field(default=True, description="Sabit train/test split kullan")
    FIXED_TRAIN_SIZE: int = Field(default=150, description="Sabit train set boyutu")
    FIXED_TEST_SIZE: int = Field(default=50, description="Sabit test set boyutu")
    
    # Feature Engineering
    FEATURE_SELECTION: bool = Field(default=False, description="Feature selection aktif mi")
    FEATURE_SELECTION_THRESHOLD: float = Field(default=0.01, description="Feature selection eşik değeri")
    
    # Performance
    ENABLE_CACHING: bool = Field(default=False, description="Cache mekanizması")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL (saniye)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_algorithm_params(self, algorithm: str) -> dict:
        """
        Belirtilen algoritma için varsayılan parametreleri döndürür
        
        Args:
            algorithm: Algoritma adı
        
        Returns:
            dict: Algoritma parametreleri
        """
        params = {
            "random_forest": {
                "n_estimators": self.RF_N_ESTIMATORS,
                "max_depth": self.RF_MAX_DEPTH,
                "min_samples_split": self.RF_MIN_SAMPLES_SPLIT,
                "random_state": self.DEFAULT_RANDOM_STATE,
                "n_jobs": -1,
                "class_weight": "balanced"
            },
            "gradient_boosting": {
                "n_estimators": self.GB_N_ESTIMATORS,
                "max_depth": self.GB_MAX_DEPTH,
                "learning_rate": self.GB_LEARNING_RATE,
                "random_state": self.DEFAULT_RANDOM_STATE
            },
            "logistic_regression": {
                "max_iter": self.LR_MAX_ITER,
                "random_state": self.DEFAULT_RANDOM_STATE,
                "class_weight": "balanced"
            },
            "neural_network": {
                "hidden_layer_sizes": self.NN_HIDDEN_LAYERS,
                "activation": self.NN_ACTIVATION,
                "max_iter": self.NN_MAX_ITER,
                "random_state": self.DEFAULT_RANDOM_STATE,
                "early_stopping": True
            }
        }
        
        return params.get(algorithm, {})
    
    def get_full_path(self, relative_path: str) -> str:
        """
        Relative path'i full path'e çevirir
        
        Args:
            relative_path: Relative path
        
        Returns:
            str: Full path
        """
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.BASE_DIR, relative_path)
    
    @property
    def data_path(self) -> str:
        """Data dizininin full path'i"""
        return self.get_full_path(self.DATA_DIR)
    
    @property
    def models_path(self) -> str:
        """Models dizininin full path'i"""
        return self.get_full_path(self.MODELS_DIR)
    
    @property
    def logs_path(self) -> str:
        """Logs dizininin full path'i"""
        return self.get_full_path(self.LOGS_DIR)
    
    @property
    def log_file_path(self) -> str:
        """Log dosyasının full path'i"""
        return os.path.join(self.logs_path, self.LOG_FILE)
    
    def validate_algorithm(self, algorithm: str) -> bool:
        """
        Algoritma adının geçerli olup olmadığını kontrol eder
        
        Args:
            algorithm: Algoritma adı
        
        Returns:
            bool: Geçerli ise True
        """
        return algorithm in self.AVAILABLE_ALGORITHMS
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        Dosya uzantısının geçerli olup olmadığını kontrol eder
        
        Args:
            filename: Dosya adı
        
        Returns:
            bool: Geçerli ise True
        """
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.ALLOWED_EXTENSIONS


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Settings instance'ını döndürür
    
    Returns:
        Settings: Global settings object
    """
    return settings


def update_settings(**kwargs):
    """
    Settings'i runtime'da günceller
    
    Args:
        **kwargs: Güncellenecek ayarlar
    """
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


# Startup'ta dizinleri oluştur
def create_directories():
    """Gerekli dizinleri oluşturur"""
    os.makedirs(settings.data_path, exist_ok=True)
    os.makedirs(settings.models_path, exist_ok=True)
    os.makedirs(settings.logs_path, exist_ok=True)


if __name__ == "__main__":
    # Test için settings'i yazdır
    print("=" * 80)
    print("CONFIGURATION SETTINGS")
    print("=" * 80)
    print(f"App Name: {settings.APP_NAME}")
    print(f"Version: {settings.APP_VERSION}")
    print(f"Host: {settings.HOST}:{settings.PORT}")
    print(f"Data Dir: {settings.data_path}")
    print(f"Models Dir: {settings.models_path}")
    print(f"Logs Dir: {settings.logs_path}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Default Algorithm: {settings.DEFAULT_ALGORITHM}")
    print(f"Available Algorithms: {settings.AVAILABLE_ALGORITHMS}")
    print("=" * 80)

    # Algoritma parametrelerini göster
    print("\nALGORITHM PARAMETERS")
    print("=" * 80)
    for algo in settings.AVAILABLE_ALGORITHMS:
        print(f"\n{algo.upper()}:")
        params = settings.get_algorithm_params(algo)
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Dizinleri oluştur
    create_directories()
    print("\n✓ Directories created!")