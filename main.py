"""
ML Pipeline FastAPI Service
============================
Tüm ML pipeline işlemlerini tek bir serviste yöneten ana uygulama.

Endpoints:
- POST /upload - Excel dosyası yükleme
- POST /config - Algoritma yapılandırması
- POST /train - Model eğitimi
- POST /predict - Tahmin yapma
- GET /metrics - Performans metrikleri
- GET /health - Servis sağlık kontrolü
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional, List
import pandas as pd
import os
import shutil
from datetime import datetime

from models import (
    ConfigRequest, ConfigResponse,
    TrainRequest, TrainResponse,
    PredictionInput, PredictionResponse,
    MetricsResponse, HealthResponse
)
from ml_pipeline import MLPipeline
from logger import setup_logger, log_request
from config import settings

# FastAPI app oluştur
app = FastAPI(
    title="ML Pipeline API",
    description="Excel verisi ile çoklu ML algoritması destekleyen tahmin servisi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Logger setup
logger = setup_logger(__name__)

# Global ML Pipeline instance
ml_pipeline: Optional[MLPipeline] = None
current_data: Optional[pd.DataFrame] = None
uploaded_filename: Optional[str] = None


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    global ml_pipeline
    
    logger.info("=" * 80)
    logger.info("ML Pipeline Service başlatılıyor...")
    logger.info("=" * 80)
    
    # Gerekli dizinleri oluştur
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    # ML Pipeline'ı başlat
    ml_pipeline = MLPipeline()
    
    logger.info(f"Data dizini: {settings.DATA_DIR}")
    logger.info(f"Model dizini: {settings.MODELS_DIR}")
    logger.info(f"Log dizini: {settings.LOGS_DIR}")
    logger.info("Servis hazır!")


@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında çalışır"""
    logger.info("ML Pipeline Service kapatılıyor...")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Servis sağlık kontrolü
    
    Returns:
        HealthResponse: Servis durumu bilgileri
        
    Tip: Mevcut algoritmaları görmek için /algorithms endpoint'ini kullanın
    """
    log_request("GET", "/health")
    
    has_data = current_data is not None
    has_model = ml_pipeline.is_trained() if ml_pipeline else False
    
    return HealthResponse(
        status="healthy",
        service="ML Pipeline API",
        version="1.0.0",
        data_loaded=has_data,
        model_trained=has_model,
        current_algorithm=ml_pipeline.current_algorithm if ml_pipeline else None,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# DATA UPLOAD ENDPOINT
# ============================================================================

@app.post("/upload", tags=["Data Management"])
async def upload_data(file: UploadFile = File(...)):
    """
    Excel dosyası yükleme endpoint'i
    
    Args:
        file: Excel dosyası (.xlsx veya .xls)
    
    Returns:
        dict: Yükleme durumu ve veri bilgileri
    
    Raises:
        HTTPException: Dosya formatı geçersizse veya yükleme başarısızsa
    """
    global current_data, uploaded_filename
    
    log_request("POST", "/upload", {"filename": file.filename})
    
    try:
        # Dosya uzantısını kontrol et
        if not file.filename.endswith(('.xlsx', '.xls')):
            logger.error(f"Geçersiz dosya formatı: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sadece Excel dosyaları (.xlsx, .xls) kabul edilir"
            )
        
        # Dosyayı geçici olarak kaydet
        temp_path = os.path.join(settings.DATA_DIR, file.filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Dosya kaydedildi: {temp_path}")
        
        # Excel dosyasını oku
        df = pd.read_excel(temp_path)
        logger.info(f"Excel okundu: {df.shape[0]} satır, {df.shape[1]} sütun")
        
        # Zorunlu kolonları kontrol et
        required_columns = [
            'company_code', 'document_number', 'description',
            'payment_type', 'amount', 'transaction_type'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Eksik kolonlar: {missing_columns}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Eksik kolonlar: {', '.join(missing_columns)}"
            )
        
        # Hedef değişkenlerin varlığını kontrol et
        target_columns = ['seller_number', 'customer_number', 'main_account']
        has_targets = all(col in df.columns for col in target_columns)
        
        # Global değişkenlere kaydet
        current_data = df
        uploaded_filename = file.filename
        
        logger.info(f"Veri başarıyla yüklendi: {file.filename}")
        logger.info(f"Hedef değişkenler mevcut: {has_targets}")
        
        return {
            "status": "success",
            "message": f"Veri başarıyla yüklendi: {file.filename}",
            "data_info": {
                "filename": file.filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "has_targets": has_targets,
                "columns_list": df.columns.tolist()
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Veri yükleme hatası: {str(e)}"
        )


# ============================================================================
# CONFIG ENDPOINT
# ============================================================================

@app.get("/algorithms", tags=["Configuration"])
async def get_available_algorithms():
    """
    Mevcut algoritmaları ve varsayılan parametrelerini döndürür
    
    Returns:
        dict: Algoritma listesi ve her birinin parametreleri
    """
    log_request("GET", "/algorithms")
    
    algorithms_info = {}
    
    for algo in settings.AVAILABLE_ALGORITHMS:
        algorithms_info[algo] = {
            "name": algo.replace("_", " ").title(),
            "default_parameters": settings.get_algorithm_params(algo),
            "description": {
                "random_forest": "Ensemble learning yöntemi, yüksek doğruluk, overfitting'e dirençli",
                "gradient_boosting": "Güçlü gradient-based öğrenme, karmaşık ilişkileri yakalar",
                "logistic_regression": "Basit ve hızlı, yorumlanabilir sonuçlar",
                "neural_network": "Derin öğrenme, karmaşık pattern'ler için"
            }.get(algo, "")
        }
    
    return {
        "status": "success",
        "available_algorithms": list(settings.AVAILABLE_ALGORITHMS),
        "algorithms": algorithms_info,
        "default": settings.DEFAULT_ALGORITHM
    }


@app.post("/config", response_model=ConfigResponse, tags=["Configuration"])
async def configure_algorithm(config: ConfigRequest):
    """
    ML algoritması yapılandırma endpoint'i
    
    Args:
        config: Algoritma seçimi ve parametreleri
    
    Returns:
        ConfigResponse: Yapılandırma durumu
    
    Raises:
        HTTPException: Pipeline hazır değilse veya geçersiz yapılandırma
    """
    log_request("POST", "/config", config.dict())
    
    try:
        if ml_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ML Pipeline başlatılamadı"
            )
        
        # Algoritma yapılandırmasını kaydet
        success = ml_pipeline.configure(
            algorithm=config.algorithm,
            parameters=config.parameters
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz algoritma yapılandırması"
            )
        
        logger.info(f"Algoritma yapılandırıldı: {config.algorithm}")
        
        return ConfigResponse(
            status="success",
            message=f"Algoritma başarıyla yapılandırıldı: {config.algorithm}",
            algorithm=config.algorithm,
            parameters=config.parameters
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Yapılandırma hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Yapılandırma hatası: {str(e)}"
        )


# ============================================================================
# TRAIN ENDPOINT
# ============================================================================

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_model(request: TrainRequest):
    """
    Model eğitimi endpoint'i
    
    Args:
        request: Eğitim parametreleri
    
    Returns:
        TrainResponse: Eğitim sonuçları ve metrikleri
    
    Raises:
        HTTPException: Veri yüklenmemişse veya eğitim başarısızsa
    """
    log_request("POST", "/train", request.dict())
    
    try:
        # Veri kontrolü
        if current_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce veri yüklemelisiniz (/upload endpoint)"
            )
        
        if ml_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ML Pipeline başlatılamadı"
            )
        
        if ml_pipeline.current_algorithm is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce algoritma yapılandırması yapmalısınız (/config endpoint)"
            )
        
        logger.info("=" * 80)
        logger.info("MODEL EĞİTİMİ BAŞLIYOR")
        logger.info("=" * 80)
        logger.info(f"Algoritma: {ml_pipeline.current_algorithm}")
        logger.info(f"Test size: {request.test_size}")
        
        # Model eğitimi
        results = ml_pipeline.train(
            data=current_data,
            test_size=request.test_size
        )
        
        logger.info("=" * 80)
        logger.info("MODEL EĞİTİMİ TAMAMLANDI")
        logger.info("=" * 80)
        
        return TrainResponse(
            status="success",
            message=f"Model başarıyla eğitildi: {ml_pipeline.current_algorithm}",
            algorithm=ml_pipeline.current_algorithm,
            results=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eğitim hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Eğitim hatası: {str(e)}"
        )


# ============================================================================
# PREDICT ENDPOINT
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Tahmin yapma endpoint'i
    
    Args:
        input_data: Tahmin için girdi verileri
    
    Returns:
        PredictionResponse: Tahmin sonuçları
    
    Raises:
        HTTPException: Model eğitilmemişse veya tahmin başarısızsa
    """
    log_request("POST", "/predict", input_data.dict())
    
    try:
        if ml_pipeline is None or not ml_pipeline.is_trained():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce model eğitilmelidir (/train endpoint)"
            )
        
        # Tahmin yap
        predictions = ml_pipeline.predict(input_data)
        
        logger.info(f"Tahmin yapıldı: {predictions}")
        
        return PredictionResponse(
            status="success",
            predictions=predictions,
            algorithm_used=ml_pipeline.current_algorithm
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin hatası: {str(e)}"
        )


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """
    Model performans metrikleri endpoint'i
    
    Returns:
        MetricsResponse: Eğitim ve test performans metrikleri
    
    Raises:
        HTTPException: Model eğitilmemişse
    """
    log_request("GET", "/metrics")
    
    try:
        if ml_pipeline is None or not ml_pipeline.is_trained():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model henüz eğitilmemiş. Önce /train endpoint'ini kullanın."
            )
        
        metrics = ml_pipeline.get_metrics()
        
        return MetricsResponse(
            status="success",
            algorithm=ml_pipeline.current_algorithm,
            metrics=metrics
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrik alma hatası: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrik alma hatası: {str(e)}"
        )


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception'ları yakalar ve loglar"""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "error_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Genel exception'ları yakalar"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )