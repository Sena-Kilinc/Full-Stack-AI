"""
Logging System
==============
Profesyonel loglama sistemi - console ve file logging
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

from config import settings


class ColoredFormatter(logging.Formatter):
    """
    Renkli console output için custom formatter
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Renk ekle
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    JSON formatında log için custom formatter
    """
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Ek alanlar ekle
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Eğer exception varsa ekle
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Logger oluşturur ve yapılandırır
    
    Args:
        name: Logger adı
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log dosya yolu (None ise sadece console)
        json_format: JSON formatında log mu
    
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    
    # Logger oluştur
    logger = logging.getLogger(name)
    
    # Eğer zaten handler varsa, tekrar ekleme
    if logger.handlers:
        return logger
    
    # Log level
    level = log_level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if json_format:
        console_formatter = CustomJsonFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file or settings.LOG_FILE:
        file_path = log_file or settings.log_file_path
        
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Rotating file handler (max 10MB, 5 backup)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_formatter = CustomJsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Propagate'i kapat (parent logger'a gitmesin)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Mevcut logger'ı döndürür veya yeni oluşturur
    
    Args:
        name: Logger adı
    
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # Eğer handler yoksa setup et
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_request(method: str, endpoint: str, data: Optional[dict] = None):
    """
    API request'i loglar
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        data: Request data (opsiyonel)
    """
    logger = get_logger("api")
    
    log_message = f"{method} {endpoint}"
    
    if data:
        # Hassas bilgileri maskele
        masked_data = mask_sensitive_data(data)
        log_message += f" - Data: {masked_data}"
    
    logger.info(log_message)


def log_response(status_code: int, endpoint: str, response_time: Optional[float] = None):
    """
    API response'u loglar
    
    Args:
        status_code: HTTP status code
        endpoint: API endpoint
        response_time: Response süresi (saniye)
    """
    logger = get_logger("api")
    
    log_message = f"Response {status_code} - {endpoint}"
    
    if response_time:
        log_message += f" - {response_time:.3f}s"
    
    if status_code >= 400:
        logger.error(log_message)
    else:
        logger.info(log_message)


def log_exception(exception: Exception, context: Optional[str] = None):
    """
    Exception'ı detaylı şekilde loglar
    
    Args:
        exception: Exception object
        context: Ek context bilgisi
    """
    logger = get_logger("errors")
    
    log_message = f"Exception: {type(exception).__name__} - {str(exception)}"
    
    if context:
        log_message = f"{context} - {log_message}"
    
    logger.error(log_message, exc_info=True)


def mask_sensitive_data(data: dict) -> dict:
    """
    Hassas bilgileri maskeler
    
    Args:
        data: Maskelenecek data
    
    Returns:
        dict: Maskelenmiş data
    """
    sensitive_fields = [
        'password', 'token', 'api_key', 'secret',
        'credit_card', 'ssn', 'pin'
    ]
    
    masked_data = data.copy()
    
    for key in masked_data:
        if any(sensitive in key.lower() for sensitive in sensitive_fields):
            masked_data[key] = "****MASKED****"
    
    return masked_data


class LogContext:
    """
    Context manager for logging blocks
    
    Usage:
        with LogContext("Training model"):
            # training code
    """
    
    def __init__(self, operation: str, logger_name: str = "app"):
        self.operation = operation
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({duration:.2f}s) - {exc_type.__name__}: {exc_val}",
                exc_info=True
            )
        
        return False  # Exception'ı propagate et


def log_performance(func):
    """
    Fonksiyon performansını loglayan decorator
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("performance")
        
        start_time = time.time()
        logger.debug(f"Starting: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Completed: {func.__name__} ({duration:.3f}s)")
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed: {func.__name__} ({duration:.3f}s) - {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise
    
    return wrapper


# ML Pipeline için özel loggerlar
ml_logger = setup_logger("ml_pipeline")
api_logger = setup_logger("api")
error_logger = setup_logger("errors")
performance_logger = setup_logger("performance")


if __name__ == "__main__":
    # Test logging
    print("=" * 80)
    print("TESTING LOGGING SYSTEM")
    print("=" * 80)
    
    # Test basic logger
    test_logger = setup_logger("test")
    
    test_logger.debug("This is a DEBUG message")
    test_logger.info("This is an INFO message")
    test_logger.warning("This is a WARNING message")
    test_logger.error("This is an ERROR message")
    test_logger.critical("This is a CRITICAL message")
    
    # Test request logging
    print("\n" + "=" * 80)
    print("TESTING REQUEST LOGGING")
    print("=" * 80)
    
    log_request("POST", "/train", {"test_size": 0.25, "password": "secret123"})
    log_response(200, "/train", 2.5)
    
    # Test context manager
    print("\n" + "=" * 80)
    print("TESTING CONTEXT MANAGER")
    print("=" * 80)
    
    with LogContext("Test operation"):
        import time
        time.sleep(1)
        print("Operation in progress...")
    
    # Test decorator
    print("\n" + "=" * 80)
    print("TESTING PERFORMANCE DECORATOR")
    print("=" * 80)
    
    @log_performance
    def test_function():
        import time
        time.sleep(0.5)
        return "Done"
    
    result = test_function()
    print(f"Result: {result}")
    
    print("\n" + "=" * 80)
    print("LOGGING TEST COMPLETED")
    print("=" * 80)