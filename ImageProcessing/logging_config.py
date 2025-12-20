"""
Модуль конфигурации логирования.
Настройка логгера для всего приложения.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Создаем директорию для логов
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

def setup_logging():
    """
    Настраивает логирование для приложения.
    
    Конфигурация:
    - В файл app.log: уровень DEBUG с подробной информацией
    - В консоль: уровень INFO с краткой информацией
    """
    
    # Создаем форматтеры
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Создаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # 1. Обработчик для файла (DEBUG уровень)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # 2. Обработчик для консоли (INFO уровень)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Отключаем логирование от сторонних библиотек
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер с указанным именем.
    
    Args:
        name: Имя логгера (обычно __name__)
    
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)

# Инициализация при импорте
logger = setup_logging()