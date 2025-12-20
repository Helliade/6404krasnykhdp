"""
Модуль конфигурации логирования из JSON файла.
"""

import os
import json
import logging.config
from pathlib import Path

def setup_logging():
    """
    Настраивает логирование из JSON конфигурации.
    """
    # Создаем директорию для логов
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    
    # Путь к JSON конфигурации
    config_path = Path(__file__).parent / "logging_config.json"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Применяем конфигурацию
        logging.config.dictConfig(config)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Логирование настроено из {config_path}")
    else:
        # Fallback: базовая настройка если JSON нет
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Файл {config_path} не найден, используется базовая конфигурация")
    
    return logging.getLogger()

def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер с указанным именем.
    """
    return logging.getLogger(name)

# Инициализация
logger = setup_logging()