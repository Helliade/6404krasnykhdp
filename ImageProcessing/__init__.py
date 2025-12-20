"""
Пакет ImageProcessing для обработки изображений кошек.

Основные компоненты:
- ImageProcessing: класс для обработки изображений
- CatImage: класс для работы с изображениями кошек  
- AsyncCatImageProcessor: асинхронный процессор через API
"""

__version__ = '1.0.0'
__author__ = 'Красных Дарья 6404'

# Экспортируем основные классы для удобного импорта
from .implementation.image_processing import ImageProcessing
from .implementation.cat_image import CatImage, ColorCatImage, GrayscaleCatImage
from .implementation.api_processor import AsyncCatImageProcessor, CatImageProcessor
from .interfaces.i_image_processing import IImageProcessing, ICatImage
from .decorators import timer_decorator, async_timer_decorator
from .logging_config import setup_logging

__all__ = [
    'ImageProcessing',
    'CatImage',
    'ColorCatImage',
    'GrayscaleCatImage',
    'AsyncCatImageProcessor',
    'CatImageProcessor',
    'IImageProcessing',
    'ICatImage',
    'timer_decorator',
    'async_timer_decorator',
    'setup_logging'
]