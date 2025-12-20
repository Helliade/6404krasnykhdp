#!/usr/bin/env python3
"""
Запуск всех тестов проекта.
"""

import unittest
import sys
import os
import logging
import ImageProcessing

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Настройка логирования для тестов
    import logging
    from ImageProcessing.logging_config import setup_logging
    setup_logging()
    
    # Отключаем подробное логирование во время тестов
    logging.getLogger().setLevel(logging.WARNING)
    
    # Запуск тестов
    loader = unittest.TestLoader()
    
    # Находим все тесты в папке tests
    start_dir = os.path.join(os.path.dirname(__file__), 'ImageProcessing', 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Возвращаем код ошибки если тесты не прошли
    sys.exit(0 if result.wasSuccessful() else 1)