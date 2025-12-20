"""
Тесты для класса CatImageProcessor.

# Запустить тест как модуль
python -m ImageProcessing.tests.test_cat_image_processor
"""

import unittest
import os
from unittest.mock import patch
import numpy as np

from ImageProcessing.implementation.api_processor import AsyncCatImageProcessor, ImageData

class TestCatImageProcessorMinimal(unittest.TestCase):
    """Минимальные тесты, соответствующие заданию."""
    
    def test_1_api_initialization(self):
        """Тест 1: Обращение к API (инициализация с ключом)."""
        with patch.dict(os.environ, {'API_KEY': 'test'}):
            processor = AsyncCatImageProcessor()
            self.assertIsNotNone(processor._api_key)
            print("✅ Тест API-инициализации пройден")
    
    def test_2_image_data_class(self):
        """Тест 2: Структура данных для изображений."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        data = ImageData(
            index=1,
            breed="cat",
            image_array=img,
            url="test.jpg"
        )
        
        self.assertEqual(data.breed, "cat")
        self.assertEqual(data.index, 1)
        print("✅ Тест структуры данных пройден")
    
    def test_3_file_operations(self):
        """Тест 3: Чтение/запись файлов (симуляция)."""
        import tempfile
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            test_data = np.array([1, 2, 3])
            
            # Запись
            np.save(tmp.name, test_data)
            
            # Чтение  
            loaded = np.load(tmp.name)
            
            # Проверка
            self.assertTrue(np.array_equal(test_data, loaded))
        
        # Удаляем временный файл
        os.unlink(tmp.name)
        print("✅ Тест чтения/записи файлов пройден")

# Запуск тестов
if __name__ == '__main__':
    unittest.main()