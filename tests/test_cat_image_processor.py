"""
Тесты для класса CatImageProcessor.

# Запустить тест как модуль
python -m tests.test_cat_image_processor
"""

import unittest
import os
import tempfile
from unittest.mock import patch
import numpy as np

from ImageProcessing.implementation.api_processor import AsyncCatImageProcessor, ImageData

class TestCatImageProcessorMinimal(unittest.TestCase):
    """Минимальные тесты, соответствующие заданию."""
    
    def setUp(self):
        """Подготовка общих тестовых данных."""
        # Общие данные для тестов ImageData
        self.test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        self.test_breed = "cat"
        self.test_url = "test.jpg"
        self.test_index = 1
        
        # Общие данные для тестов файлов
        self.test_array_data = np.array([1, 2, 3])
        
        # Создаём ImageData для переиспользования
        self.image_data = ImageData(
            index=self.test_index,
            breed=self.test_breed,
            image_array=self.test_image,
            url=self.test_url
        )
    
    def test_api_key_is_set(self):
        """Тест: API ключ устанавливается при инициализации."""
        with patch.dict(os.environ, {'API_KEY': 'test_key'}):
            processor = AsyncCatImageProcessor()
            self.assertEqual(processor._api_key, 'test_key')
    
    def test_headers_contain_api_key(self):
        """Тест: заголовки содержат API ключ."""
        with patch.dict(os.environ, {'API_KEY': 'test_key'}):
            processor = AsyncCatImageProcessor()
            self.assertIn('x-api-key', processor._headers)
    
    def test_image_data_stores_breed(self):
        """Тест: ImageData корректно хранит породу."""
        self.assertEqual(self.image_data.breed, self.test_breed)
    
    def test_image_data_stores_index(self):
        """Тест: ImageData корректно хранит индекс."""
        self.assertEqual(self.image_data.index, self.test_index)
    
    def test_image_data_stores_url(self):
        """Тест: ImageData корректно хранит URL."""
        self.assertEqual(self.image_data.url, self.test_url)
    
    def test_image_data_stores_image_array(self):
        """Тест: ImageData корректно хранит массив изображения."""
        self.assertTrue(np.array_equal(self.image_data.image_array, self.test_image))
    
    def test_npy_file_can_be_written(self):
        """Тест: данные могут быть записаны в .npy файл."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, self.test_array_data)
            self.assertTrue(os.path.exists(tmp.name))
        os.unlink(tmp.name)
    
    def test_npy_file_can_be_read(self):
        """Тест: данные могут быть прочитаны из .npy файла."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, self.test_array_data)
            loaded = np.load(tmp.name)
            self.assertTrue(np.array_equal(self.test_array_data, loaded))
        os.unlink(tmp.name)
    
    def test_file_write_and_read_preserves_data(self):
        """Тест: запись и чтение сохраняют данные."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, self.test_array_data)
            loaded = np.load(tmp.name)
            self.assertTrue(np.array_equal(self.test_array_data, loaded))
        os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()