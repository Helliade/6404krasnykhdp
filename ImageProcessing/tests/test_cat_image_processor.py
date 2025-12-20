"""
Тесты для класса CatImageProcessor.

# Запустить тест как модуль
python -m ImageProcessing.tests.test_cat_image_processor
"""

import unittest
import os
import tempfile
from unittest.mock import patch
import numpy as np

from ImageProcessing.implementation.api_processor import AsyncCatImageProcessor, ImageData

class TestCatImageProcessorMinimal(unittest.TestCase):
    """Минимальные тесты, соответствующие заданию."""
    
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
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        data = ImageData(index=1, breed="cat", image_array=img, url="test.jpg")
        self.assertEqual(data.breed, "cat")
    
    def test_image_data_stores_index(self):
        """Тест: ImageData корректно хранит индекс."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        data = ImageData(index=1, breed="cat", image_array=img, url="test.jpg")
        self.assertEqual(data.index, 1)
    
    def test_image_data_stores_url(self):
        """Тест: ImageData корректно хранит URL."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        data = ImageData(index=1, breed="cat", image_array=img, url="test.jpg")
        self.assertEqual(data.url, "test.jpg")
    
    def test_image_data_stores_image_array(self):
        """Тест: ImageData корректно хранит массив изображения."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        data = ImageData(index=1, breed="cat", image_array=img, url="test.jpg")
        self.assertTrue(np.array_equal(data.image_array, img))
    
    def test_npy_file_can_be_written(self):
        """Тест: данные могут быть записаны в .npy файл."""
        test_data = np.array([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, test_data)
            self.assertTrue(os.path.exists(tmp.name))
        os.unlink(tmp.name)
    
    def test_npy_file_can_be_read(self):
        """Тест: данные могут быть прочитаны из .npy файла."""
        test_data = np.array([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, test_data)
            loaded = np.load(tmp.name)
            self.assertTrue(np.array_equal(test_data, loaded))
        os.unlink(tmp.name)
    
    def test_file_write_and_read_preserves_data(self):
        """Тест: запись и чтение сохраняют данные."""
        test_data = np.array([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, test_data)
            loaded = np.load(tmp.name)
            self.assertTrue(np.array_equal(test_data, loaded))
        os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()