"""
Тесты для класса CatImage.

Запуск:
cd D:\TehProg_Py\6404krasnykhdp
python -m ImageProcessing.tests.test_cat_image
"""

import unittest
import numpy as np

from ImageProcessing.implementation.cat_image import CatImage, ColorCatImage, GrayscaleCatImage

class TestCatImage(unittest.TestCase):
    """Тестирование функциональности CatImage."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.rgb_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        self.breed = "test_breed"
        self.url = "http://example.com/cat.jpg"
        
        self.cat_image = CatImage(
            image_array=self.rgb_image,
            breed=self.breed,
            image_url=self.url,
            index=1
        )
    
    def test_initialization(self):
        """Тест инициализации CatImage."""
        self.assertEqual(self.cat_image.breed, self.breed)
        self.assertEqual(self.cat_image.image_url, self.url)
        self.assertEqual(self.cat_image.index, 1)
        self.assertTrue(np.array_equal(self.cat_image.image_array, self.rgb_image))
    
    def test_rgb_to_bw_conversion(self):
        """Тест преобразования RGB в черно-белое через GrayscaleCatImage."""
        # Создаем GrayscaleCatImage из RGB
        gray_cat = GrayscaleCatImage(
            image_array=self.rgb_image,
            breed=self.breed,
            image_url=self.url
        )
        
        # Проверяем что изображение стало 2D (черно-белым)
        self.assertEqual(len(gray_cat.image_array.shape), 2)
    
    def test_convolution(self):
        """Тест операции свертки через детекторы границ."""
        edges = self.cat_image.detect_edges_custom()
        
        # Проверяем что свертка выполняется и возвращает корректную форму
        self.assertIsInstance(edges, np.ndarray)
        self.assertEqual(len(edges.shape), 2)  # 2D после обработки
        self.assertEqual(edges.shape[0], self.rgb_image.shape[0])
        self.assertEqual(edges.shape[1], self.rgb_image.shape[1])
    
    def test_addition(self):
        """Тест сложения изображений."""
        other_image = CatImage(
            image_array=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            breed="other_breed",
            image_url="http://example.com/cat2.jpg",
            index=2
        )
        
        result = self.cat_image + other_image
        
        # Проверяем результат сложения
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.rgb_image.shape)
        self.assertTrue(np.all(result <= 255))
    
    def test_str_representation(self):
        """Тест строкового представления."""
        str_repr = str(self.cat_image)
        self.assertIn(self.breed, str_repr)
        self.assertIn(str(self.rgb_image.shape), str_repr)
        self.assertIn(self.url, str_repr)

if __name__ == '__main__':
    unittest.main()