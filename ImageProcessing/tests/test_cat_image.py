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
    
    def test_breed_is_correct(self):
        """Тест: порода корректно сохраняется."""
        self.assertEqual(self.cat_image.breed, self.breed)
    
    def test_image_url_is_correct(self):
        """Тест: URL изображения корректно сохраняется."""
        self.assertEqual(self.cat_image.image_url, self.url)
    
    def test_index_is_correct(self):
        """Тест: индекс корректно сохраняется."""
        self.assertEqual(self.cat_image.index, 1)
    
    def test_image_array_is_correct(self):
        """Тест: массив изображения корректно сохраняется."""
        self.assertTrue(np.array_equal(self.cat_image.image_array, self.rgb_image))
    
    def test_grayscale_conversion_produces_2d_array(self):
        """Тест: GrayscaleCatImage создает 2D массив."""
        gray_cat = GrayscaleCatImage(
            image_array=self.rgb_image,
            breed=self.breed,
            image_url=self.url
        )
        self.assertEqual(len(gray_cat.image_array.shape), 2)
    
    def test_edge_detection_returns_ndarray(self):
        """Тест: detect_edges_custom возвращает np.ndarray."""
        edges = self.cat_image.detect_edges_custom()
        self.assertIsInstance(edges, np.ndarray)
    
    def test_edge_detection_returns_2d_array(self):
        """Тест: detect_edges_custom возвращает 2D массив."""
        edges = self.cat_image.detect_edges_custom()
        self.assertEqual(len(edges.shape), 2)
    
    def test_edge_detection_preserves_height(self):
        """Тест: высота сохраняется при детекции границ."""
        edges = self.cat_image.detect_edges_custom()
        self.assertEqual(edges.shape[0], self.rgb_image.shape[0])
    
    def test_edge_detection_preserves_width(self):
        """Тест: ширина сохраняется при детекции границ."""
        edges = self.cat_image.detect_edges_custom()
        self.assertEqual(edges.shape[1], self.rgb_image.shape[1])
    
    def test_addition_returns_ndarray(self):
        """Тест: сложение возвращает np.ndarray."""
        other_image = CatImage(
            image_array=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            breed="other_breed",
            image_url="http://example.com/cat2.jpg",
            index=2
        )
        result = self.cat_image + other_image
        self.assertIsInstance(result, np.ndarray)
    
    def test_addition_preserves_shape(self):
        """Тест: сложение сохраняет форму."""
        other_image = CatImage(
            image_array=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            breed="other_breed",
            image_url="http://example.com/cat2.jpg",
            index=2
        )
        result = self.cat_image + other_image
        self.assertEqual(result.shape, self.rgb_image.shape)
    
    def test_addition_values_within_range(self):
        """Тест: значения после сложения не превышают 255."""
        other_image = CatImage(
            image_array=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            breed="other_breed",
            image_url="http://example.com/cat2.jpg",
            index=2
        )
        result = self.cat_image + other_image
        self.assertTrue(np.all(result <= 255))
    
    def test_str_contains_breed(self):
        """Тест: строковое представление содержит породу."""
        str_repr = str(self.cat_image)
        self.assertIn(self.breed, str_repr)
    
    def test_str_contains_shape(self):
        """Тест: строковое представление содержит форму."""
        str_repr = str(self.cat_image)
        self.assertIn(str(self.rgb_image.shape), str_repr)
    
    def test_str_contains_url(self):
        """Тест: строковое представление содержит URL."""
        str_repr = str(self.cat_image)
        self.assertIn(self.url, str_repr)

if __name__ == '__main__':
    unittest.main()