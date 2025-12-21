"""
Тесты для класса ImageProcessing.

# Запустить тест как модуль
python -m tests.test_image_processing
"""

import unittest
import numpy as np

from ImageProcessing.implementation.image_processing import ImageProcessing

class TestImageProcessing(unittest.TestCase):
    """Тестирование функциональности ImageProcessing."""
    
    def setUp(self):
        self.processor = ImageProcessing()
        self.rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    
    def test_convolution_new_returns_ndarray(self):
        """Тест: свертка возвращает np.ndarray."""
        result = self.processor.convolution(self.rgb_image, self.kernel, variant="new")
        self.assertIsInstance(result, np.ndarray)
    
    def test_convolution_new_has_correct_shape(self):
        """Тест: свертка сохраняет форму изображения."""
        result = self.processor.convolution(self.rgb_image, self.kernel, variant="new")
        self.assertEqual(result.shape, self.rgb_image.shape)
    
    def test_convolution_new_has_correct_dtype(self):
        """Тест: свертка возвращает uint8."""
        result = self.processor.convolution(self.rgb_image, self.kernel, variant="new")
        self.assertEqual(result.dtype, np.uint8)
    
    def test_rgb_to_grayscale_new_is_2d(self):
        """Тест: RGB to grayscale возвращает 2D массив."""
        result = self.processor.rgb_to_grayscale(self.rgb_image, variant="new")
        self.assertEqual(len(result.shape), 2)
    
    def test_rgb_to_grayscale_new_has_correct_height(self):
        """Тест: высота сохраняется при конвертации."""
        result = self.processor.rgb_to_grayscale(self.rgb_image, variant="new")
        self.assertEqual(result.shape[0], self.rgb_image.shape[0])
    
    def test_rgb_to_grayscale_new_has_correct_width(self):
        """Тест: ширина сохраняется при конвертации."""
        result = self.processor.rgb_to_grayscale(self.rgb_image, variant="new")
        self.assertEqual(result.shape[1], self.rgb_image.shape[1])
    
    def test_rgb_to_grayscale_new_has_uint8_dtype(self):
        """Тест: конвертация возвращает uint8."""
        result = self.processor.rgb_to_grayscale(self.rgb_image, variant="new")
        self.assertEqual(result.dtype, np.uint8)
    
    def test_edge_detection_new_is_2d(self):
        """Тест: детекция границ возвращает 2D массив."""
        result = self.processor.edge_detection(self.rgb_image, variant="new")
        self.assertEqual(len(result.shape), 2)
    
    def test_edge_detection_new_has_uint8_dtype(self):
        """Тест: детекция границ возвращает uint8."""
        result = self.processor.edge_detection(self.rgb_image, variant="new")
        self.assertEqual(result.dtype, np.uint8)
    
    def test_edge_detection_new_has_binary_values(self):
        """Тест: детекция границ возвращает бинарные значения (0 или 255)."""
        result = self.processor.edge_detection(self.rgb_image, variant="new")
        unique_values = np.unique(result)
        self.assertTrue(all(val in [0, 255] for val in unique_values))

if __name__ == '__main__':
    unittest.main()