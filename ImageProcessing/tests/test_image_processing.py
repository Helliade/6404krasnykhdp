"""
Тесты для класса ImageProcessing.

# Запустить тест как модуль
python -m ImageProcessing.tests.test_image_processing
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
    
    def test_convolution_new(self):
        """Тест свертки (новая реализация)."""
        result = self.processor.convolution(self.rgb_image, self.kernel, variant="new")
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.rgb_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_rgb_to_grayscale_new(self):
        """Тест преобразования RGB в grayscale (новая реализация)."""
        result = self.processor.rgb_to_grayscale(self.rgb_image, variant="new")
        
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], self.rgb_image.shape[0])
        self.assertEqual(result.shape[1], self.rgb_image.shape[1])
        self.assertEqual(result.dtype, np.uint8)
    
    def test_edge_detection_new(self):
        """Тест детекции границ (новая реализация)."""
        result = self.processor.edge_detection(self.rgb_image, variant="new")
        
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.dtype, np.uint8)
        unique_values = np.unique(result)
        self.assertTrue(all(val in [0, 255] for val in unique_values))

if __name__ == '__main__':
    unittest.main()