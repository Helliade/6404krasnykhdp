import numpy as np
from typing import Union, Optional
import cv2
from interfaces.i_image_processing import ICatImage
from implementation.image_processing import ImageProcessing

class CatImage(ICatImage):
    """Наследуемся от абстрактного базового класса для изображений по заданию из 2 ЛР."""
    
    def __init__(self, image_array: np.ndarray, breed: str, image_url: str, index: Optional[int] = None):
        self._image_array = image_array
        self._breed = breed
        self._image_url = image_url
        self._index = index  # Порядковый номер
        self._processor = ImageProcessing()
    
    @property
    def image_array(self) -> np.ndarray:
        return self._image_array
    
    @property
    def breed(self) -> str:
        return self._breed
    
    @property
    def image_url(self) -> str:
        return self._image_url

    @property
    def index(self) -> int:
       return self._index if self._index is not None else 0
    
    def detect_edges_custom(self) -> np.ndarray:
        """Пользовательская реализация выделения контуров."""
        return self._processor.edge_detection(self._image_array, variant="new")
    
    def detect_edges_library(self) -> np.ndarray:
        """Библиотечная реализация выделения контуров."""
        return self._processor.edge_detection(self._image_array, variant="old")
    
    def __add__(self, other: 'CatImage') -> np.ndarray:
        """Сложение с другим CatImage или numpy array."""
        other_array = self._get_other_array(other)
        self._validate_shapes(self._image_array, other_array)
        return cv2.add(self._image_array, other_array)
    
    def __sub__(self, other: 'CatImage') -> np.ndarray:
        """Вычитание другого CatImage или numpy array."""
        other_array = self._get_other_array(other)
        self._validate_shapes(self._image_array, other_array)
        return cv2.subtract(self._image_array, other_array)
    
    def _get_other_array(self, other: Union['CatImage', np.ndarray]) -> np.ndarray:
        """Извлекает массив из other в зависимости от типа."""
        if isinstance(other, CatImage):
            return other.image_array
        elif isinstance(other, np.ndarray):
            return other
        else:
            raise TypeError(f"Неподдерживаемый тип: {type(other)}. Ожидается CatImage или numpy array")
    
    def _validate_shapes(self, array1: np.ndarray, array2: np.ndarray):
        """Проверяет совместимость размеров массивов."""
        if array1.shape != array2.shape:
            raise ValueError(
                f"Несовместимые размеры: {array1.shape} vs {array2.shape}. "
                "Изображения должны иметь одинаковые размеры")

    def __str__(self) -> str:
        return f"CatImage(breed={self._breed}, shape={self._image_array.shape}, url={self._image_url})"

class ColorCatImage(CatImage):
    """Реализация для цветных изображений."""
    def __init__(self, image_array: np.ndarray, breed: str, image_url: str):
        # Конвертируем в оттенки серого при инициализации
        if len(image_array.shape) == 1:
            gray_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        else:
            gray_array = image_array
        super().__init__(gray_array, breed, image_url)

class GrayscaleCatImage(CatImage):
    """Реализация для черно-белых изображений."""
    
    def __init__(self, image_array: np.ndarray, breed: str, image_url: str):
        # Конвертируем в оттенки серого при инициализации
        processor = ImageProcessing()
        if len(image_array.shape) == 3:
            gray_array = processor.rgb_to_grayscale(image_array, variant="new")
        else:
            gray_array = image_array
        super().__init__(gray_array, breed, image_url)