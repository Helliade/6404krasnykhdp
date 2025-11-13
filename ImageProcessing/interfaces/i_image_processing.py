"""
Модуль интерфейса i_image_processing.py

Содержит класс IImageProcessing, определяющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

from abc import ABC, abstractmethod


import numpy as np


class IImageProcessing(ABC):
    """
    Интерфейс для реализации методов обработки изображений.

    Определяет набор абстрактных методов, которые должны быть реализованы
    в наследуемых классах для выполнения различных операций над изображениями.
    """

    @abstractmethod
    def convolution(self, image: np.ndarray, kernel: np.ndarray, variant: str = "new") -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение.
            kernel (np.ndarray): Ядро свёртки.

        Returns:
            np.ndarray: Результат применения свёртки к изображению.
        """
        pass

    @abstractmethod
    def rgb_to_grayscale(self, image: np.ndarray, variant: str = "new") -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Изображение в оттенках серого.
        """
        pass

    @abstractmethod
    def gamma_correction(self, image: np.ndarray, gamma: float, variant: str = "new") -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Значение гамма-коррекции.

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray, variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными границами.
        """
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray, k: float = 0.04, threshold: float = 0.03, variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными углами.
        """
        pass

    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        pass

class ICatImage(ABC):
    """Интерфейс для класса CatImage из задания."""
    
    @abstractmethod
    def detect_edges_custom(self) -> np.ndarray:
        """Пользовательский метод выделения контуров."""
        pass
    
    @abstractmethod
    def detect_edges_library(self) -> np.ndarray:
        """Библиотечный метод выделения контуров."""
        pass
    
    @abstractmethod
    def __add__(self, other):
        """Сложение изображений."""
        pass
    
    @abstractmethod
    def __sub__(self, other):
        """Вычитание изображений."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Строковое представление."""
        pass    
