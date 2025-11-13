"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2
import time as t

import interfaces

import numpy as np


class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        convolution(image, kernel): Выполняет свёртку изображения с ядром.
        rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    Вспомогательные методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _convolution_2d(padded_image, kernel, k_size, pad): Выполняет свертку 2D изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        _sobel_gradients(image): Вычисляет градиенты на изображении.
        _non_maximum_suppression(magnitude, direction): Подавляет немаксимумы на изображении (поиск всех возможных границ).
        _hysteresis_thresholding(image, low_threshold, high_threshold): Выбирает границы с пороговым значением выше high_threshold 
            и прилежащие к ним с пороговым значением выше low_threshold.
    """

    def convolution(self, 
                    image: np.ndarray, 
                    kernel: np.ndarray,
                    variant: str = "new") -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует ручную реализацию свёртки.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).
            variant (str): свой вариант реализации(new) или через cv2(old)

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._convolution(image, kernel, variant)
        output_image = np.clip(output_image, 0, 255) # Ограничиваем значения пикселей диапазоном [0, 255]
        output_image = output_image.astype(np.uint8) # Возвращаем изображение в формате uint8

        end = t.time()
        print(f"Execution time: {end - start}")

        return output_image

    def _convolution(self, 
                     image: np.ndarray, 
                     kernel: np.ndarray,
                     variant: str = "new") -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Метод проверяет корректность данных, добавляет поля к изображению и вызывает вспомогательный метод, 
        реализующий свертку посредством перемножения кусочков изображения и ядра свертки ("вырезается" кусочек изображения 
        размерности ядра свертки) и сложения результатов для получения нового значения пикселя (центрального для кусочка изображения).

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица нечетного размера).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        if variant == "new":

            kernel = np.flipud(np.fliplr(kernel)) # 180° поворот
            k_size = kernel.shape[0]
            pad = k_size // 2

            if image.ndim == 2:
                padded = np.pad(image, pad, mode='reflect')
                output_image = self._convolution_2d(padded, kernel, pad)

            else:
                for channel in range(image.shape[2]):
                    padded = np.pad(image[:, :, channel], pad, mode='reflect')
                    output_image[:, :, channel] = self._convolution_2d(padded, kernel, pad)
        
        else:
            output_image = cv2.filter2D(image, -1, kernel)

        return output_image    

    def _convolution_2d(self, 
                        padded_image: np.ndarray, 
                        kernel: np.ndarray, 
                        pad: int) -> np.ndarray:
        """
        Вспомогательный метод для 2D свертки.
        
        Args:
            padded_image (np.ndarray): изображение с добавленными отступами (обрабатывается один цветовой канал).
            kernel (np.ndarray): ядро свертки.
            k_size (int): размер ядра.
            pad (int): размер отступа.
            
        Returns:
            np.ndarray: Изображение после применения свёртки для одного канала.
        """
        h, w = padded_image.shape
        output_image = np.zeros((h - 2 * pad, w - 2 * pad), dtype=np.float64)

        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                window = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
                output_image[i-pad, j-pad] = np.sum(window * kernel)

        return output_image

    def rgb_to_grayscale(self,
                         image: np.ndarray,
                         variant: str = "new") -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует взвешенное среднее для преобразования каждого пикселя.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._rgb_to_grayscale(image, variant)
        output_image = np.clip(output_image, 0, 255) # Ограничиваем значения пикселей диапазоном [0, 255]
        output_image = output_image.astype(np.uint8) # Возвращаем изображение в формате uint8
        
        end = t.time()
        print(f"Execution time: {end - start}")

        return output_image
    
    def _rgb_to_grayscale(self, 
                          image: np.ndarray,
                          variant = "new") -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует умножение матрицы на вектор коэффициентов для преобразования цветного изображения
        в чёрно-белое.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        if variant == "new":

            if image.ndim == 2:
                return image.copy()
            
            if image.ndim == 3:
                coeffs = np.array([0.114, 0.587, 0.299])
                output_image = np.dot(image.astype(np.float64), coeffs)
                output_image = np.clip(output_image, 0, 255)

        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return output_image # Возвращаем float для точности

    def gamma_correction(self, 
                         image: np.ndarray, 
                         gamma: float,
                         variant: str = "new") -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """

        start, end = 0.0, 0.0
        start = t.time()

        output_image = self._gamma_correction(image, gamma, variant).astype(np.uint8)
        
        end = t.time()
        print(f"Execution time: {end - start}")

        return output_image

    def _gamma_correction(self, 
                          image: np.ndarray, 
                          gamma: float,
                          variant: str = "new") -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью нормализации и степенного преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).
                y < 1 - осветление, y > 1 - затемнение, y = 1 - без изменений

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        if variant == "new":

            if image.size == 0:
                return image.copy()
            
            if gamma == 1:
                output_image = image

            else:
                output_image = np.power(image.astype(np.float64) / 255.0, gamma)*255
        
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return output_image

    def edge_detection(self, 
                       image: np.ndarray,
                       variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении по алгоритму Кэнни.
        Алгоритм состоит из преобразования изображение в чб вариант, вычисления градиентов (через Собеля и свертки), 
        подавления немаксимумов и двойной пороговой фильтрации.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        start, end = 0.0, 0.0
        start = t.time()

        if variant == "new":

            gray = self._rgb_to_grayscale(image)
            grad_x, grad_y = self._sobel_gradients(gray)
            magnitude = np.sqrt(grad_x**2 + grad_y**2) 
            direction = np.arctan2(grad_y, grad_x)  # в радианах

            suppressed = self._non_maximum_suppression(magnitude, direction)
            edges = self._hysteresis_thresholding(suppressed, low_threshold=0.085, high_threshold=0.175)
            output_image = edges.astype(np.uint8) # Возвращаем изображение в формате uint8

        else:
             gray = self._rgb_to_grayscale(image, variant ="old")
             output_image = cv2.Canny(gray, 100, 200)

        end = t.time()
        print(f"Execution time: {end - start}")
        
        return output_image
    
    def _sobel_gradients(self, 
                         image: np.ndarray) -> tuple:
        """
        Вычисляет градиенты по X и Y с помощью оператора Собеля.
        
        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tuple: Кортеж с двумя массивами np.ndarray, отражающими градиенты на изображении.
        """
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float64)
        
        grad_x = self._convolution(image, sobel_x)
        grad_y = self._convolution(image, sobel_y)

        return grad_x, grad_y
    
    def _non_maximum_suppression(self, 
                                 magnitude: np.ndarray, 
                                 direction: np.ndarray) -> np.ndarray:
        """
        Подавляет немаксимумы, т.е. оставляет только самые сильные границы.
        
        Args:
            magnitude (np.ndarray): Изображение, отражающее резкость перехода цвета.
            direction (np.ndarray): Двумерный массив, отражающий направление перехода цвета на изображении.

        Returns:
            np.ndarray: Изображение с самыми сильными границами.
        """
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:  # нормализуем, избегаем деления на ноль
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)

        suppressed = np.zeros_like(magnitude)
        direction_deg = np.degrees(direction) % 180
        
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                if magnitude[i, j] == 0:
                    continue

                angle = direction_deg[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]

                elif 22.5 <= angle < 67.5:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

                elif 67.5 <= angle < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]

                else:  # 112.5 <= angle < 157.5
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]

        return suppressed
    
    def _hysteresis_thresholding(self, 
                                 image: np.ndarray, 
                                 low_threshold: float, 
                                 high_threshold: float) -> np.ndarray:
        """
        Фильтрует границы по двум порогам резкости.
        
        Args:
            image (np.ndarray): Изображение с самыми сильными границами.
            low_threshold (float): Нижнее пороговое значение для слабых граней.
            high_threshold (float): Нижнее пороговое значение для сильных граней.

        Returns:
            np.ndarray: Изображение с границами с пороговым значением выше high_threshold и прилежащие к ним с пороговым значением 
        выше low_threshold.
        """       
        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)
        edges = strong_edges.astype(np.uint8) * 255
        
        for i in range(1, edges.shape[0] - 1):
            for j in range(1, edges.shape[1] - 1):
                if weak_edges[i, j] and edges[i, j] == 0:
                    if np.any(strong_edges[i-1:i+1, j-1:j+1]):
                        edges[i, j] = 255

        return edges

    def corner_detection(self, 
                         image: np.ndarray, 
                         k: float = 0.04, 
                         threshold: float = 0.03,
                         variant: str = "new") -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.
        Функция последовательно вычисляет градиенты изображения и матрицу Харриса
        для каждого пикселя, и на основе "карты отклика" находит и выделяет углы.

        Args:
            image (np.ndarray): Входное изображение (RGB).
            k (float): Свободный параметр детектора Харриса для вычисления отклика.
            threshold_ratio (float): Коэффициент для определения порога отсечки
                слабых углов. (отбираются углы "силой" в 1% от самого сильного).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        start, end = 0.0, 0.0
        start = t.time()

        if variant == "new":

            gray = self._rgb_to_grayscale(image)
            grad_x, grad_y = self._sobel_gradients(gray)

            sobel_x = grad_x**2
            sobel_y = grad_y**2
            sobel_xy = grad_x * grad_y

            Ghaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

            smoothed_sob_x = self._convolution(sobel_x, Ghaussian_kernel)
            smoothed_sob_y = self._convolution(sobel_y, Ghaussian_kernel)
            smoothed_sob_xy = self._convolution(sobel_xy, Ghaussian_kernel)

            # Вычисляем отклик Харриса для каждого пикселя
            det_m = smoothed_sob_x * smoothed_sob_y - smoothed_sob_xy**2
            trace_m = smoothed_sob_x + smoothed_sob_y
            R = det_m - k * (trace_m)**2

            # Находим углы и рисуем их на исходном изображении
            result_image = image.copy()
            threshold = threshold * R.max()
            corner_coords = np.where(R > threshold)

            result_image[corner_coords] = [0, 0, 255]
            output_image = result_image.astype(np.uint8)
            
        else:
            gray = self._rgb_to_grayscale(image, variant = "old")
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            result = image.copy()
            result[dst > 0.01 * dst.max()] = [255, 0, 0]
            output_image = result

        end = t.time()
        print(f"Execution time: {end - start}")

        return output_image

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
