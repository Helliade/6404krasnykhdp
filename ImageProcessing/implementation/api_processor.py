import os
import requests
import numpy as np
import cv2
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image
import io

from decorators import timer_decorator
from implementation.cat_image import ColorCatImage, GrayscaleCatImage

load_dotenv()

class CatImageProcessor:
    """Класс для работы с API и обработки изображений."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv('API_KEY')
        if not self._api_key:
            raise ValueError("API_KEY не найден. Убедитесь, что он указан в .env файле")
        
        self._base_url = "https://api.thecatapi.com/v1"
        self._headers = {"x-api-key": self._api_key}
    
    @timer_decorator
    def download_images(self, limit: int = 1) -> List[ColorCatImage]:
        """Скачивание изображений с информацией о породах."""
        print(f"Начинаем загрузку {limit} изображений...")
        
        url = f"{self._base_url}/images/search"
        params = {
            "limit": limit,
            "has_breeds": 1,
            "size": "med"
        }
        
        response = requests.get(url, headers=self._headers, params=params)
        response.raise_for_status()
        
        cat_images = []
        for data in response.json():
            image_url = data['url']
            breed_info = data.get('breeds', [{}])[0]
            breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()
            
            # Скачиваем изображение
            img_response = requests.get(image_url)
            img_array = self._url_to_array(img_response.content)
            
            cat_image = ColorCatImage(img_array, breed_name, image_url)
            cat_images.append(cat_image)
            print(f"Загружено изображение: {breed_name}")
        
        return cat_images
    
    def _url_to_array(self, image_content: bytes) -> np.ndarray:
        """Конвертация содержимого изображения в numpy array."""
        image = Image.open(io.BytesIO(image_content))
        return np.array(image)
    
    @timer_decorator
    def process_and_save_images(self, cat_images: List[ColorCatImage], output_dir: str = "processed_images"):
        """Обработка и сохранение изображений."""
        print(f"Начинаем обработку {len(cat_images)} изображений...")
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        print(f"Создана директория: {output_dir}")
        
        for i, cat_image in enumerate(cat_images, 1):
            print(f"Обрабатываем изображение {i}: {cat_image.breed}")
            
            # Обработка изображений
            custom_edges = cat_image.detect_edges_custom()
            library_edges = cat_image.detect_edges_library()
            
            # Сохранение файлов
            base_filename = f"{i}_{cat_image.breed}"
            
            # Исходное изображение
            original_path = os.path.join(output_dir, f"{base_filename}_original.png")
            cv2.imwrite(original_path, cv2.cvtColor(cat_image.image_array, cv2.COLOR_RGB2BGR))
            
            # Пользовательская обработка
            custom_path = os.path.join(output_dir, f"{base_filename}_custom_edges.png")
            cv2.imwrite(custom_path, custom_edges)
            
            # Библиотечная обработка
            library_path = os.path.join(output_dir, f"{base_filename}_library_edges.png")
            cv2.imwrite(library_path, library_edges)
            
            print(f"Сохранены файлы для {cat_image.breed}")
    
    @timer_decorator
    def demonstrate_operations(self, cat_images: List[ColorCatImage], output_dir: str = "processed_images"):
        """Демонстрация операций с изображениями."""
        if len(cat_images) >= 1:
            print("Демонстрация операций с изображениями...")
            edges_gray = cat_images[0].detect_edges_library()
            edges_bgr = cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR)

            # Выполняем операции в BGR
            result_add_bgr = cat_images[0] + edges_bgr
            result_sub_bgr = cat_images[0] - edges_bgr

            # Конвертируем результат в RGB для отображения
            result_add = cv2.cvtColor(result_add_bgr, cv2.COLOR_BGR2RGB)
            result_sub = cv2.cvtColor(result_sub_bgr, cv2.COLOR_BGR2RGB)

            #result_add = cat_images[0] + cv2.cvtColor(cat_images[0].detect_edges_library(), cv2.COLOR_GRAY2BGR)
            #result_sub = cat_images[0] - cv2.cvtColor(cat_images[0].detect_edges_library(), cv2.COLOR_GRAY2BGR)

            print("Операции сложения и вычитания выполнены успешно")

            # Сохранение файлов
            base_filename = f"{1}_{cat_images[0].breed}"
            custom_path = os.path.join(output_dir, f"{base_filename}_add.png")
            cv2.imwrite(custom_path, result_add)
            library_path = os.path.join(output_dir, f"{base_filename}_sub.png")
            cv2.imwrite(library_path, result_sub)

            return result_add, result_sub
        return None, None