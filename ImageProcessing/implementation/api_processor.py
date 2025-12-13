import os
import aiohttp
import aiofiles
import asyncio
import numpy as np
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

load_dotenv() # загружает переменные окружения из файла .env для доступа через os.getenv('API_KEY')

@dataclass
class ImageData: # DTO, необходим для сериализации
    """Данные изображения."""
    index: int
    breed: str
    image_array: np.ndarray
    url: str
    custom_edges: np.ndarray = None
    library_edges: np.ndarray = None

def process_edges_sync_wrapper(args: tuple) -> tuple:
    """Обертка для обработки границ в процессе с использованием ImageProcessing."""
    idx, image_array, variant = args
    import os
    from implementation.image_processing import ImageProcessing # Импортируем ImageProcessing в процессе
    
    pid = os.getpid()
    print(f"[PID {pid}] edge_detection called, variant={variant}, image shape={image_array.shape}")
    
    # Создаем процессор в этом процессе
    processor = ImageProcessing()
    
    if variant == "new":
        print(f"[PID {pid}] Starting new variant...")
    else:
        print(f"[PID {pid}] Starting old variant...")
    
    edges = processor.edge_detection(image_array, variant=variant)
    
    print(f"[PID {pid}] Обработано изображение {idx}, вариант {variant}, форма {edges.shape}")
    return (idx, variant, edges, pid)

class AsyncCatImageProcessor:
    """Асинхронный класс с истинно параллельной обработкой."""
    
    def __init__(self, max_workers: int = None):
        self._api_key = os.getenv('API_KEY')
        if not self._api_key:
            raise ValueError("API_KEY не найден. Убедитесь, что он указан в .env файле")
        
        self._base_url = "https://api.thecatapi.com/v1"
        self._headers = {"x-api-key": self._api_key}
        self._session = None
        self._max_workers = max_workers or os.cpu_count()
        print(f"Инициализирован процессор с {self._max_workers} воркерами")
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    # ========== ПАРАЛЛЕЛЬНЫЙ ПАЙПЛАЙН ==========
    
    async def download_generator(self, limit: int = 1) -> AsyncGenerator[ImageData, None]:
        """Генератор скачивания."""
        print(f"[Download] Начинаем скачивание {limit} изображений")
        
        # Получаем URL
        url = f"{self._base_url}/images/search"
        params = {"limit": limit, "has_breeds": 1, "size": "med"}
        
        async with self._session.get(url, params=params) as response:
            data = await response.json()
        
        # Создаем задачи скачивания
        tasks = []
        for idx, item in enumerate(data, start=1):
            img_url = item['url']
            breed_info = item.get('breeds', [{}])[0]
            breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()
            
            tasks.append(self._download_single(idx, img_url, breed_name))
        
        # Обрабатываем по мере готовности
        for task in asyncio.as_completed(tasks):
            yield await task
    
    async def _download_single(self, idx: int, img_url: str, breed: str) -> ImageData:
        """Скачивание одного изображения."""
        print(f"[Downloading image {idx} started]")
        
        async with self._session.get(img_url) as response:
            content = await response.read()
        
        # Конвертация
        img_array = await asyncio.get_event_loop().run_in_executor(
            None,
            self._url_to_array_sync,
            content
        )
        
        print(f"[Downloading image {idx} finished] - форма: {img_array.shape}")
        return ImageData(index=idx, breed=breed, image_array=img_array, url=img_url)
    
    async def processing_generator(self, 
                                 download_gen: AsyncGenerator[ImageData, None]) -> AsyncGenerator[ImageData, None]:
        """Генератор для ПАРАЛЛЕЛЬНОЙ обработки изображений."""
        print(f"[Processing] Запуск ПАРАЛЛЕЛЬНОЙ обработки (workers: {self._max_workers})")
        
        # Собираем все изображения для параллельной обработки
        images_to_process = []
        
        async for image_data in download_gen:
            print(f"[Processing] Получено изображение {image_data.index}, форма: {image_data.image_array.shape}")
            images_to_process.append(image_data)
        
        if not images_to_process:
            return
        
        print(f"[Processing] Начинаем параллельную обработку {len(images_to_process)} изображений...")
        
        # ПАРАЛЛЕЛЬНАЯ обработка всех изображений
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            # Подготавливаем ВСЕ задачи обработки
            all_futures = []
            future_info = {}  # Связь future с информацией
            
            for img_data in images_to_process:
                print(f"[Convolution for image {img_data.index} started]")
                
                # Проверяем данные
                if img_data.image_array is None:
                    print(f"[ERROR] image_array is None для изображения {img_data.index}")
                    continue
                
                # Создаем задачи для ОБОИХ типов обработки каждого изображения
                custom_task = (img_data.index, img_data.image_array.copy(), "new")
                library_task = (img_data.index, img_data.image_array.copy(), "old")
                
                # Отправляем задачи в пул процессов
                custom_future = loop.run_in_executor(
                    executor,
                    process_edges_sync_wrapper,
                    custom_task
                )
                
                library_future = loop.run_in_executor(
                    executor,
                    process_edges_sync_wrapper,
                    library_task
                )
                
                # Сохраняем связь
                future_info[custom_future] = {"img_index": img_data.index, "type": "custom"}
                future_info[library_future] = {"img_index": img_data.index, "type": "library"}
                all_futures.extend([custom_future, library_future])
            
            # Обрабатываем результаты по мере готовности
            results = {}  # {img_index: {"custom": edges, "library": edges, "pid": pid}}
            
            # Ждем завершения ВСЕХ задач
            for future in asyncio.as_completed(all_futures):
                try:
                    img_idx, variant, edges, pid = await future
                    
                    print(f"[Processing] Получен результат: изображение {img_idx}, вариант {variant}, PID {pid}")
                    
                    # Инициализируем запись для этого изображения если нужно
                    if img_idx not in results:
                        results[img_idx] = {"custom": None, "library": None, "pid": pid}
                    
                    # Сохраняем результат по правильному ключу
                    if variant == "new":
                        results[img_idx]["custom"] = edges
                    elif variant == "old":
                        results[img_idx]["library"] = edges
                    
                    # Обновляем PID (берем из любого успешного результата)
                    if results[img_idx]["pid"] is None:
                        results[img_idx]["pid"] = pid
                    
                except Exception as e:
                    print(f"[ERROR] Ошибка при обработке: {e}")
                    continue
            
            # Возвращаем обработанные изображения в исходном порядке
            for img_data in images_to_process:
                img_idx = img_data.index
                if img_idx in results:
                    result = results[img_idx]
                    
                    # Проверяем что оба результата получены
                    if result["custom"] is not None and result["library"] is not None:
                        img_data.custom_edges = result["custom"]
                        img_data.library_edges = result["library"]
                        
                        print(f"[Convolution for image {img_idx} finished (PID {result['pid']})]")
                        print(f"  Custom edges shape: {img_data.custom_edges.shape}")
                        print(f"  Library edges shape: {img_data.library_edges.shape}")
                        
                        yield img_data
                    else:
                        print(f"[ERROR] Неполные результаты для изображения {img_idx}")
                        print(f"  Custom: {result['custom'] is not None}, Library: {result['library'] is not None}")
                        print(f"  PID: {result['pid']}")
    
    async def save_generator(self, 
                           processing_gen: AsyncGenerator[ImageData, None],
                           output_dir: str = "processed_images") -> AsyncGenerator[int, None]:
        """Генератор асинхронного сохранения."""
        print("[Save] Запуск асинхронного сохранения")
        
        # Создаем директорию
        os.makedirs(output_dir, exist_ok=True)
        
        async for image_data in processing_gen:
            print(f"[Saving image {image_data.index} started]")
            
            try:
                # Проверяем данные перед сохранением
                if image_data.custom_edges is None:
                    print(f"[ERROR] custom_edges is None для изображения {image_data.index}")
                    continue
                
                if image_data.library_edges is None:
                    print(f"[ERROR] library_edges is None для изображения {image_data.index}")
                    continue
                
                print(f"[Save] Данные для изображение {image_data.index}:")
                print(f"  Original shape: {image_data.image_array.shape}")
                print(f"  Custom edges shape: {image_data.custom_edges.shape}")
                print(f"  Library edges shape: {image_data.library_edges.shape}")
                
                # Сохраняем
                await self._save_single_image(image_data, output_dir)
                print(f"[Saving image {image_data.index} finished]")
                
                yield image_data.index
                
            except Exception as e:
                print(f"[ERROR] Ошибка при сохранении изображения {image_data.index}: {e}")
                continue
    
    async def _save_single_image(self, image_data: ImageData, output_dir: str) -> None:
        """Асинхронное сохранение одного изображения."""
        base_filename = f"{image_data.index}_{image_data.breed}"
        
        # Сохраняем все три файла ПАРАЛЛЕЛЬНО
        save_tasks = []
        
        try:
            # Исходное изображение
            original_path = os.path.join(output_dir, f"{base_filename}_original.png")
            save_tasks.append(self._save_image_async(
                original_path,
                cv2.cvtColor(image_data.image_array, cv2.COLOR_RGB2BGR)
            ))
            
            # Пользовательские границы
            custom_path = os.path.join(output_dir, f"{base_filename}_custom_edges.png")
            save_tasks.append(self._save_image_async(custom_path, image_data.custom_edges))
            
            # Библиотечные границы
            library_path = os.path.join(output_dir, f"{base_filename}_library_edges.png")
            save_tasks.append(self._save_image_async(library_path, image_data.library_edges))
            
            # Ждем сохранения всех файлов
            await asyncio.gather(*save_tasks)
            
        except Exception as e:
            print(f"[ERROR] Ошибка при подготовке сохранения изображения {image_data.index}: {e}")
            raise
    
    async def _save_image_async(self, filepath: str, image_array: np.ndarray) -> None:
        """Асинхронное сохранение изображения."""
        try:
            # Проверяем что массив не пустой
            if image_array is None:
                raise ValueError(f"Пустой массив для {filepath}")
            
            if image_array.size == 0:
                raise ValueError(f"Пустой массив (size=0) для {filepath}")
            
            print(f"[Save] Сохранение {filepath}, форма: {image_array.shape}")
            
            # Кодируем
            success, encoded_image = cv2.imencode('.png', image_array)
            if not success:
                raise ValueError(f"Ошибка кодирования: {filepath}")
            
            # Записываем асинхронно
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(encoded_image.tobytes())
                
            print(f"[Save] Успешно сохранено: {filepath}")
                
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения {filepath}: {e}")
            raise
    
    # ========== ОСНОВНОЙ ПАЙПЛАЙН ==========
    
    async def processing_pipeline(self, limit: int = 1, 
                                output_dir: str = "processed_images") -> AsyncGenerator[int, None]:
        """Основной пайплайн с параллельной обработкой."""
        print("[Pipeline] Запуск пайплайна с параллельной обработкой")
        
        # Цепочка генераторов
        download_gen = self.download_generator(limit)
        processing_gen = self.processing_generator(download_gen)
        save_gen = self.save_generator(processing_gen, output_dir)
        
        # Проходим по цепочке
        async for img_index in save_gen:
            yield img_index
    
    def _url_to_array_sync(self, image_content: bytes) -> np.ndarray:
        """Конвертация изображения."""
        image = Image.open(io.BytesIO(image_content))
        return np.array(image)
    
CatImageProcessor = AsyncCatImageProcessor