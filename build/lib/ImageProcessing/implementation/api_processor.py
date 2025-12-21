import os
import aiohttp
import aiofiles
import asyncio
import numpy as np
import logging
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

# Логгер для модуля
logger = logging.getLogger(__name__)

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
    import logging #!!!
    # Логгер для процесса
    process_logger = logging.getLogger(__name__)
    
    idx, image_array, variant = args
    import os
    from ImageProcessing.implementation.image_processing import ImageProcessing # Импортируем ImageProcessing в процессе
    
    pid = os.getpid()
    process_logger.debug(f"[PID {pid}] edge_detection called, variant={variant}, image shape={image_array.shape}")
    # print(f"[PID {pid}] edge_detection called, variant={variant}, image shape={image_array.shape}")
    
    # Создаем процессор в этом процессе
    processor = ImageProcessing()
    
    if variant == "new":
        process_logger.debug(f"[PID {pid}] Starting new variant...")
        # print(f"[PID {pid}] Starting new variant...")
    else:
        process_logger.debug(f"[PID {pid}] Starting old variant...")
        # print(f"[PID {pid}] Starting old variant...")
    
    edges = processor.edge_detection(image_array, variant=variant)
    
    process_logger.debug(f"[PID {pid}] Обработано изображение {idx}, вариант {variant}, форма {edges.shape}")
    # print(f"[PID {pid}] Обработано изображение {idx}, вариант {variant}, форма {edges.shape}")
    return (idx, variant, edges, pid)

class AsyncCatImageProcessor:
    """Асинхронный класс с истинно параллельной обработкой."""
    
    def __init__(self, max_workers: int = None):
        self._api_key = os.getenv('API_KEY')
        if not self._api_key:
            logger.error("API_KEY не найден. Убедитесь, что он указан в .env файле")
            raise ValueError("API_KEY не найден. Убедитесь, что он указан в .env файле")
        
        self._base_url = "https://api.thecatapi.com/v1"
        self._headers = {"x-api-key": self._api_key}
        self._session = None
        self._max_workers = max_workers or os.cpu_count()
        logger.info(f"Инициализирован процессор с {self._max_workers} воркерами")
        # print(f"Инициализирован процессор с {self._max_workers} воркерами")
    
    async def __aenter__(self):
        logger.debug("Создание aiohttp сессии")
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb): #!!!
        logger.debug("Закрытие aiohttp сессии")
        if self._session:
            await self._session.close()
    
    # ========== ПАРАЛЛЕЛЬНЫЙ ПАЙПЛАЙН ==========
    
    async def download_generator(self, limit: int = 1) -> AsyncGenerator[ImageData, None]:
        """Генератор скачивания."""
        logger.info(f"Начинаем скачивание {limit} изображений")
        # print(f"[Download] Начинаем скачивание {limit} изображений")
        
        try:
            # Получаем URL
            url = f"{self._base_url}/images/search"
            params = {"limit": limit, "has_breeds": 1, "size": "med"}
            
            logger.debug(f"Запрос к API: {url} с параметрами {params}")
            async with self._session.get(url, params=params) as response:
                data = await response.json()
            
            logger.debug(f"Получено {len(data)} изображений от API")

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
        except Exception as e:
            logger.error(f"Ошибка при скачивании изображений: {e}", exc_info=True)
            raise    
    
    async def _download_single(self, idx: int, img_url: str, breed: str) -> ImageData:
        """Скачивание одного изображения."""
        logger.debug(f"Скачивание изображения {idx} с URL: {img_url}")
        # print(f"[Downloading image {idx} started]")
        
        try:
            async with self._session.get(img_url) as response:
                content = await response.read()
            
            # Конвертация
            img_array = await asyncio.get_event_loop().run_in_executor(
                None,
                self._url_to_array_sync,
                content
            )
            
            logger.debug(f"Изображение {idx} скачано, форма: {img_array.shape}")
            # print(f"[Downloading image {idx} finished] - форма: {img_array.shape}")
            return ImageData(index=idx, breed=breed, image_array=img_array, url=img_url)
        except Exception as e:
            logger.error(f"Ошибка при скачивании изображения {idx}: {e}", exc_info=True)
            raise

    async def processing_generator(self, 
                                 download_gen: AsyncGenerator[ImageData, None]) -> AsyncGenerator[ImageData, None]:
        """Генератор для ПАРАЛЛЕЛЬНОЙ обработки изображений."""
        logger.info(f"Запуск ПАРАЛЛЕЛЬНОЙ обработки (workers: {self._max_workers})")
        # print(f"[Processing] Запуск ПАРАЛЛЕЛЬНОЙ обработки (workers: {self._max_workers})")
        
        # Собираем все изображения для параллельной обработки
        images_to_process = []
        
        async for image_data in download_gen:
            logger.debug(f"Получено изображение {image_data.index}, форма: {image_data.image_array.shape}, порода: {image_data.breed}")
            # print(f"[Processing] Получено изображение {image_data.index}, форма: {image_data.image_array.shape}")
            images_to_process.append(image_data)
        
        if not images_to_process:
            logger.warning("Нет изображений для обработки в processing_generator")
            return
        
        logger.info(f"Начинаем параллельную обработку {len(images_to_process)} изображений")
        # print(f"[Processing] Начинаем параллельную обработку {len(images_to_process)} изображений...")
        
        logger.debug("Инициализация ProcessPoolExecutor для параллельной обработки")
        # ПАРАЛЛЕЛЬНАЯ обработка всех изображений
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            logger.debug(f"ProcessPoolExecutor создан с {self._max_workers} воркерами")
            
            loop = asyncio.get_event_loop()
            
            # Подготавливаем ВСЕ задачи обработки
            all_futures = []
            future_info = {}  # Связь future с информацией
            
            for img_data in images_to_process:
                logger.debug(f"Подготовка задачи обработки для изображения {img_data.index}")
                # print(f"[Convolution for image {img_data.index} started]")
                
                # Проверяем данные
                if img_data.image_array is None:
                    logger.error(f"image_array is None для изображения {img_data.index}")
                    # print(f"[ERROR] image_array is None для изображения {img_data.index}")
                    continue
                
                # Создаем задачи для ОБОИХ типов обработки каждого изображения
                custom_task = (img_data.index, img_data.image_array.copy(), "new")
                library_task = (img_data.index, img_data.image_array.copy(), "old")
                
                logger.debug(f"Созданы задачи для изображения {img_data.index}: custom и library варианты")

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
            
            logger.debug(f"Всего создано {len(all_futures)} задач обработки")

            # Обрабатываем результаты по мере готовности
            results = {}  # {img_index: {"custom": edges, "library": edges, "pid": pid}}
            
            # Ждем завершения ВСЕХ задач
            completed_count = 0
            for future in asyncio.as_completed(all_futures):
                try:
                    img_idx, variant, edges, pid = await future
                    completed_count += 1

                    logger.debug(f"Получен результат обработки: изображение {img_idx}, вариант {variant}, PID {pid}, форма границ: {edges.shape}")
                    logger.debug(f"Прогресс: {completed_count}/{len(all_futures)} задач завершено")
                    # print(f"[Processing] Получен результат: изображение {img_idx}, вариант {variant}, PID {pid}")
                    
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
                    logger.error(f"Ошибка при обработке изображения в процессе: {e}", exc_info=True)
                    # print(f"[ERROR] Ошибка при обработке: {e}")
                    continue
            
            logger.info(f"Параллельная обработка завершена. Успешно обработано {len(results)} из {len(images_to_process)} изображений")
            # Возвращаем обработанные изображения в исходном порядке
            for img_data in images_to_process:
                img_idx = img_data.index
                if img_idx in results:
                    result = results[img_idx]
                    
                    # Проверяем что оба результата получены
                    if result["custom"] is not None and result["library"] is not None:
                        img_data.custom_edges = result["custom"]
                        img_data.library_edges = result["library"]
                        
                        logger.debug(f"Обработка изображения {img_idx} завершена (PID {result['pid']})")
                        logger.debug(f"Custom edges shape: {img_data.custom_edges.shape}, Library edges shape: {img_data.library_edges.shape}")

                        # print(f"[Convolution for image {img_idx} finished (PID {result['pid']})]")
                        # print(f"  Custom edges shape: {img_data.custom_edges.shape}")
                        # print(f"  Library edges shape: {img_data.library_edges.shape}")
                        
                        yield img_data
                    else:
                        logger.error(f"Неполные результаты для изображения {img_idx}")
                        logger.error(f"Custom: {result['custom'] is not None}, Library: {result['library'] is not None}, PID: {result['pid']}")

                        # print(f"[ERROR] Неполные результаты для изображения {img_idx}")
                        # print(f"  Custom: {result['custom'] is not None}, Library: {result['library'] is not None}")
                        # print(f"  PID: {result['pid']}")
                else:
                    logger.warning(f"Результаты для изображения {img_idx} не найдены")

    async def save_generator(self, 
                           processing_gen: AsyncGenerator[ImageData, None],
                           output_dir: str = "processed_images") -> AsyncGenerator[int, None]:
        """Генератор асинхронного сохранения."""
        logger.info("Запуск асинхронного сохранения обработанных изображений")
        logger.debug(f"Выходная директория: {output_dir}")
        # print("[Save] Запуск асинхронного сохранения")
        
        # Создаем директорию
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Директория {output_dir} создана/проверена")
        except Exception as e:
            logger.error(f"Ошибка при создании директории {output_dir}: {e}")
            return
        
        saved_count = 0
        async for image_data in processing_gen:
            logger.debug(f"Начало сохранения изображения {image_data.index}, порода: {image_data.breed}")
            # print(f"[Saving image {image_data.index} started]")
            
            try:
                # Проверяем данные перед сохранением
                if image_data.custom_edges is None:
                    logger.error(f"custom_edges is None для изображения {image_data.index} - пропускаем сохранение")
                    # print(f"[ERROR] custom_edges is None для изображения {image_data.index}")
                    continue
                
                if image_data.library_edges is None:
                    logger.error(f"library_edges is None для изображения {image_data.index} - пропускаем сохранение")
                    # print(f"[ERROR] library_edges is None для изображения {image_data.index}")
                    continue
                
                logger.debug(f"Данные для изображения {image_data.index}:")
                logger.debug(f"  Original shape: {image_data.image_array.shape}")
                logger.debug(f"  Custom edges shape: {image_data.custom_edges.shape}")
                logger.debug(f"  Library edges shape: {image_data.library_edges.shape}")

                # print(f"[Save] Данные для изображение {image_data.index}:")
                # print(f"  Original shape: {image_data.image_array.shape}")
                # print(f"  Custom edges shape: {image_data.custom_edges.shape}")
                # print(f"  Library edges shape: {image_data.library_edges.shape}")
                
                # Сохраняем
                await self._save_single_image(image_data, output_dir)
                saved_count += 1

                logger.info(f"Изображение {image_data.index} успешно сохранено ({saved_count} всего)")
                # print(f"[Saving image {image_data.index} finished]")
                
                yield image_data.index
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении изображения {image_data.index}: {e}", exc_info=True)
                # print(f"[ERROR] Ошибка при сохранении изображения {image_data.index}: {e}")
                continue
        
        logger.info(f"Сохранение завершено. Успешно сохранено {saved_count} изображений")
    
    async def _save_single_image(self, image_data: ImageData, output_dir: str) -> None:
        """Асинхронное сохранение одного изображения."""
        base_filename = f"{image_data.index}_{image_data.breed}"
        logger.debug(f"Сохранение изображения {image_data.index} в файлы с базовым именем: {base_filename}")
        
        # Сохраняем все три файла ПАРАЛЛЕЛЬНО
        save_tasks = []
        
        try:
            logger.debug(f"Запуск параллельного сохранения 3 файлов для изображения {image_data.index}")

            # Исходное изображение
            original_path = os.path.join(output_dir, f"{base_filename}_original.png")
            save_tasks.append(self._save_image_async(
                original_path,
                cv2.cvtColor(image_data.image_array, cv2.COLOR_RGB2BGR)
            ))
            logger.debug(f"Задача сохранения оригинала: {original_path}")
            
            # Пользовательские границы
            custom_path = os.path.join(output_dir, f"{base_filename}_custom_edges.png")
            save_tasks.append(self._save_image_async(custom_path, image_data.custom_edges))
            logger.debug(f"Задача сохранения custom edges: {custom_path}")

            # Библиотечные границы
            library_path = os.path.join(output_dir, f"{base_filename}_library_edges.png")
            save_tasks.append(self._save_image_async(library_path, image_data.library_edges))
            logger.debug(f"Задача сохранения library edges: {library_path}")

            # Ждем сохранения всех файлов
            await asyncio.gather(*save_tasks)
            logger.debug(f"Все 3 файла для изображения {image_data.index} успешно сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке сохранения изображения {image_data.index}: {e}", exc_info=True)
            # print(f"[ERROR] Ошибка при подготовке сохранения изображения {image_data.index}: {e}")
            raise
    
    async def _save_image_async(self, filepath: str, image_array: np.ndarray) -> None:
        """Асинхронное сохранение изображения."""
        try:
            logger.debug(f"Начало сохранения файла: {filepath}")

            # Проверяем что массив не пустой
            if image_array is None:
                logger.error(f"Пустой массив для {filepath}")
                raise ValueError(f"Пустой массив для {filepath}")
            
            if image_array.size == 0:
                logger.error(f"Пустой массив (size=0) для {filepath}")
                raise ValueError(f"Пустой массив (size=0) для {filepath}")
            
            logger.debug(f"Сохранение {filepath}, форма: {image_array.shape}")
            # print(f"[Save] Сохранение {filepath}, форма: {image_array.shape}")
            
            # Кодируем
            success, encoded_image = cv2.imencode('.png', image_array)
            if not success:
                logger.error(f"Ошибка кодирования: {filepath}")
                raise ValueError(f"Ошибка кодирования: {filepath}")
            
            logger.debug(f"Изображение успешно закодировано в PNG, размер: {len(encoded_image.tobytes())} байт")

            # Записываем асинхронно
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(encoded_image.tobytes())
                
            logger.info(f"Успешно сохранено: {filepath}")
            # print(f"[Save] Успешно сохранено: {filepath}")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения {filepath}: {e}", exc_info=True)
            # print(f"[ERROR] Ошибка сохранения {filepath}: {e}")
            raise
    
    # ========== ОСНОВНОЙ ПАЙПЛАЙН ==========
    
    async def processing_pipeline(self, limit: int = 1, 
                                output_dir: str = "processed_images") -> AsyncGenerator[int, None]:
        """Основной пайплайн с параллельной обработкой."""
        logger.info(f"Запуск основного пайплайна с параметрами: limit={limit}, output_dir={output_dir}")
        logger.debug("Создание цепочки генераторов: download → processing → save")
        # print("[Pipeline] Запуск пайплайна с параллельной обработкой")
        
        # Цепочка генераторов
        download_gen = self.download_generator(limit)
        processing_gen = self.processing_generator(download_gen)
        save_gen = self.save_generator(processing_gen, output_dir)
        
        logger.debug("Начало прохода по цепочке пайплайна")

        # Проходим по цепочке
        processed_count = 0
        async for img_index in save_gen:
            processed_count += 1
            logger.debug(f"Изображение {img_index} прошло полный пайплайн ({processed_count}/{limit})")
            yield img_index
        
        logger.info(f"Пайплайн завершен. Всего обработано изображений: {processed_count}")

    def _url_to_array_sync(self, image_content: bytes) -> np.ndarray:
        """Конвертация изображения."""
        logger.debug(f"Конвертация байтов в numpy array, размер: {len(image_content)} байт")
        
        try:
            image = Image.open(io.BytesIO(image_content))
            img_array = np.array(image)
            
            logger.debug(f"Изображение успешно конвертировано, форма: {img_array.shape}, dtype: {img_array.dtype}")
            return img_array
            
        except Exception as e:
            logger.error(f"Ошибка при конвертации изображения: {e}", exc_info=True)
            raise
    
CatImageProcessor = AsyncCatImageProcessor