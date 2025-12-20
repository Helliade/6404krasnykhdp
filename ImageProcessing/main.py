# python -m ImageProcessing.__main__

import asyncio
import time
import logging
import ImageProcessing
import sys

# Настройка логгера
logger = logging.getLogger(__name__)

def get_image_count() -> int:
    """Запрашивает у пользователя количество изображений."""
    while True:
        try:
            limit = int(input("Введите количество изображений для обработки (1-10): "))
            if 1 <= limit <= 10:
                logger.info(f"Пользователь выбрал {limit} изображений")
                return limit
            else:
                logger.warning(f"Некорректный ввод: {limit}. Ожидается 1-10")
                print("Пожалуйста, введите число от 1 до 10")
        except ValueError:
            logger.error("Введено нечисловое значение")
            print("Пожалуйста, введите корректное число")

def get_worker_count() -> int:
    """Запрашивает у пользователя количество воркеров."""
    logger.debug("Запрос количества воркеров у пользователя")
    import os
    cpu_count = os.cpu_count() or 2
    
    while True:
        try:
            workers = input(f"Введите количество воркеров (1-{cpu_count}, по умолчанию {cpu_count}): ").strip()
            
            if not workers:  # Если пустая строка - используем значение по умолчанию
                logger.info(f"Используется значение по умолчанию: {cpu_count} воркеров")
                return cpu_count
            
            workers = int(workers)
            if 1 <= workers <= cpu_count:
                logger.info(f"Пользователь выбрал {workers} воркеров")
                return workers
            else:
                logger.warning(f"Некорректное количество воркеров: {workers}")
                print(f"Пожалуйста, введите число от 1 до {cpu_count}")
        except ValueError:
            logger.error("Введено нечисловое значение для воркеров")
            print("Пожалуйста, введите корректное число")

async def main():
    """Основная функция с пользовательским вводом."""
    logger.info("=" * 50)
    logger.info("ЗАПУСК ПРИЛОЖЕНИЯ ОБРАБОТКИ ИЗОБРАЖЕНИЙ")
    logger.info("=" * 50)
    
    # Запрос параметров у пользователя
    logger.info("Настройка параметров обработки")
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБРАБОТКИ")
    print("-" * 40)
    
    limit = get_image_count()
    workers = get_worker_count()
    
    logger.info(f"Запуск пайплайна с параметрами: images={limit}, workers={workers}")
    print("\n" + "ЗАПУСК ПАЙПЛАЙНА")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        async with ImageProcessing.AsyncCatImageProcessor(max_workers=workers) as processor: # + создается сессия для работы с сервером (100 соед, 5 минут)
            logger.info(f"Начало обработки {limit} изображений...")
            print(f"Начало обработки {limit} изображений...")
            
            results = []
            count = 0
            
            async for img_index in processor.processing_pipeline(limit=limit):
                count += 1
                results.append(img_index)
                logger.debug(f"Изображение {img_index} обработано ({count}/{limit})")
                print(f"  ✅ Изображение {img_index} обработано ({count}/{limit})")
        
        total_time = time.time() - start_time

        logger.info(f"Обработка завершена. Время: {total_time:.2f} сек")
        logger.info(f"Результаты: {len(results)} изображений, {workers} воркеров")
        
        print(f"\n" + "-" * 80)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
        print("-" * 80)
        print(f"Общее время: {total_time:.2f} секунд")
        print(f"Изображений обработано: {len(results)}")
        print(f"Воркеров использовано: {workers}")
        print(f"Порядок обработки: {results}")
        print(f"Среднее время на изображение: {total_time/len(results):.2f} сек")

        logger.info("=" * 50)
        logger.info("ПРИЛОЖЕНИЕ УСПЕШНО ЗАВЕРШИЛО РАБОТУ")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке: {e}", exc_info=True)
        print(f"\n❌ Ошибка при обработке: {e}")

if __name__ == "__main__":
    # Импортируем конфигурацию логирования
    from ImageProcessing.logging_config import setup_logging
    setup_logging()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
        print("\n\nПриложение остановлено пользователем")
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка: {e}", exc_info=True)