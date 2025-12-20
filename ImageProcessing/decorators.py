import time
import logging
from functools import wraps
from typing import Any, Callable

# Логгер для декораторов
logger = logging.getLogger(__name__)

def timer_decorator(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения методов класса."""
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()

        logger.debug(f"Метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")

        # print(f"Метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

def async_timer_decorator(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения асинхронных методов."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(self, *args, **kwargs)
        end_time = time.time()

        logger.debug(f"Асинхронный метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")

        # print(f"Асинхронный метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper