import time
from functools import wraps
from typing import Any, Callable

def timer_decorator(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения методов класса."""
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"Метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper