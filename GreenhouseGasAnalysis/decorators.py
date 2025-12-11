import time
from functools import wraps
from typing import Any, Callable

def timer(func: Callable) -> Callable:
    """Декоратор для измерения времени выполнения методов класса."""
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"Метод {self.__class__.__name__}.{func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

def log_execution(func: Callable) -> Callable:
    """Декоратор для логирования выполнения функций."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        print(f"🔧 Выполняется {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func.__name__} завершена успешно")
            return result
        except Exception as e:
            print(f"❌ {func.__name__} завершена с ошибкой: {e}")
            raise
    return wrapper


def debug_args(func: Callable) -> Callable:
    """Декоратор для отладки аргументов функции."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        print(f"🐛 {func.__name__} вызвана с args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"🐛 {func.__name__} вернула: {type(result)}")
        return result
    return wrapper