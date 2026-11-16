import asyncio
import time
from functools import wraps
import logging
from requests.exceptions import HTTPError
from typing import Callable, Any, Coroutine

from crowd_sdk.core.utils.http_client import APIResponseError
from crowd_sdk.core.utils.jwt_client import Token
from crowd_sdk.tagme import TagmeClientAdvanced


def handle_http_errors(max_retries: int = 3):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):

            nonlocal max_retries
            client = kwargs.get("client") or next(
                (arg for arg in args if isinstance(arg, TagmeClientAdvanced)), None
            )
            token: Token = client.client.token_marker

            while max_retries > 0:
                try:
                    return await func(*args, **kwargs)
                except HTTPError as e:
                    if e.response.status_code == 401:
                        logging.warning("Токен устарел, обновляю...")
                        if token:
                            token.get_access_token()  # Обновляем токен
                            max_retries -= 1
                            continue
                        else:
                            logging.error("Токен не передан, невозможно обновить")
                            break
                    else:
                        logging.error(f"HTTP ошибка ({e.response.status_code}): {e}")
                        break
                except APIResponseError:
                    logging.warning("Задача недоступна, пропускаю...")
                    continue
                except Exception as e:
                    logging.error(f"Неизвестная ошибка: {e}")
                    break
            logging.error(f"Превышено число попыток ({max_retries})")
            return None

        return wrapper

    return decorator


class Cache:
    def __init__(self, ttl=300):
        self.cache = dict()
        self.ttl = ttl
        self.lock = asyncio.Lock()

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            async with self.lock:
                if key in self.cache:
                    cached_time, result = self.cache[key]
                    if time.time() - cached_time < self.ttl:
                        return result

            result = await func(*args, **kwargs)
            print(kwargs.get('task_id'), result)
            async with self.lock:
                if result is not None and result != (None, None, None):
                    self.cache[key] = (time.time(), result)

                self._cleanup()

            return result

        return wrapper

    def _cleanup(self):
        current_time = time.time()
        keys_to_remove = [
            key for key, (cached_time, _) in self.cache.items()
            if current_time - cached_time > self.ttl
        ]

        for key in keys_to_remove:
            del self.cache[key]
