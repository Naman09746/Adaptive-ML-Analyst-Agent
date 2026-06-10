# ama2/backend/app/utils/retry.py

import asyncio
import functools
import time
from typing import Callable, Any
from .logging import get_logger

logger = get_logger("retry_decorator")

def retry(max_attempts: int = 3, backoff: float = 2.0, exceptions: tuple[type[BaseException], ...] = (Exception,)):
    """
    Decorator that retries a synchronous or asynchronous function on failure with exponential backoff.
    
    :param max_attempts: Maximum number of times to attempt execution.
    :param backoff: Multiplier to scale the delay after each failed attempt.
    :param exceptions: Tuple of exception classes that trigger a retry.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 1
                delay = 1.0
                while attempt <= max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts:
                            logger.error("retry_exhausted", func=func.__name__, attempt=attempt, error=str(e))
                            raise
                        logger.warning(
                            "retry_attempt_failed",
                            func=func.__name__,
                            attempt=attempt,
                            next_delay=delay,
                            error=str(e)
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        delay *= backoff
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 1
                delay = 1.0
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts:
                            logger.error("retry_exhausted", func=func.__name__, attempt=attempt, error=str(e))
                            raise
                        logger.warning(
                            "retry_attempt_failed",
                            func=func.__name__,
                            attempt=attempt,
                            next_delay=delay,
                            error=str(e)
                        )
                        time.sleep(delay)
                        attempt += 1
                        delay *= backoff
            return sync_wrapper
    return decorator
