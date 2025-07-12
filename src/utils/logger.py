"""
ロギング設定モジュール

vibe-loggerを使用した非同期対応のロギング設定を提供します。
"""

import asyncio
import functools
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

from vibelogger import create_file_logger

from ..config import get_config


class AsyncLoggerWrapper:
    """非同期環境でvibe-loggerを使用するためのラッパー"""
    
    def __init__(self, logger, executor: Optional[ThreadPoolExecutor] = None):
        self.logger = logger
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        
        # ログレベルの設定
        config = get_config()
        log_level = getattr(logging, config.log.level.upper())
        
        # vibe-loggerの内部ロガーのレベルを設定
        if hasattr(logger, '_logger'):
            logger._logger.setLevel(log_level)
    
    async def log_async(self, level: str, **kwargs):
        """非同期でログを出力"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            functools.partial(getattr(self.logger, level), **kwargs)
        )
    
    async def debug(self, **kwargs):
        """DEBUGレベルのログを非同期で出力"""
        await self.log_async("debug", **kwargs)
    
    async def info(self, **kwargs):
        """INFOレベルのログを非同期で出力"""
        await self.log_async("info", **kwargs)
    
    async def warning(self, **kwargs):
        """WARNINGレベルのログを非同期で出力"""
        await self.log_async("warning", **kwargs)
    
    async def error(self, **kwargs):
        """ERRORレベルのログを非同期で出力"""
        await self.log_async("error", **kwargs)
    
    async def critical(self, **kwargs):
        """CRITICALレベルのログを非同期で出力"""
        await self.log_async("critical", **kwargs)


def create_async_logger(name: str = "voice_gemini_app") -> AsyncLoggerWrapper:
    """非同期ロガーを作成"""
    config = get_config()
    
    # ログディレクトリの作成
    log_dir = Path(config.log.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # vibe-loggerの作成（project_nameのみ指定）
    logger = create_file_logger(config.log.project_name)
    
    # 非同期ラッパーで包む
    return AsyncLoggerWrapper(logger)


def log_execution(func: Callable) -> Callable:
    """関数の実行をログするデコレーター（非同期関数用）"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = create_async_logger()
        start_time = time.time()
        
        await logger.info(
            operation=f"function_start",
            message=f"関数実行開始: {func.__name__}",
            function=func.__name__,
            args=str(args)[:100],  # 長すぎる場合は切り詰め
            kwargs=str(kwargs)[:100]
        )
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            await logger.info(
                operation=f"function_success",
                message=f"関数実行成功: {func.__name__}",
                function=func.__name__,
                execution_time=execution_time
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            await logger.error(
                operation=f"function_error",
                message=f"関数実行エラー: {func.__name__}",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                traceback=traceback.format_exc()
            )
            raise
    
    return wrapper


def log_processor_metrics(processor_name: str, metrics: Dict[str, Any]):
    """プロセッサーのメトリクスをログ出力する同期関数"""
    config = get_config()
    logger = create_file_logger(config.log.project_name)
    
    logger.info(
        operation="processor_metrics",
        processor=processor_name,
        **metrics
    )


# グローバルロガーインスタンス（遅延初期化）
_logger: Optional[AsyncLoggerWrapper] = None


def get_logger() -> AsyncLoggerWrapper:
    """グローバルロガーインスタンスを取得"""
    global _logger
    if _logger is None:
        _logger = create_async_logger()
    return _logger