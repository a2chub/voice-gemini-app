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

try:
    from vibe_logger import Logger
    VIBE_LOGGER_AVAILABLE = True
except ImportError:
    VIBE_LOGGER_AVAILABLE = False
    print("警告: vibe-logger がインストールされていません。標準ロギングを使用します。")

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
    
    async def log_async(self, level: str, message: str, **kwargs):
        """非同期でログを出力"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # イベントループがない場合は同期的に実行
            if VIBE_LOGGER_AVAILABLE and hasattr(self.logger, level):
                getattr(self.logger, level)(message, **kwargs)
            else:
                getattr(self.logger, level)(f"{message} | {kwargs}")
            return
        
        # vibe-loggerの場合
        if VIBE_LOGGER_AVAILABLE and hasattr(self.logger, level):
            # vibe-loggerは第一引数にメッセージを取る
            await loop.run_in_executor(
                self.executor,
                functools.partial(getattr(self.logger, level), message, **kwargs)
            )
        else:
            # 標準ロギングの場合
            log_func = getattr(self.logger, level)
            log_entry = f"{message} | {kwargs}"
            await loop.run_in_executor(
                self.executor,
                log_func,
                log_entry
            )
    
    async def debug(self, message: str, **kwargs):
        """DEBUGレベルのログを非同期で出力"""
        await self.log_async("debug", message, **kwargs)
    
    async def info(self, message: str, **kwargs):
        """INFOレベルのログを非同期で出力"""
        await self.log_async("info", message, **kwargs)
    
    async def warning(self, message: str, **kwargs):
        """WARNINGレベルのログを非同期で出力"""
        await self.log_async("warning", message, **kwargs)
    
    async def error(self, message: str, **kwargs):
        """ERRORレベルのログを非同期で出力"""
        await self.log_async("error", message, **kwargs)
    
    async def critical(self, message: str, **kwargs):
        """CRITICALレベルのログを非同期で出力"""
        await self.log_async("critical", message, **kwargs)


def create_async_logger(name: str = "voice_gemini_app") -> AsyncLoggerWrapper:
    """非同期ロガーを作成"""
    config = get_config()
    
    # ログディレクトリの作成
    log_dir = Path(config.log.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if VIBE_LOGGER_AVAILABLE:
        # vibe-loggerを使用
        logger = Logger(
            config.log.project_name,
            log_dir=str(log_dir)
        )
    else:
        # 標準のPythonロギングを使用
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.log.level.upper()))
        
        # ファイルハンドラーの設定
        if not logger.handlers:
            log_file = log_dir / f"{name}.log"
            handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    # 非同期ラッパーで包む
    return AsyncLoggerWrapper(logger)


def log_execution(func: Callable) -> Callable:
    """関数の実行をログするデコレーター（非同期関数用）"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = create_async_logger()
        start_time = time.time()
        
        await logger.info(
            f"関数実行開始: {func.__name__}",
            operation=f"function_start",
            function=func.__name__,
            args=str(args)[:100],  # 長すぎる場合は切り詰め
            kwargs=str(kwargs)[:100]
        )
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            await logger.info(
                f"関数実行成功: {func.__name__}",
                operation=f"function_success",
                function=func.__name__,
                execution_time=execution_time
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            await logger.error(
                f"関数実行エラー: {func.__name__}",
                operation=f"function_error",
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
    
    if VIBE_LOGGER_AVAILABLE:
        logger = Logger(config.log.project_name)
        logger.info(
            "Processor metrics",
            operation="processor_metrics",
            processor=processor_name,
            **metrics
        )
    else:
        # 標準ロギングを使用
        logger = logging.getLogger("voice_gemini_metrics")
        logger.info(f"Processor metrics: {processor_name} - {metrics}")


# グローバルロガーインスタンス（遅延初期化）
_logger: Optional[AsyncLoggerWrapper] = None


def get_logger() -> AsyncLoggerWrapper:
    """グローバルロガーインスタンスを取得"""
    global _logger
    if _logger is None:
        _logger = create_async_logger()
    return _logger