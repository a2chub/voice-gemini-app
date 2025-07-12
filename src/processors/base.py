"""
プロセッサー基底クラス

genai-processorsのProcessorインターフェースを実装する基底クラスを提供します。
"""

from abc import abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
import time

from genai_processors import Processor, ProcessorPart

from ..utils.logger import get_logger
from ..utils.exceptions import ProcessorError


class BaseProcessor(Processor):
    """Voice Gemini App用の基底プロセッサークラス"""
    
    def __init__(self, name: str):
        """
        基底プロセッサーの初期化
        
        Args:
            name: プロセッサー名
        """
        self.name = name
        self.logger = get_logger()
        self._metrics: Dict[str, Any] = {}
    
    async def call(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        プロセッサーのメイン処理
        
        Args:
            input_stream: 入力ストリーム
        
        Yields:
            処理結果のProcessorPart
        """
        await self.logger.info(
            f"{self.name}プロセッサーを開始します",
            operation=f"{self.name}_start"
        )
        
        start_time = time.time()
        
        try:
            # サブクラスで実装される処理を実行
            async for result in self.process(input_stream):
                yield result
            
            # メトリクスの記録
            self._metrics['processing_time'] = time.time() - start_time
            await self._log_metrics()
            
        except Exception as e:
            await self.logger.error(
                f"{self.name}プロセッサーでエラーが発生しました",
                operation=f"{self.name}_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise ProcessorError(
                f"{self.name}プロセッサーでエラーが発生しました",
                error_code=f"{self.name.upper()}_ERROR",
                details={'original_error': str(e)}
            )
    
    async def __call__(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        プロセッサーのメイン処理（互換性のため）
        
        Args:
            input_stream: 入力ストリーム
        
        Yields:
            処理結果のProcessorPart
        """
        async for part in self.call(input_stream):
            yield part
    
    @abstractmethod
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        実際の処理を実装するメソッド（サブクラスで実装）
        
        Args:
            input_stream: 入力ストリーム
        
        Yields:
            処理結果のProcessorPart
        """
        pass
    
    async def _log_metrics(self):
        """プロセッサーのメトリクスをログに記録"""
        await self.logger.info(
            f"{self.name}プロセッサーのメトリクス",
            operation=f"{self.name}_metrics",
            processor=self.name,
            **self._metrics
        )
    
    def add_metric(self, key: str, value: Any):
        """メトリクスを追加"""
        self._metrics[key] = value


class AudioProcessorBase(BaseProcessor):
    """音声処理に特化した基底プロセッサークラス"""
    
    def __init__(self, name: str, sample_rate: int = 16000):
        """
        音声処理プロセッサーの初期化
        
        Args:
            name: プロセッサー名
            sample_rate: サンプリングレート
        """
        super().__init__(name)
        self.sample_rate = sample_rate
        self.add_metric('sample_rate', sample_rate)


class TextProcessorBase(BaseProcessor):
    """テキスト処理に特化した基底プロセッサークラス"""
    
    def __init__(self, name: str, language: str = "ja"):
        """
        テキスト処理プロセッサーの初期化
        
        Args:
            name: プロセッサー名
            language: 処理言語
        """
        super().__init__(name)
        self.language = language
        self.add_metric('language', language)