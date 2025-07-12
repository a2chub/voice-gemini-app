"""
音声ファイル処理プロセッサー

音声ファイルを読み込み、次のプロセッサーに渡します。
"""

from pathlib import Path
from typing import AsyncIterator, Union, Dict, Any
import asyncio
import numpy as np
import scipy.io.wavfile as wav

from genai_processors import ProcessorPart

from .base import AudioProcessorBase
from ..utils.exceptions import AudioError


class AudioFileProcessor(AudioProcessorBase):
    """音声ファイルを読み込むプロセッサー"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        音声ファイルプロセッサーの初期化
        
        Args:
            sample_rate: 期待するサンプリングレート
        """
        super().__init__("audio_file", sample_rate)
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        音声ファイルを読み込んで処理
        
        Args:
            input_stream: ファイルパスを含む入力ストリーム
        
        Yields:
            音声データを含むProcessorPart
        """
        async for part in input_stream:
            # ファイルパスの取得
            file_path = self._extract_file_path(part)
            if not file_path:
                continue
            
            # 音声ファイルの読み込み
            audio_data, actual_sample_rate = await self._load_audio_file(file_path)
            
            # サンプリングレートの確認
            if actual_sample_rate != self.sample_rate:
                await self.logger.warning(
                    operation="sample_rate_mismatch",
                    message=f"サンプリングレートが異なります",
                    expected=self.sample_rate,
                    actual=actual_sample_rate,
                    file_path=str(file_path)
                )
                # TODO: リサンプリング処理を追加
            
            # メトリクスの更新
            self.add_metric('file_path', str(file_path))
            self.add_metric('duration_seconds', len(audio_data) / actual_sample_rate)
            self.add_metric('actual_sample_rate', actual_sample_rate)
            
            # ProcessorPartの作成
            yield ProcessorPart(
                content=None,  # テキストコンテンツはなし
                metadata={
                    'audio_data': audio_data,
                    'sample_rate': actual_sample_rate,
                    'file_path': str(file_path),
                    'duration': len(audio_data) / actual_sample_rate
                }
            )
    
    def _extract_file_path(self, part: ProcessorPart) -> Path:
        """ProcessorPartからファイルパスを抽出"""
        # contentにファイルパスが含まれている場合
        if part.content:
            return Path(part.content.strip())
        
        # metadataにfile_pathが含まれている場合
        if part.metadata and 'file_path' in part.metadata:
            return Path(part.metadata['file_path'])
        
        return None
    
    async def _load_audio_file(self, file_path: Path) -> tuple[np.ndarray, int]:
        """音声ファイルを非同期で読み込み"""
        if not file_path.exists():
            raise AudioError(f"音声ファイルが見つかりません: {file_path}")
        
        loop = asyncio.get_event_loop()
        
        # ファイル読み込み（ブロッキング操作を非同期で実行）
        sample_rate, audio_data = await loop.run_in_executor(
            None,
            lambda: wav.read(str(file_path))
        )
        
        # float32に正規化
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        await self.logger.info(
            operation="audio_file_loaded",
            message=f"音声ファイルを読み込みました",
            file_path=str(file_path),
            sample_rate=sample_rate,
            duration_seconds=len(audio_data) / sample_rate,
            shape=audio_data.shape
        )
        
        return audio_data, sample_rate


class AudioStreamProcessor(AudioProcessorBase):
    """リアルタイム音声ストリーム処理プロセッサー（将来実装）"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        """
        音声ストリームプロセッサーの初期化
        
        Args:
            sample_rate: サンプリングレート
            chunk_size: チャンクサイズ
        """
        super().__init__("audio_stream", sample_rate)
        self.chunk_size = chunk_size
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        音声ストリームを処理（将来実装）
        
        Args:
            input_stream: 音声データのストリーム
        
        Yields:
            処理された音声データ
        """
        # TODO: リアルタイムストリーミング処理の実装
        async for part in input_stream:
            yield part