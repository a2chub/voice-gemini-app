"""
Speech-to-Text プロセッサー

Whisperを使用して音声をテキストに変換します。
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any
import numpy as np
import whisper
import tempfile
from pathlib import Path
import scipy.io.wavfile as wav

from genai_processors import ProcessorPart

from .base import BaseProcessor
from ..config import get_config
from ..utils.exceptions import TranscriptionError, WhisperError


class WhisperProcessor(BaseProcessor):
    """Whisperを使用した音声認識プロセッサー"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Whisperプロセッサーの初期化
        
        Args:
            model_name: Whisperモデル名（tiny, base, small, medium, large）
            device: 使用デバイス（cpu, cuda）
            language: 認識言語（ja, en など）
        """
        super().__init__("whisper_stt")
        
        config = get_config()
        self.model_name = model_name or config.whisper.model
        self.device = device or config.whisper.device
        self.language = language or config.whisper.language
        
        # モデルの遅延読み込み用
        self._model = None
        
        # メトリクスの初期化
        self.add_metric('model_name', self.model_name)
        self.add_metric('device', self.device)
        self.add_metric('language', self.language)
    
    async def _load_model(self):
        """Whisperモデルを非同期で読み込み"""
        if self._model is None:
            await self.logger.info(
                f"Whisperモデル '{self.model_name}' を読み込んでいます",
                operation="whisper_model_loading",
                model=self.model_name,
                device=self.device
            )
            
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            # モデルの読み込み（CPU集約的な処理を非同期で実行）
            self._model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model(self.model_name, device=self.device)
            )
            
            load_time = time.time() - start_time
            self.add_metric('model_load_time', load_time)
            
            await self.logger.info(
                f"Whisperモデルの読み込みが完了しました",
                operation="whisper_model_loaded",
                model=self.model_name,
                device=self.device,
                load_time=load_time
            )
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        音声データをテキストに変換
        
        Args:
            input_stream: 音声データを含む入力ストリーム
        
        Yields:
            変換されたテキストを含むProcessorPart
        """
        # モデルの読み込み
        await self._load_model()
        
        async for part in input_stream:
            # 音声データの取得
            audio_data = self._extract_audio_data(part)
            if audio_data is None:
                continue
            
            # 音声認識の実行
            try:
                result = await self._transcribe(audio_data)
                
                # メトリクスの更新
                self.add_metric('transcription_time', result['transcription_time'])
                self.add_metric('text_length', len(result['text']))
                self.add_metric('segments_count', len(result.get('segments', [])))
                
                # ProcessorPartの作成
                yield ProcessorPart(
                    result['text'],
                    metadata={
                        'transcription': result,
                        'language': result.get('language', self.language),
                        'processor': 'whisper',
                        'model': self.model_name
                    }
                )
                
            except Exception as e:
                await self.logger.error(
                    "音声認識中にエラーが発生しました",
                    operation="whisper_transcription_error",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise WhisperError(f"音声認識に失敗しました: {e}")
    
    def _extract_audio_data(self, part: ProcessorPart) -> Optional[np.ndarray]:
        """ProcessorPartから音声データを抽出"""
        if not part.metadata:
            return None
        
        # metadataから音声データを取得
        audio_data = part.metadata.get('audio_data')
        if audio_data is not None and isinstance(audio_data, np.ndarray):
            return audio_data
        
        return None
    
    async def _transcribe(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """音声データをテキストに変換"""
        loop = asyncio.get_event_loop()
        
        # 一時ファイルに音声データを保存（Whisperがファイルパスを要求するため）
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # WAVファイルとして保存
            sample_rate = 16000  # Whisperは16kHzを期待
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav.write(str(tmp_path), sample_rate, audio_int16)
            
            try:
                # 音声認識の実行
                start_time = time.time()
                
                result = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        str(tmp_path),
                        language=self.language,
                        verbose=False
                    )
                )
                
                transcription_time = time.time() - start_time
                
                # 結果の整形
                processed_result = {
                    'text': result['text'].strip(),
                    'language': result.get('language', self.language),
                    'segments': result.get('segments', []),
                    'transcription_time': transcription_time
                }
                
                await self.logger.info(
                    "音声認識が完了しました",
                    operation="whisper_transcription_complete",
                    text_preview=processed_result['text'][:100],
                    text_length=len(processed_result['text']),
                    transcription_time=transcription_time,
                    segments_count=len(processed_result['segments'])
                )
                
                return processed_result
                
            finally:
                # 一時ファイルの削除
                tmp_path.unlink(missing_ok=True)


class GoogleSpeechProcessor(BaseProcessor):
    """Google Cloud Speech-to-Text プロセッサー（将来実装）"""
    
    def __init__(self, language: str = "ja-JP"):
        """
        Google Speech プロセッサーの初期化
        
        Args:
            language: 認識言語
        """
        super().__init__("google_speech_stt")
        self.language = language
        # TODO: Google Cloud Speech-to-Text APIの実装
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        Google Cloud Speech-to-Text による音声認識（将来実装）
        """
        # TODO: 実装
        async for part in input_stream:
            yield part