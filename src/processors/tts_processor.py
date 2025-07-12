"""
Text-to-Speech プロセッサー

gTTSを使用してテキストを音声に変換します。
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, Any
import tempfile
from gtts import gTTS
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment

from genai_processors import ProcessorPart

from .base import TextProcessorBase
from ..config import get_config
from ..utils.exceptions import TTSError


class GTTSProcessor(TextProcessorBase):
    """gTTSを使用した音声合成プロセッサー"""
    
    def __init__(
        self,
        language: Optional[str] = None,
        slow: Optional[bool] = None,
        tld: str = "com"
    ):
        """
        gTTS プロセッサーの初期化
        
        Args:
            language: 音声合成の言語
            slow: ゆっくり話すかどうか
            tld: 使用するGoogleドメイン（com, co.jp など）
        """
        config = get_config()
        language = language or config.tts.language
        
        super().__init__("gtts", language)
        
        self.slow = slow if slow is not None else config.tts.slow
        self.tld = tld
        
        # メトリクスの初期化
        self.add_metric('slow', self.slow)
        self.add_metric('tld', self.tld)
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        テキストを音声に変換
        
        Args:
            input_stream: テキストを含む入力ストリーム
        
        Yields:
            音声データを含むProcessorPart
        """
        async for part in input_stream:
            # テキストの取得
            text = self._extract_text(part)
            if not text:
                continue
            
            # 音声合成の実行
            try:
                audio_path = await self._synthesize_speech(text)
                
                # 音声データの読み込み
                audio_data, sample_rate = await self._load_audio(audio_path)
                
                # ProcessorPartの作成
                yield ProcessorPart(
                    content=None,  # 音声データなのでcontentは空
                    metadata={
                        'audio_data': audio_data,
                        'sample_rate': sample_rate,
                        'audio_path': str(audio_path),
                        'text': text,
                        'processor': 'gtts',
                        'language': self.language
                    }
                )
                
            except Exception as e:
                await self.logger.error(
                    operation="tts_error",
                    message="音声合成中にエラーが発生しました",
                    error=str(e),
                    error_type=type(e).__name__,
                    text_preview=text[:100]
                )
                raise TTSError(f"音声合成に失敗しました: {e}")
    
    def _extract_text(self, part: ProcessorPart) -> Optional[str]:
        """ProcessorPartからテキストを抽出"""
        # contentにテキストが含まれている場合
        if part.content:
            return part.content.strip()
        
        # metadataにgemini_responseが含まれている場合
        if part.metadata and 'gemini_response' in part.metadata:
            response = part.metadata['gemini_response']
            if isinstance(response, dict) and 'text' in response:
                return response['text'].strip()
        
        return None
    
    async def _synthesize_speech(self, text: str) -> Path:
        """テキストを音声に変換"""
        loop = asyncio.get_event_loop()
        
        await self.logger.info(
            operation="tts_synthesis_start",
            message="音声合成を開始します",
            text_preview=text[:100],
            text_length=len(text),
            language=self.language,
            slow=self.slow
        )
        
        start_time = time.time()
        
        # 一時ファイルの作成
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # gTTSオブジェクトの作成と保存（ブロッキング操作を非同期で実行）
            await loop.run_in_executor(
                None,
                lambda: self._generate_tts(text, str(tmp_path))
            )
            
            synthesis_time = time.time() - start_time
            
            # MP3からWAVに変換
            wav_path = await self._convert_to_wav(tmp_path)
            
            # メトリクスの更新
            self.add_metric('synthesis_time', synthesis_time)
            self.add_metric('output_file_size', wav_path.stat().st_size)
            
            await self.logger.info(
                operation="tts_synthesis_complete",
                message="音声合成が完了しました",
                synthesis_time=synthesis_time,
                output_path=str(wav_path),
                file_size_bytes=wav_path.stat().st_size
            )
            
            return wav_path
            
        finally:
            # MP3ファイルの削除
            tmp_path.unlink(missing_ok=True)
    
    def _generate_tts(self, text: str, output_path: str):
        """gTTSで音声を生成（同期処理）"""
        tts = gTTS(
            text=text,
            lang=self.language,
            slow=self.slow,
            tld=self.tld
        )
        tts.save(output_path)
    
    async def _convert_to_wav(self, mp3_path: Path) -> Path:
        """MP3をWAVに変換"""
        loop = asyncio.get_event_loop()
        
        # 出力パスの生成
        wav_path = mp3_path.with_suffix('.wav')
        
        # 変換処理（ブロッキング操作を非同期で実行）
        await loop.run_in_executor(
            None,
            lambda: self._convert_audio_format(str(mp3_path), str(wav_path))
        )
        
        return wav_path
    
    def _convert_audio_format(self, input_path: str, output_path: str):
        """音声フォーマットを変換（同期処理）"""
        audio = AudioSegment.from_mp3(input_path)
        audio.export(output_path, format="wav")
    
    async def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """音声ファイルを読み込み"""
        loop = asyncio.get_event_loop()
        
        # WAVファイルの読み込み
        sample_rate, audio_data = await loop.run_in_executor(
            None,
            lambda: wav.read(str(audio_path))
        )
        
        # float32に正規化
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        return audio_data, sample_rate


class VoiceVoxProcessor(TextProcessorBase):
    """VOICEVOX を使用した音声合成プロセッサー（将来実装）"""
    
    def __init__(self, speaker_id: int = 1):
        """
        VOICEVOX プロセッサーの初期化
        
        Args:
            speaker_id: 話者ID
        """
        super().__init__("voicevox", "ja")
        self.speaker_id = speaker_id
        # TODO: VOICEVOX APIの実装
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        VOICEVOX による音声合成（将来実装）
        """
        # TODO: 実装
        async for part in input_stream:
            yield part