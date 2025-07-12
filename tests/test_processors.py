"""
プロセッサーのテスト
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import AsyncIterator

from genai_processors import ProcessorPart

from src.processors.audio_processor import AudioFileProcessor
from src.processors.stt_processor import WhisperProcessor
from src.processors.gemini_processor import GeminiChatProcessor
from src.processors.tts_processor import GTTSProcessor
from src.utils.exceptions import AudioError, WhisperError, GeminiAPIError, TTSError


async def stream_parts(parts: list) -> AsyncIterator[ProcessorPart]:
    """テスト用のProcessorPartストリームを生成"""
    for part in parts:
        yield part


class TestAudioFileProcessor:
    """AudioFileProcessorのテスト"""
    
    @pytest.mark.asyncio
    async def test_audio_file_loading(self, sample_audio_file, sample_audio_data):
        """音声ファイル読み込みのテスト"""
        processor = AudioFileProcessor()
        
        # 入力ストリームの作成
        input_parts = [ProcessorPart(content=str(sample_audio_file))]
        
        # 処理実行
        results = []
        async for part in processor(stream_parts(input_parts)):
            results.append(part)
        
        # 検証
        assert len(results) == 1
        result = results[0]
        
        assert result.metadata is not None
        assert 'audio_data' in result.metadata
        assert 'sample_rate' in result.metadata
        assert result.metadata['sample_rate'] == 16000
        
        # 音声データの確認
        loaded_audio = result.metadata['audio_data']
        assert isinstance(loaded_audio, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_audio_file_not_found(self):
        """存在しないファイルのエラーテスト"""
        processor = AudioFileProcessor()
        
        input_parts = [ProcessorPart(content="non_existent.wav")]
        
        with pytest.raises(AudioError) as exc_info:
            async for _ in processor(stream_parts(input_parts)):
                pass
        
        assert "音声ファイルが見つかりません" in str(exc_info.value)


class TestWhisperProcessor:
    """WhisperProcessorのテスト"""
    
    @pytest.mark.asyncio
    async def test_whisper_transcription(self, sample_audio_data):
        """Whisper音声認識のテスト"""
        processor = WhisperProcessor(model_name="tiny")
        
        # Whisperモデルのモック
        mock_model = Mock()
        mock_result = {
            'text': 'テスト音声の認識結果です',
            'language': 'ja',
            'segments': []
        }
        mock_model.transcribe.return_value = mock_result
        
        with patch('whisper.load_model', return_value=mock_model):
            # 入力ストリームの作成
            input_parts = [ProcessorPart(
                content=None,
                metadata={'audio_data': sample_audio_data}
            )]
            
            # 処理実行
            results = []
            async for part in processor(stream_parts(input_parts)):
                results.append(part)
            
            # 検証
            assert len(results) == 1
            result = results[0]
            
            assert result.content == 'テスト音声の認識結果です'
            assert result.metadata['language'] == 'ja'
            assert result.metadata['processor'] == 'whisper'
    
    @pytest.mark.asyncio
    async def test_whisper_model_loading(self):
        """Whisperモデル読み込みのテスト"""
        processor = WhisperProcessor(model_name="tiny", device="cpu")
        
        with patch('whisper.load_model') as mock_load:
            mock_load.return_value = Mock()
            
            # モデルの読み込みをトリガー
            await processor._load_model()
            
            # モデル読み込みが呼ばれたことを確認
            mock_load.assert_called_once_with("tiny", device="cpu")


class TestGeminiChatProcessor:
    """GeminiChatProcessorのテスト"""
    
    @pytest.mark.asyncio
    async def test_gemini_response(self, sample_text, sample_gemini_response):
        """Gemini API応答のテスト"""
        processor = GeminiChatProcessor()
        
        # Gemini APIのモック
        mock_response = Mock()
        mock_response.text = sample_gemini_response['text']
        
        mock_session = Mock()
        mock_session.send_message.return_value = mock_response
        
        mock_model = Mock()
        mock_model.start_chat.return_value = mock_session
        
        with patch('google.generativeai.GenerativeModel', return_value=mock_model), \
             patch('google.generativeai.configure'):
            
            # モデルの初期化
            await processor._initialize_model()
            
            # 入力ストリームの作成
            input_parts = [ProcessorPart(content=sample_text)]
            
            # 処理実行
            results = []
            async for part in processor(stream_parts(input_parts)):
                results.append(part)
            
            # 検証
            assert len(results) == 1
            result = results[0]
            
            assert result.content == sample_gemini_response['text']
            assert result.metadata['processor'] == 'gemini'
            assert result.metadata['input_text'] == sample_text
    
    @pytest.mark.asyncio
    async def test_gemini_api_error(self):
        """Gemini APIエラーのテスト"""
        processor = GeminiChatProcessor()
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = Mock()
            mock_session = Mock()
            mock_session.send_message.side_effect = Exception("API Error")
            mock_model.start_chat.return_value = mock_session
            mock_model_class.return_value = mock_model
            
            with patch('google.generativeai.configure'):
                await processor._initialize_model()
                
                input_parts = [ProcessorPart(content="テスト")]
                
                with pytest.raises(GeminiAPIError) as exc_info:
                    async for _ in processor(stream_parts(input_parts)):
                        pass
                
                assert "Gemini API呼び出しに失敗しました" in str(exc_info.value)


class TestGTTSProcessor:
    """GTTSProcessorのテスト"""
    
    @pytest.mark.asyncio
    async def test_tts_synthesis(self, sample_text, temp_dir):
        """gTTS音声合成のテスト"""
        processor = GTTSProcessor(language="ja")
        
        # gTTSのモック
        mock_tts = Mock()
        
        # AudioSegmentのモック
        mock_audio = Mock()
        
        with patch('gtts.gTTS', return_value=mock_tts), \
             patch('pydub.AudioSegment.from_mp3', return_value=mock_audio), \
             patch('scipy.io.wavfile.read', return_value=(16000, np.zeros(16000))), \
             patch('tempfile.NamedTemporaryFile'):
            
            # 入力ストリームの作成
            input_parts = [ProcessorPart(content=sample_text)]
            
            # 処理実行
            results = []
            async for part in processor(stream_parts(input_parts)):
                results.append(part)
            
            # 検証
            assert len(results) == 1
            result = results[0]
            
            assert result.content is None  # 音声データなのでcontentは空
            assert 'audio_data' in result.metadata
            assert result.metadata['text'] == sample_text
            assert result.metadata['processor'] == 'gtts'
            assert result.metadata['language'] == 'ja'
            
            # gTTSが呼ばれたことを確認
            mock_tts.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tts_error_handling(self):
        """TTS エラーハンドリングのテスト"""
        processor = GTTSProcessor()
        
        with patch('gtts.gTTS', side_effect=Exception("TTS Error")):
            input_parts = [ProcessorPart(content="テスト")]
            
            with pytest.raises(TTSError) as exc_info:
                async for _ in processor(stream_parts(input_parts)):
                    pass
            
            assert "音声合成に失敗しました" in str(exc_info.value)