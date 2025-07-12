"""
カスタム例外クラス

Voice Gemini App で使用する例外クラスを定義します。
"""

from typing import Optional, Dict, Any


class VoiceGeminiError(Exception):
    """Voice Gemini Appの基底例外クラス"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AudioError(VoiceGeminiError):
    """音声処理関連のエラー"""
    pass


class RecordingError(AudioError):
    """音声録音エラー"""
    pass


class PlaybackError(AudioError):
    """音声再生エラー"""
    pass


class TranscriptionError(VoiceGeminiError):
    """音声認識エラー"""
    pass


class WhisperError(TranscriptionError):
    """Whisper関連エラー"""
    pass


class APIError(VoiceGeminiError):
    """外部API関連エラー"""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.status_code = status_code


class GeminiAPIError(APIError):
    """Gemini API関連エラー"""
    pass


class TTSError(VoiceGeminiError):
    """Text-to-Speech関連エラー"""
    pass


class ConfigurationError(VoiceGeminiError):
    """設定関連エラー"""
    pass


class ProcessorError(VoiceGeminiError):
    """プロセッサー関連エラー"""
    pass


class PipelineError(VoiceGeminiError):
    """パイプライン処理エラー"""
    pass