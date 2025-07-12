"""
設定管理モジュール

環境変数と設定ファイルから設定を読み込み、アプリケーション全体で使用する設定を管理します。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class AudioConfig(BaseSettings):
    """音声関連の設定"""
    sample_rate: int = Field(default=16000, alias="AUDIO_SAMPLE_RATE")
    chunk_size: int = Field(default=1024, alias="AUDIO_CHUNK_SIZE")
    format: str = Field(default="wav", alias="AUDIO_FORMAT")
    recording_max_seconds: int = Field(default=30, alias="RECORDING_MAX_SECONDS")
    speech_language: str = Field(default="ja-JP", alias="SPEECH_LANGUAGE")
    
    model_config = SettingsConfigDict(
        env_prefix="AUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class WhisperConfig(BaseSettings):
    """Whisper音声認識の設定"""
    model: str = Field(default="base", alias="WHISPER_MODEL")
    device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    language: str = Field(default="ja", alias="WHISPER_LANGUAGE")
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if v not in valid_models:
            raise ValueError(f"Invalid Whisper model: {v}. Must be one of {valid_models}")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="WHISPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class GeminiConfig(BaseSettings):
    """Gemini APIの設定"""
    api_key: str = Field(..., alias="GEMINI_API_KEY")
    model: str = Field(default="gemini-pro", alias="GEMINI_MODEL")
    max_tokens: int = Field(default=2048, alias="GEMINI_MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {v}")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class TTSConfig(BaseSettings):
    """Text-to-Speech設定"""
    language: str = Field(default="ja", alias="TTS_LANGUAGE")
    slow: bool = Field(default=False, alias="TTS_SLOW")
    
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class LogConfig(BaseSettings):
    """ロギング設定"""
    project_name: str = Field(default="voice_gemini_app", alias="LOG_PROJECT_NAME")
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_size_mb: int = Field(default=100, alias="LOG_MAX_SIZE_MB")
    retention_days: int = Field(default=7, alias="LOG_RETENTION_DAYS")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class AppConfig(BaseSettings):
    """アプリケーション全体の設定"""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    
    # アプリケーション設定
    app_name: str = Field(default="Voice Gemini App", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


def load_config() -> AppConfig:
    """設定をロードする"""
    # .envファイルの存在確認
    env_path = Path(".env")
    if not env_path.exists():
        # .env.exampleから.envを作成するよう促す
        example_path = Path(".env.example")
        if example_path.exists():
            raise FileNotFoundError(
                f".envファイルが見つかりません。\n"
                f"{example_path}を{env_path}にコピーして、必要な値を設定してください。"
            )
        else:
            raise FileNotFoundError(".envファイルが見つかりません。")
    
    try:
        config = AppConfig()
        return config
    except Exception as e:
        raise ValueError(f"設定の読み込みに失敗しました: {e}")


# グローバル設定インスタンス（遅延読み込み）
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """設定のシングルトンインスタンスを取得"""
    global _config
    if _config is None:
        _config = load_config()
    return _config