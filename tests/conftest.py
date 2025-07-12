"""
pytest設定ファイル

テスト用のフィクスチャと設定を提供します。
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os
from typing import Generator, AsyncGenerator
import numpy as np
import scipy.io.wavfile as wav


@pytest.fixture(scope="session")
def event_loop():
    """非同期テスト用のイベントループ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """一時ディレクトリを作成"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """テスト用のサンプル音声データを生成"""
    # 1秒間の440Hz（A4音）のサイン波
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    return audio_data.astype(np.float32)


@pytest.fixture
def sample_audio_file(temp_dir: Path, sample_audio_data: np.ndarray) -> Path:
    """テスト用の音声ファイルを作成"""
    audio_path = temp_dir / "test_audio.wav"
    
    # int16形式で保存
    audio_int16 = (sample_audio_data * 32767).astype(np.int16)
    wav.write(str(audio_path), 16000, audio_int16)
    
    return audio_path


@pytest.fixture
def mock_env_vars(monkeypatch):
    """テスト用の環境変数を設定"""
    test_env = {
        "GEMINI_API_KEY": "test_api_key_12345",
        "LOG_PROJECT_NAME": "test_voice_gemini",
        "LOG_LEVEL": "DEBUG",
        "WHISPER_MODEL": "tiny",  # テスト用に小さいモデル
        "WHISPER_DEVICE": "cpu",
        "GEMINI_MODEL": "gemini-pro",
        "GEMINI_TEMPERATURE": "0.5",
        "TTS_LANGUAGE": "ja",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture
def mock_config(mock_env_vars, temp_dir, monkeypatch):
    """テスト用の設定を作成"""
    # ログディレクトリをテンポラリに設定
    monkeypatch.setenv("LOG_DIR", str(temp_dir / "logs"))
    
    # .envファイルの作成
    env_file = temp_dir / ".env"
    env_content = "\n".join([f"{k}={v}" for k, v in mock_env_vars.items()])
    env_file.write_text(env_content)
    
    # 作業ディレクトリを一時的に変更
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield
    
    # 元のディレクトリに戻る
    os.chdir(original_cwd)


@pytest.fixture
async def mock_logger():
    """テスト用のモックロガー"""
    class MockLogger:
        def __init__(self):
            self.logs = []
        
        async def log_async(self, level, **kwargs):
            self.logs.append({"level": level, **kwargs})
        
        async def debug(self, **kwargs):
            await self.log_async("debug", **kwargs)
        
        async def info(self, **kwargs):
            await self.log_async("info", **kwargs)
        
        async def warning(self, **kwargs):
            await self.log_async("warning", **kwargs)
        
        async def error(self, **kwargs):
            await self.log_async("error", **kwargs)
        
        async def critical(self, **kwargs):
            await self.log_async("critical", **kwargs)
    
    return MockLogger()


@pytest.fixture
def sample_text() -> str:
    """テスト用のサンプルテキスト"""
    return "こんにちは、Voice Gemini Appのテストです。"


@pytest.fixture
def sample_gemini_response() -> dict:
    """テスト用のGemini応答"""
    return {
        "text": "こんにちは！Voice Gemini Appのテストへようこそ。何かお手伝いできることはありますか？",
        "api_latency": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }