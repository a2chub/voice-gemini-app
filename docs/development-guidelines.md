# 開発ガイドライン - Voice Gemini App

このドキュメントは、Voice Gemini Appの開発に参加する際のガイドラインを提供します。

## コーディング規約

### Python スタイルガイド

**基本ルール**:
- PEP 8 に準拠
- 最大行長: 88文字（Black formatterのデフォルト）
- インデント: スペース4つ

**命名規則**:
```python
# クラス名: PascalCase
class AudioProcessor:
    pass

# 関数名・変数名: snake_case
def process_audio_file(file_path: str) -> bytes:
    audio_data = load_audio(file_path)
    return audio_data

# 定数: UPPER_SNAKE_CASE
MAX_RECORDING_SECONDS = 30
DEFAULT_SAMPLE_RATE = 16000

# プライベート: アンダースコアプレフィックス
def _internal_function():
    pass
```

### 型ヒント

すべての関数に型ヒントを付ける：

```python
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import numpy as np

def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "ja",
    model_size: str = "base"
) -> Tuple[str, Dict[str, float]]:
    """
    音声ファイルをテキストに変換する
    
    Args:
        audio_path: 音声ファイルのパス
        language: 言語コード
        model_size: Whisperモデルのサイズ
    
    Returns:
        変換されたテキストとメタデータのタプル
    """
    pass
```

### Docstring

Google スタイルの docstring を使用：

```python
def process_voice_input(
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> ProcessingResult:
    """
    音声入力を処理してAI応答を生成する
    
    Args:
        audio_data: 音声データの numpy 配列
        sample_rate: サンプリングレート（Hz）
    
    Returns:
        ProcessingResult: 処理結果を含むオブジェクト
    
    Raises:
        AudioProcessingError: 音声処理に失敗した場合
        APIError: 外部API呼び出しに失敗した場合
    
    Example:
        >>> audio = record_audio(duration=5)
        >>> result = process_voice_input(audio)
        >>> print(result.transcribed_text)
    """
    pass
```

## プロジェクト構造

### ディレクトリ構成

```
voice-gemini-app/
├── src/
│   ├── __init__.py
│   ├── main.py              # エントリーポイント
│   ├── audio/               # 音声処理モジュール
│   │   ├── __init__.py
│   │   ├── recorder.py      # 音声録音
│   │   ├── player.py        # 音声再生
│   │   └── processor.py     # 音声前処理
│   ├── processors/          # genai-processors実装
│   │   ├── __init__.py
│   │   ├── base.py          # 基底クラス
│   │   ├── stt.py           # Speech-to-Text
│   │   ├── llm.py           # LLM (Gemini)
│   │   └── tts.py           # Text-to-Speech
│   ├── models/              # データモデル
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   └── conversation.py
│   ├── utils/               # ユーティリティ
│   │   ├── __init__.py
│   │   ├── config.py        # 設定管理
│   │   ├── logger.py        # ロギング設定
│   │   └── exceptions.py    # カスタム例外
│   └── api/                 # API関連（将来実装）
│       ├── __init__.py
│       └── endpoints.py
├── tests/                   # テストコード
│   ├── __init__.py
│   ├── conftest.py          # pytest設定
│   ├── unit/                # 単体テスト
│   ├── integration/         # 統合テスト
│   └── fixtures/            # テストデータ
├── docs/                    # ドキュメント
├── scripts/                 # ユーティリティスクリプト
├── configs/                 # 設定ファイル
└── notebooks/               # Jupyterノートブック（実験用）
```

### モジュール設計原則

1. **単一責任の原則**: 各モジュールは一つの明確な責任を持つ
2. **依存性の注入**: ハードコーディングを避け、設定可能にする
3. **インターフェース分離**: 抽象基底クラスを使用してインターフェースを定義

```python
# src/processors/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart

class BaseAudioProcessor(Processor, ABC):
    """音声処理プロセッサーの基底クラス"""
    
    @abstractmethod
    async def process(
        self, 
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """処理を実行する抽象メソッド"""
        pass
```

## 非同期プログラミング

### 基本パターン

```python
import asyncio
from typing import List, AsyncIterator

async def process_audio_async(audio_path: str) -> str:
    """非同期で音声を処理"""
    # CPU バウンドなタスクは run_in_executor を使用
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, transcribe_sync, audio_path)
    return text

async def process_multiple_files(file_paths: List[str]) -> List[str]:
    """複数ファイルを並行処理"""
    tasks = [process_audio_async(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return results
```

### ストリーミング処理

```python
async def stream_audio_chunks() -> AsyncIterator[bytes]:
    """音声チャンクをストリーミング"""
    chunk_size = 1024
    with open("audio.wav", "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk
            await asyncio.sleep(0)  # 他のタスクに制御を渡す
```

## エラーハンドリング

### カスタム例外

```python
# src/utils/exceptions.py
class VoiceGeminiError(Exception):
    """基底例外クラス"""
    pass

class AudioProcessingError(VoiceGeminiError):
    """音声処理エラー"""
    pass

class TranscriptionError(AudioProcessingError):
    """音声認識エラー"""
    pass

class APIError(VoiceGeminiError):
    """外部API関連エラー"""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code
```

### エラーハンドリングパターン

```python
async def safe_process_audio(audio_path: str) -> Optional[str]:
    """エラーを適切に処理する例"""
    try:
        result = await process_audio(audio_path)
        return result
    except AudioProcessingError as e:
        logger.error(
            "音声処理エラー",
            error=str(e),
            audio_path=audio_path,
            traceback=traceback.format_exc()
        )
        # リトライロジック
        return await retry_with_backoff(process_audio, audio_path)
    except Exception as e:
        logger.error(
            "予期しないエラー",
            error=str(e),
            audio_path=audio_path
        )
        raise
```

## ロギング規約

### vibe-logger の使用

```python
from vibelogger import create_file_logger
import functools

# グローバルロガーの設定
logger = create_file_logger(
    "voice_gemini_app",
    log_dir="logs",
    project_name="voice_gemini_app"
)

# デコレーターでログを自動化
def log_execution(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(
            f"関数実行開始: {func.__name__}",
            function=func.__name__,
            args=str(args)[:100],  # 長すぎる場合は切り詰め
            kwargs=str(kwargs)[:100]
        )
        try:
            result = await func(*args, **kwargs)
            logger.info(
                f"関数実行成功: {func.__name__}",
                function=func.__name__,
                execution_time=time.time() - start_time
            )
            return result
        except Exception as e:
            logger.error(
                f"関数実行エラー: {func.__name__}",
                function=func.__name__,
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
    return wrapper
```

### ログレベルとコンテキスト

```python
# 構造化ログの例
logger.info(
    "Gemini API呼び出し",
    operation="gemini_api_call",
    prompt_length=len(prompt),
    model="gemini-pro",
    temperature=0.7,
    user_id=user_id  # トレーサビリティのため
)

# メトリクスの記録
logger.info(
    "パフォーマンスメトリクス",
    operation="performance_metrics",
    transcription_time=1.23,
    api_latency=0.45,
    tts_time=0.67,
    total_time=2.35,
    audio_duration=5.0
)
```

## テスト

### テスト構造

```python
# tests/unit/test_audio_recorder.py
import pytest
from unittest.mock import Mock, patch
import numpy as np

class TestAudioRecorder:
    """AudioRecorderの単体テスト"""
    
    @pytest.fixture
    def recorder(self):
        """テスト用レコーダーのフィクスチャ"""
        from src.audio.recorder import AudioRecorder
        return AudioRecorder(sample_rate=16000)
    
    @pytest.mark.asyncio
    async def test_record_audio(self, recorder):
        """音声録音のテスト"""
        # モックデータの準備
        mock_audio = np.random.rand(16000 * 3)  # 3秒分
        
        with patch('sounddevice.rec') as mock_rec:
            mock_rec.return_value = mock_audio
            
            result = await recorder.record(duration=3)
            
            assert result.shape == (16000 * 3,)
            assert result.dtype == np.float32
```

### 統合テスト

```python
# tests/integration/test_pipeline.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline():
    """完全なパイプラインのテスト"""
    # テスト用音声ファイルを使用
    test_audio = "tests/fixtures/test_audio.wav"
    
    # パイプラインの実行
    result = await process_voice_pipeline(test_audio)
    
    # 結果の検証
    assert result.transcribed_text
    assert result.ai_response
    assert result.audio_output
    assert result.total_time < 10  # 10秒以内に完了
```

## パフォーマンス最適化

### プロファイリング

```python
import cProfile
import asyncio
from memory_profiler import profile

@profile  # メモリプロファイリング
async def memory_intensive_function():
    # 大量のデータを処理
    pass

# 実行時間のプロファイリング
async def profile_async_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    await your_async_function()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

### 最適化のヒント

1. **キャッシング**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_computation(param):
       # 計算コストの高い処理
       pass
   ```

2. **バッチ処理**
   ```python
   async def process_in_batches(items: List, batch_size: int = 10):
       for i in range(0, len(items), batch_size):
           batch = items[i:i + batch_size]
           await asyncio.gather(*[process_item(item) for item in batch])
   ```

## セキュリティ

### APIキーの管理

```python
# 環境変数からの読み込み
import os
from typing import Optional

def get_api_key(key_name: str) -> Optional[str]:
    """APIキーを安全に取得"""
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"{key_name}が設定されていません")
    return key

# 設定ファイルでの検証
from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    gemini_api_key: SecretStr
    
    class Config:
        env_file = ".env"
```

### 入力検証

```python
import re
from pathlib import Path

def validate_audio_file(file_path: str) -> Path:
    """音声ファイルパスの検証"""
    path = Path(file_path)
    
    # パストラバーサル攻撃の防止
    if ".." in str(path):
        raise ValueError("無効なファイルパス")
    
    # 拡張子の確認
    if path.suffix.lower() not in ['.wav', '.mp3', '.m4a']:
        raise ValueError("サポートされていないファイル形式")
    
    # ファイルサイズの確認
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        raise ValueError("ファイルサイズが大きすぎます")
    
    return path
```

## CI/CD

### GitHub Actions 設定例

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 貢献ガイドライン

### プルリクエストのプロセス

1. **フォークとブランチ作成**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **コミットメッセージ**
   ```
   feat: 新機能の追加
   fix: バグ修正
   docs: ドキュメントの更新
   refactor: リファクタリング
   test: テストの追加・修正
   ```

3. **テストの実行**
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **プルリクエストの作成**
   - 変更内容の説明
   - 関連するIssue番号
   - テスト結果のスクリーンショット（必要に応じて）

## 更新履歴

- 2025-07-12: 初版作成（voice-gemini-app.mdから分離）