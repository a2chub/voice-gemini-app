# トラブルシューティングガイド - Voice Gemini App

このドキュメントは、Voice Gemini App使用時に発生する可能性のある問題と解決方法を説明します。

## 一般的な問題

### アプリケーションが起動しない

**症状**: `python src/main.py`を実行してもエラーが表示される

**原因と解決策**:

1. **Python バージョンの確認**
   ```bash
   python --version
   # Python 3.10以上であることを確認
   ```

2. **仮想環境の有効化**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **依存関係の再インストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **モジュールパスの確認**
   ```bash
   # PYTHONPATH の設定
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

### 環境変数が読み込まれない

**症状**: APIキーエラーや設定が反映されない

**解決策**:
```bash
# .envファイルの存在確認
ls -la .env

# .envファイルの内容確認（APIキーは隠す）
cat .env | grep -v API_KEY

# python-dotenvの確認
pip show python-dotenv
```

## 音声入力の問題

### マイクが認識されない

**症状**: 「音声デバイスが見つかりません」エラー

**解決策**:

1. **デバイスリストの確認**
   ```bash
   python -m sounddevice
   ```

2. **OS別の対処法**

   **macOS**:
   - システム環境設定 → セキュリティとプライバシー → マイク
   - ターミナルまたはPythonにマイクアクセスを許可

   **Windows**:
   - 設定 → プライバシー → マイク
   - アプリにマイクへのアクセスを許可

   **Linux**:
   ```bash
   # PulseAudioの確認
   pactl list sources
   
   # ALSAの確認
   arecord -l
   ```

3. **デバイス指定**
   ```python
   # 特定のデバイスを使用
   import sounddevice as sd
   sd.default.device = [0, 1]  # [input, output]
   ```

### 録音ができない/音声が小さい

**症状**: 録音は完了するが、音声が認識されない

**解決策**:

1. **録音レベルの確認**
   ```python
   import sounddevice as sd
   import numpy as np
   
   def audio_callback(indata, frames, time, status):
       volume_norm = np.linalg.norm(indata) * 10
       print("|" * int(volume_norm))  # 音量をビジュアル表示
   
   # 音量モニター
   with sd.InputStream(callback=audio_callback):
       sd.sleep(10000)
   ```

2. **ゲイン調整**
   - OS の音声設定でマイクの入力レベルを調整
   - 外部マイクの場合はゲインコントロールを確認

## 音声認識（Whisper）の問題

### Whisperモデルのダウンロードが失敗する

**症状**: モデルダウンロード中にエラー

**解決策**:

1. **手動ダウンロード**
   ```bash
   # キャッシュディレクトリの確認
   ls ~/.cache/whisper/
   
   # 手動でモデルをダウンロード
   import whisper
   whisper.load_model("base", download_root="~/.cache/whisper")
   ```

2. **ディスク容量の確認**
   ```bash
   df -h ~/.cache/
   ```

3. **ネットワークプロキシ設定**
   ```bash
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

### 音声認識の精度が低い

**症状**: 日本語が正しく認識されない

**解決策**:

1. **言語設定の確認**
   ```python
   # Whisperの言語指定
   result = model.transcribe(audio, language="ja")
   ```

2. **モデルサイズの変更**
   ```python
   # より大きなモデルを使用
   model = whisper.load_model("small")  # または "medium"
   ```

3. **音声前処理**
   ```python
   import scipy.signal
   
   # ノイズ除去
   filtered = scipy.signal.wiener(audio_data)
   ```

## Gemini API の問題

### API キーエラー

**症状**: `Invalid API key`エラー

**解決策**:

1. **環境変数の確認**
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   api_key = os.getenv("GEMINI_API_KEY")
   print(f"API Key length: {len(api_key) if api_key else 0}")
   ```

2. **APIキーの再生成**
   - [Google AI Studio](https://makersuite.google.com/app/apikey)で新しいキーを生成

### レート制限エラー

**症状**: `429 Too Many Requests`エラー

**解決策**:

1. **リトライロジックの実装**
   ```python
   import time
   from tenacity import retry, wait_exponential, stop_after_attempt
   
   @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
          stop=stop_after_attempt(3))
   def call_gemini_api(prompt):
       # API呼び出し
       pass
   ```

2. **使用量の確認**
   - Google Cloud Consoleで使用状況を確認

### 応答が遅い

**症状**: Gemini APIの応答に時間がかかる

**解決策**:

1. **タイムアウト設定**
   ```python
   # タイムアウトを設定
   response = model.generate_content(
       prompt,
       generation_config=genai.types.GenerationConfig(
           max_output_tokens=1024,  # トークン数を制限
       )
   )
   ```

2. **非同期処理の最適化**
   ```python
   # 並列処理で高速化
   async def process_multiple(prompts):
       tasks = [process_single(p) for p in prompts]
       return await asyncio.gather(*tasks)
   ```

## 音声出力の問題

### 音声が再生されない

**症状**: TTSは成功するが音声が聞こえない

**解決策**:

1. **音声デバイスの確認**
   ```python
   import simpleaudio as sa
   
   # テスト音声の再生
   wave_obj = sa.WaveObject.from_wave_file("test.wav")
   play_obj = wave_obj.play()
   play_obj.wait_done()
   ```

2. **音声形式の確認**
   ```bash
   # ファイル形式の確認
   file test_output.wav
   
   # ffmpegでの変換
   ffmpeg -i input.mp3 -acodec pcm_s16le -ar 44100 output.wav
   ```

### 日本語の発音が不自然

**症状**: gTTSの日本語音声が聞き取りにくい

**解決策**:

1. **TTS設定の調整**
   ```python
   from gtts import gTTS
   
   # ゆっくり話す設定
   tts = gTTS(text=text, lang='ja', slow=True)
   ```

2. **代替TTSエンジン**（将来実装）
   - Google Cloud Text-to-Speech API
   - Amazon Polly
   - VOICEVOX

## ロギングの問題

### ログファイルが作成されない

**症状**: `logs/`ディレクトリにファイルがない

**解決策**:

1. **ディレクトリ権限の確認**
   ```bash
   ls -la logs/
   chmod 755 logs/
   ```

2. **ロガー設定の確認**
   ```python
   # ロガーの手動テスト
   from vibelogger import create_file_logger
   
   logger = create_file_logger(
       "test_logger",
       log_dir="logs",
       project_name="voice_gemini_app"
   )
   logger.info("Test log message")
   ```

### ログファイルが大きすぎる

**症状**: ディスク容量を圧迫

**解決策**:

1. **ログローテーション設定**
   ```env
   # .env ファイル
   LOG_MAX_SIZE_MB=50
   LOG_RETENTION_DAYS=3
   ```

2. **ログレベルの調整**
   ```env
   LOG_LEVEL=WARNING  # INFO から WARNING に変更
   ```

## パフォーマンスの問題

### メモリ使用量が多い

**症状**: アプリケーションがメモリを大量に使用

**解決策**:

1. **Whisperモデルのアンロード**
   ```python
   # 使用後にモデルを削除
   del model
   import gc
   gc.collect()
   ```

2. **音声ファイルの削除**
   ```python
   # 一時ファイルの削除
   import tempfile
   import atexit
   
   temp_dir = tempfile.mkdtemp()
   atexit.register(lambda: shutil.rmtree(temp_dir))
   ```

### 処理が遅い

**症状**: 音声処理に時間がかかる

**解決策**:

1. **プロファイリング**
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   # 処理実行
   profiler.disable()
   
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)
   ```

2. **並列処理の活用**
   ```python
   # ProcessPoolExecutorで並列化
   from concurrent.futures import ProcessPoolExecutor
   
   with ProcessPoolExecutor() as executor:
       results = executor.map(process_audio, audio_files)
   ```

## デバッグのヒント

### 詳細ログの有効化

```bash
# 環境変数でデバッグモード
export LOG_LEVEL=DEBUG
export PYTHONASYNCIODEBUG=1

# または起動時に指定
python src/main.py --debug
```

### 問題の切り分け

1. **各コンポーネントの個別テスト**
   ```bash
   # 音声録音のテスト
   python tests/test_audio_recording.py
   
   # Whisperのテスト
   python tests/test_whisper.py
   
   # Gemini APIのテスト
   python tests/test_gemini.py
   
   # TTSのテスト
   python tests/test_tts.py
   ```

2. **ログの解析**
   ```bash
   # エラーログの抽出
   grep ERROR logs/*.log
   
   # 特定の処理の追跡
   grep -A 5 -B 5 "audio_recording" logs/*.log
   ```

## サポートとコミュニティ

問題が解決しない場合:

1. [GitHub Issues](https://github.com/[your-username]/voice-gemini-app/issues)で報告
2. ログファイルを添付（APIキーは削除）
3. 実行環境の詳細を記載

## 更新履歴

- 2025-07-12: 初版作成（voice-gemini-app.mdから分離）

## Pydantic v2 関連の問題

### BaseSettings のインポートエラー

**症状**: 
```
ImportError: `BaseSettings` has been moved to the `pydantic-settings` package
```

**原因**: Pydantic v2では`BaseSettings`が別パッケージに移動された

**解決策**:

1. **pydantic-settings パッケージのインストール**
   ```bash
   pip install pydantic-settings
   ```

2. **インポート文の修正**
   ```python
   # 変更前
   from pydantic import BaseSettings, Field, validator
   
   # 変更後
   from pydantic_settings import BaseSettings, SettingsConfigDict
   from pydantic import Field, field_validator
   ```

3. **バリデータの更新**
   ```python
   # 変更前
   @validator("model")
   def validate_model(cls, v):
       ...
   
   # 変更後
   @field_validator("model")
   @classmethod
   def validate_model(cls, v):
       ...
   ```

4. **設定クラスの更新**
   ```python
   # 変更前
   class Config:
       env_prefix = "AUDIO_"
   
   # 変更後
   model_config = SettingsConfigDict(
       env_prefix="AUDIO_",
       env_file=".env",
       env_file_encoding="utf-8",
       extra="ignore"  # 他の環境変数を無視
   )
   ```

### 環境変数の検証エラー

**症状**: 
```
Extra inputs are not permitted [type=extra_forbidden]
```

**原因**: 各設定クラスが他のクラスの環境変数も読み込もうとしている

**解決策**:
```python
# model_config に extra="ignore" を追加
model_config = SettingsConfigDict(
    env_prefix="GEMINI_",
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore"  # 重要：他の環境変数を無視
)
```

## genai-processors 関連の問題

### create_part 関数が存在しない

**症状**: 
```
ImportError: cannot import name 'create_part' from 'genai_processors'
```

**原因**: genai-processors APIの変更

**解決策**:
```python
# 変更前
from genai_processors import ProcessorPart, create_part

yield create_part(
    content=text,
    metadata={...}
)

# 変更後
from genai_processors import ProcessorPart

yield ProcessorPart(
    content=text,
    metadata={...}
)
```

### Processor の call メソッドエラー

**症状**: 
```
Can't instantiate abstract class AudioFileProcessor without an implementation for abstract method 'call'
```

**原因**: genai-processors の Processor クラスは `call` メソッドを要求

**解決策**:
```python
# BaseProcessor クラスで両方のメソッドを実装
class BaseProcessor(Processor):
    async def call(self, input_stream):
        """genai-processors が要求するメソッド"""
        # 実装
    
    async def __call__(self, input_stream):
        """互換性のためのメソッド"""
        async for part in self.call(input_stream):
            yield part
```

### パイプライン演算子エラー

**症状**: 
```
unsupported operand type(s) for >>: 'AudioFileProcessor' and 'WhisperProcessor'
```

**原因**: >> 演算子が正しく動作しない

**解決策**:
```python
# 変更前
from genai_processors import streams

self.pipeline = (
    self.audio_processor >> 
    self.stt_processor >> 
    self.gemini_processor >> 
    self.tts_processor
)

# 変更後
from genai_processors import streams, chain

self.pipeline = chain([
    self.audio_processor,
    self.stt_processor,
    self.gemini_processor,
    self.tts_processor
])
```

## vibe-logger 関連の問題

### create_file_logger の引数エラー

**症状**: 
```
create_file_logger() got an unexpected keyword argument 'log_dir'
```

**原因**: vibe-logger API が期待と異なる

**解決策**:
```python
# 変更前
logger = create_file_logger(
    name,
    log_dir=str(log_dir),
    project_name=config.log.project_name,
    level=config.log.level,
    max_size_mb=config.log.max_size_mb,
    retention_days=config.log.retention_days
)

# 変更後（project_name のみ指定）
logger = create_file_logger(config.log.project_name)

# ログレベルは別途設定
if hasattr(logger, '_logger'):
    logger._logger.setLevel(log_level)
```

## 依存関係の問題

### google-generativeai がインストールされていない

**症状**: 
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**解決策**:
```bash
pip install google-generativeai
```

### その他の必要なパッケージ

**症状**: 各種インポートエラー

**解決策**:
```bash
# 音声合成と音声認識
pip install gtts openai-whisper

# 音声処理
pip install pydub simpleaudio

# すべての依存関係を一度に
pip install -r requirements.txt
```

## デバッグのヒント

### 段階的な問題解決

1. **依存関係の確認**
   ```bash
   pip list | grep -E "pydantic|genai|vibe|google"
   ```

2. **設定ファイルの確認**
   ```bash
   # .env ファイルの存在確認
   ls -la .env
   
   # 環境変数の確認（APIキーは隠す）
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GEMINI_API_KEY exists:', bool(os.getenv('GEMINI_API_KEY')))"
   ```

3. **個別モジュールのテスト**
   ```python
   # 設定の読み込みテスト
   python -c "from src.config import get_config; config = get_config(); print('Config loaded successfully')"
   
   # プロセッサーのインポートテスト
   python -c "from src.processors.audio_processor import AudioFileProcessor; print('Processor imported successfully')"
   ```

### よくある落とし穴

1. **仮想環境の確認**: 必ず仮想環境内で実行する
2. **Python バージョン**: 3.9以上が必要（genai-processors の要件）
3. **相対インポート**: `-m` オプションでモジュールとして実行する
   ```bash
   # 正しい
   python -m src.main
   
   # 間違い（相対インポートエラーになる）
   python src/main.py
   ```