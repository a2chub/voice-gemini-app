# Voice Gemini App - 音声対話型AI アプリケーション

## 概要

このアプリケーションは、genai-processorsライブラリを使用して、音声入力による対話型AIアプリケーションを実現します。ユーザーの音声を録音し、テキストに変換してGemini AIと対話し、応答を音声で返す、録音後処理方式を採用しています。

## ドキュメント構成

本プロジェクトのドキュメントは、管理と保守性を考慮して以下のファイルに分割されています：

| ドキュメント | 説明 | 更新タイミング |
|-------------|------|----------------|
| [実装計画](implementation-plan.md) | フェーズごとの機能実装計画とマイルストーン | 新機能追加時、計画変更時 |
| [実装ログ](implementation-log.md) | 実装の進捗と技術的な発見事項の記録 | 実装作業の開始・完了時 |
| [セットアップガイド](setup-guide.md) | 環境構築と初期設定の詳細手順 | 依存関係や設定変更時 |
| [使用方法ガイド](usage-guide.md) | アプリケーションの操作方法と活用例 | 新機能追加、UI変更時 |
| [トラブルシューティング](troubleshooting.md) | 問題解決のためのガイド | 新しい問題と解決策発見時 |
| [開発ガイドライン](development-guidelines.md) | コーディング規約と開発プロセス | 規約変更、ツール更新時 |

### ドキュメント運用ルール

1. **マスターファイル（本ファイル）**: プロジェクトの概要とアーキテクチャを記載
2. **個別ドキュメント**: 詳細な情報は各専門ドキュメントに記載
3. **更新方針**: 変更があった際は、該当するドキュメントのみを更新
4. **相互参照**: 各ドキュメント間で適切にリンクを設定

## アーキテクチャ

### 処理フロー（録音後処理方式）

```
1. 音声録音フェーズ（同期処理）
┌─────────────────┐     ┌─────────────────┐
│   音声入力      │────▶│  音声バッファ   │
│  (マイク)       │     │  (WAVファイル)  │
└─────────────────┘     └─────────────────┘

2. 非同期処理パイプライン（genai-processors使用）
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ AudioProcessor  │────▶│ STTProcessor    │────▶│ GeminiProcessor │
│ (音声読込)      │     │ (音声→テキスト) │     │ (AI対話)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   音声出力      │◀────│ TTSProcessor    │◀────│ ResponseBuffer  │
│  (スピーカー)   │     │ (テキスト→音声) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘

3. ロギング層（全プロセッサーから利用）
                        ┌─────────────────┐
                        │  vibe-logger    │
                        │  (非同期対応)   │
                        └─────────────────┘
```

### genai-processorsの活用方法

各処理ステップを`Processor`として実装し、非同期ストリームで連結：

```python
# 各プロセッサーの定義
audio_processor = AudioFileProcessor()  # WAVファイルを読み込み
stt_processor = WhisperProcessor()      # 音声をテキストに変換
gemini_processor = GeminiChatProcessor() # Gemini APIで対話
tts_processor = GTTSProcessor()         # テキストを音声に変換

# パイプラインの構築
pipeline = audio_processor >> stt_processor >> gemini_processor >> tts_processor
```

### ロギング設計

vibe-loggerを非同期環境で使用し、各プロセッサーから構造化ログを出力：

```python
# 非同期対応のロガーセットアップ
import asyncio
from vibelogger import create_file_logger

class AsyncLoggerWrapper:
    def __init__(self, logger):
        self.logger = logger
        self.executor = None
    
    async def log_async(self, level, **kwargs):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor, 
            getattr(self.logger, level),
            **kwargs
        )
```

**ログ対象：**
- **音声録音**: ファイルサイズ、録音時間、フォーマット
- **音声認識**: 変換結果、処理時間、使用モデル
- **Gemini通信**: プロンプト、レスポンス、トークン数、レイテンシ
- **音声合成**: 生成時間、音声長、エラー
- **パイプライン**: 各プロセッサーの実行時間、メモリ使用量

## 必要な依存関係

### Python パッケージ

```bash
# コアライブラリ
genai-processors>=0.1.0    # 非同期パイプライン処理
vibelogger>=0.1.0         # AI最適化ロギング

# 音声処理（録音）
sounddevice>=0.4.6        # クロスプラットフォーム音声録音
scipy>=1.10.0             # WAVファイル処理

# 音声認識
openai-whisper>=20230918  # OpenAI Whisper（ローカル音声認識）
# または
google-cloud-speech>=2.20.0  # Google Cloud Speech-to-Text

# 音声合成
gTTS>=2.3.0              # Google Text-to-Speech
pydub>=0.25.1            # 音声ファイル変換
simpleaudio>=1.0.4       # 音声再生

# 非同期処理
aiofiles>=23.0.0         # 非同期ファイル操作

# その他
python-dotenv>=1.0.0     # 環境変数管理
pydantic>=2.0.0          # 設定管理とバリデーション
```

### 依存関係の明確化

- **genai-processors**: Gemini APIとの通信を含む、すべての非同期処理パイプラインの基盤
- **google-generativeai**: genai-processorsが内部で使用するため、直接インストール不要
- **sounddevice**: pyaudioより安定したクロスプラットフォーム対応の録音ライブラリ

### システム要件

- Python 3.10以上
- マイクとスピーカー（音声入出力用）
- インターネット接続（Gemini API通信用）
- 対応OS: Windows, macOS, Linux

## セットアップ手順

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python -m venv venv

# 有効化 (Windows)
venv\Scripts\activate

# 有効化 (macOS/Linux)
source venv/bin/activate
```

### 2. 依存関係のインストール

```bash
# requirements.txtを使用
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env` ファイルを作成し、以下を設定：

```env
# Gemini API キー
GEMINI_API_KEY=your_gemini_api_key_here

# ロギング設定
LOG_PROJECT_NAME=voice_gemini_app
LOG_LEVEL=INFO
LOG_MAX_SIZE_MB=100  # ログファイルの最大サイズ
LOG_RETENTION_DAYS=7  # ログ保持期間

# 音声設定
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_FORMAT=wav
RECORDING_MAX_SECONDS=30  # 最大録音時間
SPEECH_LANGUAGE=ja-JP

# 音声認識設定（Whisper使用時）
WHISPER_MODEL=base  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # cpu または cuda

# Gemini設定
GEMINI_MODEL=gemini-pro
GEMINI_MAX_TOKENS=2048
GEMINI_TEMPERATURE=0.7

# TTS設定
TTS_LANGUAGE=ja
TTS_SLOW=False
```

### 4. プロジェクト構造

```
voice-gemini-app/
├── docs/
│   └── voice-gemini-app.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py
│   │   └── synthesizer.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── speech_to_text.py
│   │   ├── gemini_processor.py
│   │   └── text_to_speech.py
│   └── utils/
│       ├── __init__.py
│       ├── logger_setup.py
│       └── config.py
├── logs/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## クイックリファレンス

### 開発を始める

1. **環境構築**: [セットアップガイド](setup-guide.md)を参照
2. **実装計画の確認**: [実装計画](implementation-plan.md)で現在の進捗を確認
3. **コーディング**: [開発ガイドライン](development-guidelines.md)に従って実装
4. **実装記録**: [実装ログ](implementation-log.md)に進捗を記録

### アプリケーションを使用する

1. **基本的な使い方**: [使用方法ガイド](usage-guide.md)を参照
2. **問題が発生したら**: [トラブルシューティング](troubleshooting.md)を確認

## 主要な技術的決定事項

### アーキテクチャの選択理由

1. **録音後処理方式**: リアルタイムストリーミングより実装が簡単で、品質が安定
2. **genai-processors**: 非同期パイプライン処理により、モジュラーで拡張性の高い設計を実現
3. **vibe-logger**: AI最適化されたロギングで、問題の分析と改善が容易

### 技術スタックの選択理由

- **sounddevice**: pyaudioより安定しており、クロスプラットフォーム対応が優れている
- **Whisper**: ローカルで動作し、高精度な日本語音声認識が可能
- **gTTS**: シンプルで信頼性が高く、日本語対応も良好
- **Gemini API**: 最新のLLMで、日本語の理解と生成能力が高い

## プロジェクトの現状

- **ドキュメント**: 完成（2025-07-12）
- **実装**: 未開始
- **次のステップ**: [実装計画](implementation-plan.md)に従って基本機能から実装開始

## 貢献方法

1. このリポジトリをフォーク
2. [開発ガイドライン](development-guidelines.md)を確認
3. [実装計画](implementation-plan.md)から作業項目を選択
4. 実装後、[実装ログ](implementation-log.md)に記録
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。