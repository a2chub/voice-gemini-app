# セットアップガイド - Voice Gemini App

このドキュメントは、Voice Gemini Appの環境構築手順を詳細に説明します。

## 前提条件

### システム要件
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 20.04+
- **Python**: 3.10以上
- **メモリ**: 最小4GB（Whisper使用時は8GB推奨）
- **ストレージ**: 2GB以上の空き容量
- **その他**: マイクとスピーカー、インターネット接続

### 必要なアカウント
- Google Cloud Platform（Gemini API用）
- GitHub（ソースコード取得用）

## インストール手順

### 1. Pythonのインストール確認

```bash
# Pythonバージョン確認
python --version
# または
python3 --version

# pipの確認
pip --version
# または
pip3 --version
```

### 2. プロジェクトのクローン

```bash
# リポジトリのクローン
git clone https://github.com/[your-username]/voice-gemini-app.git
cd voice-gemini-app

# または既存のdroneAIプロジェクトを使用
cd /path/to/droneAI
```

### 3. 仮想環境のセットアップ

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 仮想環境が有効化されているか確認（プロンプトに(venv)が表示される）
```

### 4. 依存関係のインストール

```bash
# requirements.txtの作成（まだ存在しない場合）
cat > requirements.txt << EOF
# Core libraries
genai-processors>=0.1.0
vibelogger>=0.1.0

# Audio processing
sounddevice>=0.4.6
scipy>=1.10.0
numpy>=1.24.0

# Speech recognition
openai-whisper>=20230918

# Text-to-speech
gTTS>=2.3.0
pydub>=0.25.1
simpleaudio>=1.0.4

# Async processing
aiofiles>=23.0.0

# Configuration
python-dotenv>=1.0.0
pydantic>=2.0.0

# Google Gemini (installed by genai-processors)
# google-generativeai>=0.3.0
EOF

# 依存関係のインストール
pip install -r requirements.txt
```

### 5. 音声デバイスの確認

```bash
# 利用可能な音声デバイスを確認
python -m sounddevice

# 出力例:
# 0 Built-in Microphone, Core Audio (2 in, 0 out)
# 1 Built-in Output, Core Audio (0 in, 2 out)
```

### 6. 環境変数の設定

```bash
# .envファイルの作成
cat > .env << EOF
# Gemini API キー
GEMINI_API_KEY=your_gemini_api_key_here

# ロギング設定
LOG_PROJECT_NAME=voice_gemini_app
LOG_LEVEL=INFO
LOG_MAX_SIZE_MB=100
LOG_RETENTION_DAYS=7

# 音声設定
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_FORMAT=wav
RECORDING_MAX_SECONDS=30
SPEECH_LANGUAGE=ja-JP

# 音声認識設定（Whisper）
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Gemini設定
GEMINI_MODEL=gemini-pro
GEMINI_MAX_TOKENS=2048
GEMINI_TEMPERATURE=0.7

# TTS設定
TTS_LANGUAGE=ja
TTS_SLOW=False
EOF

# .envファイルの保護
echo ".env" >> .gitignore
```

### 7. Gemini API キーの取得

1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. 「Get API key」をクリック
3. 新しいAPIキーを作成
4. キーをコピーして`.env`ファイルの`GEMINI_API_KEY`に設定

### 8. プロジェクト構造の作成

```bash
# 必要なディレクトリの作成
mkdir -p src/audio src/processors src/utils logs tests

# __init__.pyファイルの作成
touch src/__init__.py
touch src/audio/__init__.py
touch src/processors/__init__.py
touch src/utils/__init__.py

# ログディレクトリの権限設定（Unix系OS）
chmod 755 logs
```

### 9. Whisperモデルの初回ダウンロード

```python
# test_whisper.py として保存
import whisper

print("Whisperモデルをダウンロード中...")
model = whisper.load_model("base")
print("ダウンロード完了！")

# テスト実行
# python test_whisper.py
```

## 動作確認

### 1. 音声録音テスト

```python
# test_audio.py として保存
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

duration = 3  # 秒
fs = 16000  # サンプリングレート

print("3秒間録音します...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("録音完了！")

# 保存
wav.write("test_recording.wav", fs, recording)
print("test_recording.wav として保存しました")
```

### 2. Gemini API接続テスト

```python
# test_gemini.py として保存
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("こんにちは")
print(response.text)
```

## トラブルシューティング（セットアップ時）

### pipインストールエラー
```bash
# pipのアップグレード
python -m pip install --upgrade pip

# キャッシュクリア
pip cache purge
```

### Whisperインストールエラー
```bash
# ffmpegが必要な場合
# macOS
brew install ffmpeg

# Ubuntu
sudo apt update && sudo apt install ffmpeg

# Windows
# ffmpeg.exeをPATHに追加
```

### 音声デバイスが見つからない
```bash
# ALSA関連（Linux）
sudo apt-get install libasound2-dev

# PortAudio関連（全OS）
# macOS
brew install portaudio

# Ubuntu
sudo apt-get install portaudio19-dev
```

## 次のステップ

セットアップが完了したら、[使用方法ガイド](usage-guide.md)を参照してアプリケーションの実行方法を確認してください。

開発を始める場合は、[開発ガイドライン](development-guidelines.md)を参照してください。

## 更新履歴

- 2025-07-12: 初版作成（voice-gemini-app.mdから分離）