# Voice Gemini App - 音声対話型AIアプリケーション

音声入力でGoogle Gemini AIと対話できるアプリケーションです。

## 機能

- 🎤 音声録音（sounddevice）
- 🗣️ 音声認識（Whisper）
- 🤖 AI対話（Google Gemini）
- 🔊 音声合成（gTTS）
- 📊 構造化ロギング（vibe-logger）

## セットアップ

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/yourusername/droneAI.git
   cd droneAI
   ```

2. **仮想環境の作成**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **依存関係のインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **環境設定**
   ```bash
   cp .env.example .env
   # .envファイルを編集してGEMINI_API_KEYを設定
   ```

5. **Gemini APIキーの取得**
   - [Google AI Studio](https://makersuite.google.com/app/apikey)でAPIキーを取得
   - `.env`ファイルの`GEMINI_API_KEY`に設定

## 使い方

```bash
python -m src.main
```

または

```bash
python src/main.py
```

### 操作方法

- **Enter**: 録音開始（最大30秒）
- **Ctrl+C**: 録音停止
- **q/quit**: アプリケーション終了

## プロジェクト構造

```
droneAI/
├── src/
│   ├── audio/          # 音声録音・再生
│   ├── processors/     # genai-processorsプロセッサー
│   ├── utils/          # ユーティリティ（設定、ログ、例外）
│   └── main.py         # メインアプリケーション
├── docs/               # ドキュメント
├── tests/              # テストコード
└── logs/               # ログファイル
```

## トラブルシューティング

問題が発生した場合は、[トラブルシューティングガイド](docs/troubleshooting.md)を参照してください。

## 開発

開発に参加する場合は、[開発ガイドライン](docs/development-guidelines.md)を確認してください。

## ライセンス

MIT License