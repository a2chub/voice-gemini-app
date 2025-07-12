# Voice Gemini App - リアルタイム音声対話AIアプリケーション

Google Gemini AIとリアルタイムで音声対話ができるアプリケーションです。話し始めると自動的に録音を開始し、話し終わると自動的にAIが応答を生成して音声で返答します。

## 主な機能

### 🎤 リアルタイム音声認識
- **自動録音開始**: 音声を検出すると自動的に録音開始
- **自動録音停止**: 1.5秒の無音で自動的に処理を開始
- **音声レベル表示**: リアルタイムで音声レベルをビジュアル表示
- **プレバッファ機能**: 話し始める0.5秒前から録音

### 🤖 AI対話機能
- **Gemini AI統合**: Google Gemini 1.5 Flashモデルを使用
- **自然な会話**: コンテキストを保持した自然な対話
- **多言語対応**: 日本語を含む多言語に対応

### 🔊 音声合成
- **自然な音声**: gTTSを使用した自然な音声合成
- **即座の再生**: AI応答を即座に音声で再生

## 必要な環境

- Python 3.8以上
- macOS、Linux、Windows（一部機能制限あり）
- マイクとスピーカー

## セットアップ

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/yourusername/voice-gemini-app.git
   cd voice-gemini-app
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
   `.env`ファイルを作成し、以下の内容を設定：
   ```env
   # Gemini API キー（必須）
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Gemini APIキーの取得**
   - [Google AI Studio](https://makersuite.google.com/app/apikey)でAPIキーを取得
   - `.env`ファイルの`GEMINI_API_KEY`に設定

## 使い方

```bash
python -m src.main
```

### 操作方法

1. アプリケーションが起動すると待機画面が表示されます
2. 話しかけると自動的に録音が開始されます（🔴 録音中...）
3. 話し終わって1.5秒経つと自動的に処理が開始されます
4. AIの応答が音声で再生されます
5. 終了するには `q` を入力してEnterキーを押します

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