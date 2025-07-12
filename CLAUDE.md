# CLAUDE.md

このファイルは、このリポジトリでコードを扱う際のClaude Code（claude.ai/code）へのガイダンスを提供します。

## プロジェクト概要

Voice Gemini App - Google Gemini AIとのリアルタイム音声対話アプリケーションです。音声アクティビティ検出（VAD）により、話し始めると自動的に録音を開始し、話し終わると自動的にAIが応答を生成して音声で返答します。

## 主要技術

- **genai-processors**: AI操作をチェーンするためのコア非同期パイプライン処理フレームワーク
- **vibe-logger**: AI最適化された構造化ログライブラリ
- **sounddevice**: クロスプラットフォーム音声録音
- **Whisper/Google Cloud Speech**: 音声テキスト変換処理
- **Gemini API**: AI会話処理
- **gTTS**: テキスト音声合成

## プロジェクト構造

想定されるプロジェクト構造は以下のパターンに従います：

```
voice-gemini-app/
├── docs/                    # ドキュメント
├── src/
│   ├── audio/              # 音声録音と合成
│   ├── processors/         # genai-processors実装
│   └── utils/              # ロガー設定と構成
├── logs/                   # ログ出力ディレクトリ
└── tests/                  # テストファイル
```

## 開発コマンド

```bash
# 環境セットアップ
./scripts/setup.sh

# アプリケーション実行
python -m src.main  # 適切なインポートのために -m フラグを使用

# テスト実行
pytest tests/

# カバレッジ付きテスト実行
./scripts/run_tests.sh --coverage

# リント実行
./scripts/run_tests.sh --lint
```

## アーキテクチャ概要

### リアルタイム音声処理フロー

1. **VAD録音フェーズ**（非同期処理）
   - 音声アクティビティ検出により自動録音開始
   - リアルタイム音声レベル表示
   - 無音検出による自動録音停止
   - プレバッファ機能（話し始める前の音声も保存）

2. **非同期処理パイプライン**（genai-processors）
   - AudioProcessor → STTProcessor → GeminiProcessor → TTSProcessor
   - 各プロセッサは個別の非同期ユニットとしてチェーン可能
   - ProcessorPartオブジェクトで通信（text属性とmetadata属性）

3. **ロギング層**
   - 標準Pythonロギングとvibe-logger互換
   - 非同期環境でのロギング対応
   - メトリクス追跡：処理時間、音声レベル、APIレスポンス

## 重要な実装詳細

### Pydantic v2 互換性
プロジェクトは設定管理に`pydantic-settings`を使用したPydantic v2を使用：
- BaseSettingsは`pydantic`ではなく`pydantic_settings`からインポート
- `@classmethod`デコレータ付きの`field_validator`を使用
- 内部`Config`クラスの代わりに`model_config = SettingsConfigDict()`
- **重要**: 検証エラーを防ぐため各設定クラスで`extra="ignore"`を設定

### genai-processors統合
- `ProcessorPart`を直接使用（第一引数にtext、metadataはキーワード引数）
- ProcessorPartは`text`属性を持つ（`content`ではない）
- ベースプロセッサで`call()`と`__call__()`の両メソッドを実装
- パイプライン構成には`chain()`関数を使用
- `genai_processors`からインポート：`streams`、`chain`、`ProcessorPart`

### ロギングセットアップ
- vibe-loggerが利用できない場合は標準Pythonロギングを使用
- 非同期操作のためAsyncLoggerWrapperでラップ
- `get_running_loop()`を使用してイベントループを取得

### 音声処理
- **VADRecorder**: リアルタイム音声アクティビティ検出
- **音声再生**: sounddeviceを使用（simpleaudioのSegfault回避）
- すべての音声データはfloat32 [-1, 1]範囲に正規化
- デフォルトサンプルレート：16kHz（Whisperの要件）
- gTTSはMP3を出力、一貫性のためWAVに変換

### 非同期処理
- **AsyncInputReader**: バックグラウンドスレッドで入力を読み取り
- selectモジュールでノンブロッキング入力（Unix/Linux/macOS）
- 適切なタスクキャンセルとクリーンアップ

## よくある問題と解決策

### モジュールインポートエラー
アプリケーションは常にモジュールとして実行：
```bash
# 正しい
python -m src.main

# 間違い（相対インポートエラーの原因）
python src/main.py
```

### Geminiモデルの変更
gemini-proは廃止されました。以下のモデルを使用：
```env
GEMINI_MODEL=gemini-1.5-flash  # 高速レスポンス
# または
GEMINI_MODEL=gemini-1.5-pro    # 高精度
```

### 環境変数
必要なキーを含む`.env`ファイルが存在することを確認：
- `GEMINI_API_KEY`（必須）
- その他の設定はconfig.pyにデフォルト値あり

## テストアプローチ

- 非同期サポート付きpytestを使用（`pytest-asyncio`）
- 外部依存関係（API、音声デバイス）をモック
- `conftest.py`のテストフィクスチャがサンプルデータを提供
- `-m`フラグで実行：`python -m pytest tests/`

## 現在のステータス

- **リアルタイム音声認識**: 実装完了（VAD機能）
- **非同期処理**: 安定動作
- **エラーハンドリング**: Ctrl+C対応済み
- **ドキュメント**: 最新化完了（2025-07-13）

## 主な変更履歴

- 2025-07-13: リアルタイム音声認識（VAD）実装
- 2025-07-13: ProcessorPartのAPI変更対応
- 2025-07-13: 音声再生をsounddeviceに変更
- 2025-07-12: 初期実装完了