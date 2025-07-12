# 機能一覧

## リアルタイム音声認識（VAD）

### 概要
Voice Activity Detection（VAD）により、話し始めると自動的に録音を開始し、話し終わると自動的に処理を開始します。

### 設定可能なパラメータ
- **energy_threshold** (デフォルト: 0.01): 音声と判定するエネルギー閾値
- **silence_duration** (デフォルト: 1.5秒): 無音と判定するまでの時間
- **pre_buffer** (デフォルト: 0.5秒): 音声検出前のバッファ時間
- **min_duration** (デフォルト: 0.5秒): 最小録音時間
- **max_duration** (デフォルト: 30秒): 最大録音時間

### 動作の流れ
1. 音声レベルをリアルタイムで監視
2. 閾値を超えたら録音開始（プレバッファ含む）
3. 無音が一定時間続いたら録音終了
4. 自動的に次の処理へ

## 音声認識（Whisper）

### 対応モデル
- tiny: 最速・低精度
- base: バランス型（デフォルト）
- small: 高精度
- medium: より高精度
- large: 最高精度

### 言語設定
- 日本語（ja）がデフォルト
- 多言語対応可能

## AI対話（Gemini）

### 使用モデル
- gemini-1.5-flash: 高速レスポンス
- gemini-1.5-pro: 高精度（要設定変更）

### 機能
- コンテキストを保持した自然な会話
- 日本語での流暢な応答
- カスタムシステムプロンプト対応

## 音声合成（gTTS）

### 特徴
- 自然な日本語音声
- 速度調整可能（通常/ゆっくり）
- MP3からWAVへの自動変換

## ログ機能

### 記録される情報
- 処理時間の計測
- エラー情報
- 音声レベル
- APIレスポンス時間

### ログファイル
- `logs/voice_gemini_app.log`: 通常ログ
- JSONフォーマットで構造化

## カスタマイズ

### 環境変数（.env）
```env
# API設定
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=2048

# 音声設定
AUDIO_SAMPLE_RATE=16000
RECORDING_MAX_SECONDS=30

# 音声認識
WHISPER_MODEL=base
WHISPER_LANGUAGE=ja

# 音声合成
TTS_LANGUAGE=ja
TTS_SLOW=False

# ログ設定
LOG_LEVEL=INFO
```

### コード拡張ポイント
- プロセッサーの追加（src/processors/）
- 音声フィルターの実装
- 新しいAIモデルの統合
- カスタムVADアルゴリズム