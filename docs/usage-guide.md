# 使用方法ガイド - Voice Gemini App

このドキュメントは、Voice Gemini Appの使用方法を説明します。

## クイックスタート

### 基本的な使用方法

```bash
# アプリケーションの起動
python src/main.py

# または実装後のエントリーポイント
python -m voice_gemini_app
```

### 初回起動時の確認事項

1. マイクへのアクセス許可（OSレベル）
2. スピーカーの音量設定
3. インターネット接続状態
4. Gemini APIキーの設定（.envファイル）

## 対話モード

### 基本的な対話フロー

1. **起動**: アプリケーションが起動メッセージを表示
2. **待機**: 「録音を開始するにはEnterキーを押してください」と表示
3. **録音**: Enterキーを押すと録音開始（最大30秒）
4. **処理**: 音声をテキストに変換→Gemini AIが応答生成→音声に変換
5. **再生**: AI の応答が音声で再生される
6. **繰り返し**: 次の質問のために待機状態に戻る

### キーボードショートカット

| キー | 動作 |
|------|------|
| Enter | 録音開始/停止 |
| Space | 録音を即座に停止 |
| Q/q | アプリケーション終了 |
| R/r | 最後の応答を再生 |
| H/h | ヘルプ表示 |

## プログラマティックな使用

### 基本的な使用例

```python
import asyncio
from src.voice_gemini import VoiceGeminiApp

async def main():
    # アプリケーションの初期化
    app = VoiceGeminiApp()
    
    # 単発の対話
    response = await app.process_voice_input("path/to/audio.wav")
    print(f"AI応答: {response}")
    
    # 対話ループの開始
    await app.start_conversation_loop()

# 実行
asyncio.run(main())
```

### カスタムパイプラインの構築

```python
from genai_processors import streams
from src.processors import (
    AudioFileProcessor,
    WhisperProcessor,
    GeminiChatProcessor,
    GTTSProcessor
)

async def custom_pipeline():
    # プロセッサーの初期化（カスタム設定）
    audio_proc = AudioFileProcessor(sample_rate=16000)
    stt_proc = WhisperProcessor(model="small", language="ja")
    gemini_proc = GeminiChatProcessor(
        model="gemini-pro",
        temperature=0.9,
        system_prompt="あなたは親切なアシスタントです。"
    )
    tts_proc = GTTSProcessor(language="ja", slow=False)
    
    # パイプラインの構築
    pipeline = audio_proc >> stt_proc >> gemini_proc >> tts_proc
    
    # 実行
    input_stream = streams.stream_content(["recording.wav"])
    async for result in pipeline(input_stream):
        if result.audio_file:
            await result.play()
```

### 会話履歴の管理

```python
from src.conversation import ConversationManager

# 会話マネージャーの初期化
conv_manager = ConversationManager()

# 新しい会話セッションの開始
session_id = conv_manager.start_session()

# メッセージの追加
conv_manager.add_message(session_id, "user", "こんにちは")
conv_manager.add_message(session_id, "assistant", "こんにちは！")

# 会話履歴の取得
history = conv_manager.get_history(session_id)

# 会話の保存
conv_manager.save_session(session_id, "conversations/session_001.json")
```

## 設定のカスタマイズ

### 環境変数による設定

`.env`ファイルで以下の設定が可能：

```env
# 音声録音設定
RECORDING_MAX_SECONDS=60  # 最大録音時間を60秒に
AUDIO_SAMPLE_RATE=48000   # 高品質録音

# Whisper設定
WHISPER_MODEL=small       # より高精度なモデル
WHISPER_DEVICE=cuda       # GPU使用（対応環境のみ）

# Gemini設定
GEMINI_TEMPERATURE=0.5    # より確定的な応答
GEMINI_MAX_TOKENS=4096    # より長い応答を許可

# TTS設定
TTS_SLOW=True            # ゆっくりした音声
```

### プログラムによる設定

```python
from src.config import AppConfig

# 設定の読み込みとカスタマイズ
config = AppConfig()
config.audio.sample_rate = 48000
config.whisper.model = "small"
config.gemini.temperature = 0.5

# アプリケーションに適用
app = VoiceGeminiApp(config=config)
```

## 高度な使用方法

### ストリーミングモード（将来実装）

```python
# リアルタイムストリーミング対話
async def streaming_conversation():
    app = VoiceGeminiApp(streaming=True)
    
    async for chunk in app.stream_conversation():
        # 音声チャンクごとに処理
        print(f"認識テキスト: {chunk.text}")
        if chunk.audio:
            await chunk.play_async()
```

### カスタムプロンプト

```python
# プロンプトテンプレートの使用
from src.prompts import PromptTemplate

template = PromptTemplate.from_file("prompts/technical_assistant.yaml")
app = VoiceGeminiApp(prompt_template=template)

# または直接指定
app.set_system_prompt("""
あなたは技術的な質問に答える専門家です。
回答は簡潔で正確にしてください。
""")
```

### バッチ処理

```python
# 複数の音声ファイルを一括処理
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

async def batch_process():
    app = VoiceGeminiApp()
    results = await app.batch_process(audio_files)
    
    for file, result in zip(audio_files, results):
        print(f"{file}: {result.text}")
        await result.save_audio(f"output_{file}")
```

## コマンドラインオプション（将来実装）

```bash
# 基本的な使用
python -m voice_gemini_app

# デバッグモード
python -m voice_gemini_app --debug

# 特定の設定ファイルを使用
python -m voice_gemini_app --config config/production.env

# 音声デバイスを指定
python -m voice_gemini_app --input-device 0 --output-device 1

# バッチモード
python -m voice_gemini_app --batch audio_files/*.wav --output results/
```

## 実行例とユースケース

### 1. シンプルな質問応答

```
ユーザー: "今日の天気はどうですか？"
AI: "申し訳ございませんが、リアルタイムの天気情報にはアクセスできません。
     天気予報を確認するには、天気予報サイトやアプリをご利用ください。"
```

### 2. 技術的な質問

```
ユーザー: "Pythonで非同期処理を実装する方法を教えて"
AI: "Pythonで非同期処理を実装するには、asyncioモジュールを使用します。
     基本的な実装方法は..."
```

### 3. 創造的なタスク

```
ユーザー: "短い物語を作ってください"
AI: "昔々、小さな村に住む少年がいました..."
```

## パフォーマンスのヒント

1. **Whisperモデルの選択**
   - `tiny`: 最速、精度は低め
   - `base`: バランス型（推奨）
   - `small`: 高精度、処理時間増加

2. **録音品質の向上**
   - 静かな環境で録音
   - マイクを適切な距離に配置
   - ノイズキャンセリング機能の活用

3. **応答速度の最適化**
   - より小さいGeminiモデルの使用
   - キャッシュの活用
   - 非同期処理の最適化

## 次のステップ

問題が発生した場合は、[トラブルシューティングガイド](troubleshooting.md)を参照してください。

開発に参加する場合は、[開発ガイドライン](development-guidelines.md)を確認してください。

## 更新履歴

- 2025-07-12: 初版作成（voice-gemini-app.mdから分離）