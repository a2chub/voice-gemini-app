"""
Voice Gemini App - メインアプリケーション

音声対話型AIアプリケーションのエントリーポイント
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import signal

from genai_processors import streams, chain, ProcessorPart

from .audio.recorder import AudioRecorder
from .audio.vad_recorder import VADRecorder
from .audio.player import AudioPlayer
from .processors.audio_processor import AudioFileProcessor
from .processors.stt_processor import WhisperProcessor
from .processors.gemini_processor import GeminiChatProcessor
from .processors.tts_processor import GTTSProcessor
from .config import get_config
from .utils.logger import get_logger
from .utils.exceptions import VoiceGeminiError, RecordingError
from .utils.async_input import AsyncInputReader


class VoiceGeminiApp:
    """Voice Gemini App のメインクラス"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.config = get_config()
        self.logger = get_logger()
        
        # コンポーネントの初期化
        self.recorder = AudioRecorder()
        self.vad_recorder = VADRecorder()
        self.player = AudioPlayer()
        
        # VADコールバックの設定
        self.vad_recorder.on_voice_start = self._on_voice_start
        self.vad_recorder.on_voice_end = self._on_voice_end
        self.vad_recorder.on_level_update = self._on_level_update
        
        # 非同期入力リーダー
        self.input_reader = AsyncInputReader()
        
        # プロセッサーの初期化
        self.audio_processor = AudioFileProcessor()
        self.stt_processor = WhisperProcessor()
        self.gemini_processor = GeminiChatProcessor()
        self.tts_processor = GTTSProcessor()
        
        # パイプラインの構築
        self.pipeline = chain([
            self.audio_processor,
            self.stt_processor,
            self.gemini_processor,
            self.tts_processor
        ])
        
        # 状態管理
        self.is_running = False
        self.last_audio_path: Optional[Path] = None
    
    async def start(self):
        """アプリケーションを開始"""
        await self.logger.info(
            "Voice Gemini App を開始します",
            operation="app_start",
            version="0.1.0"
        )
        
        # 起動メッセージ
        self._print_welcome_message()
        
        # 入力リーダーを開始
        self.input_reader.start()
        
        # 対話ループの開始
        self.is_running = True
        try:
            await self.conversation_loop()
        except KeyboardInterrupt:
            await self.logger.info(
                "ユーザーによって中断されました",
                operation="app_interrupted"
            )
        except Exception as e:
            await self.logger.error(
                "アプリケーションエラー",
                operation="app_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            await self.cleanup()
    
    async def conversation_loop(self):
        """対話ループ"""
        while self.is_running:
            try:
                # ユーザー入力の待機
                print("\n" + "="*50)
                print("🎤 話し始めると自動的に録音を開始します")
                print("🔇 話し終わって約1.5秒後に自動的に処理を開始します")
                print("終了するには 'q' を入力してください")
                print("="*50)
                
                # quitコマンドのチェックタスク
                async def check_for_quit():
                    while self.is_running:
                        if await self.input_reader.check_for_quit():
                            self.is_running = False
                            self.vad_recorder.stop_recording()
                            break
                        await asyncio.sleep(0.1)
                
                quit_task = asyncio.create_task(check_for_quit())
                
                audio_path = None
                try:
                    print("\n🎯 話しかけてください...")
                    print("📊mber 音声レベル: ", end="", flush=True)
                    
                    # VAD付き録音の実行
                    audio_data, audio_path = await self.vad_recorder.record_with_vad(
                        max_duration=self.config.audio.recording_max_seconds,
                        min_duration=0.5,
                        pre_buffer=0.5
                    )
                    self.last_audio_path = audio_path
                    print(f"\n✅ 録音完了: {audio_path}")
                    
                except RecordingError as e:
                    if "音声が検出されませんでした" in str(e):
                        print("\n⚠️  音声が検出されませんでした")
                    audio_path = None
                finally:
                    # quitタスクのクリーンアップ
                    quit_task.cancel()
                    try:
                        await quit_task
                    except asyncio.CancelledError:
                        pass
                
                # 録音が成功した場合のみパイプライン処理
                if audio_path is not None:
                    print("\n🔄 処理中...")
                    await self.process_audio(audio_path)
                
            except KeyboardInterrupt:
                # 録音中の中断をキャッチ
                print("\n⚠️  録音を中断しました")
                self.is_running = False
                break
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
                await self.logger.error(
                    "対話ループでエラーが発生しました",
                    operation="conversation_loop_error",
                    error=str(e),
                    error_type=type(e).__name__
                )
                # エラー後も続行
                await asyncio.sleep(1)
    
    async def process_audio(self, audio_path: Path):
        """音声ファイルを処理"""
        try:
            # 入力ストリームの作成（ProcessorPartでラップ）
            async def create_input_stream():
                async for path in streams.stream_content([str(audio_path)]):
                    yield ProcessorPart(path)
            
            input_stream = create_input_stream()
            
            # パイプラインの実行
            result_parts = []
            async for part in self.pipeline(input_stream):
                result_parts.append(part)
            
            # 最後の結果（TTS出力）を取得
            if result_parts:
                final_part = result_parts[-1]
                
                # 音声パスの取得
                if final_part.metadata and 'audio_path' in final_part.metadata:
                    tts_audio_path = Path(final_part.metadata['audio_path'])
                    
                    # 認識されたテキストの表示
                    if 'text' in final_part.metadata:
                        print(f"\n💬 あなた: {final_part.metadata['text']}")
                    
                    # Geminiの応答テキストの表示（パイプライン中の情報から取得）
                    for part in result_parts:
                        if part.text and part.metadata and part.metadata.get('processor') == 'gemini':
                            print(f"\n🤖 Gemini: {part.text}")
                            break
                    
                    # 音声の再生
                    print("\n🔊 応答を再生中...")
                    await self.player.play_file(tts_audio_path)
                    print("✅ 再生完了")
                    
                    # 一時ファイルの削除
                    tts_audio_path.unlink(missing_ok=True)
            
        except Exception as e:
            await self.logger.error(
                "音声処理中にエラーが発生しました",
                operation="audio_processing_error",
                error=str(e),
                error_type=type(e).__name__,
                audio_path=str(audio_path)
            )
            raise VoiceGeminiError(f"音声処理に失敗しました: {e}")
    
    async def cleanup(self):
        """クリーンアップ処理"""
        await self.logger.info(
            "クリーンアップを実行しています",
            operation="app_cleanup"
        )
        
        # 入力リーダーを停止
        self.input_reader.stop()
        
        # 音声プレイヤーを停止
        try:
            await self.player.stop()
        except:
            pass
        
        # 最後の音声ファイルの削除
        if self.last_audio_path and self.last_audio_path.exists():
            self.last_audio_path.unlink(missing_ok=True)
        
        print("\n👋 Voice Gemini App を終了しました")
    
    def _print_welcome_message(self):
        """ウェルカムメッセージを表示"""
        print("\n" + "="*60)
        print("🎙️  Voice Gemini App へようこそ！")
        print("="*60)
        print("\n音声で質問すると、Gemini AI が音声で応答します。")
        print("\n🆕 リアルタイム音声認識モード:")
        print("  - 話し始めると自動的に録音開始")
        print("  - 話し終わると自動的に処理開始")
        print("  - 'q' または 'quit': アプリケーション終了")
        print("\n設定:")
        print(f"  - Whisperモデル: {self.config.whisper.model}")
        print(f"  - Geminiモデル: {self.config.gemini.model}")
        print(f"  - 音声言語: {self.config.tts.language}")
        print("="*60)
    
    def _on_voice_start(self):
        """音声検出開始時のコールバック"""
        print("\n🔴 録音中...", end="", flush=True)
    
    def _on_voice_end(self):
        """音声検出終了時のコールバック"""
        print(" 完了!")
    
    def _on_level_update(self, level: float):
        """音声レベル更新時のコールバック"""
        # レベルメーターの表示
        bar_length = 30
        filled_length = int(bar_length * min(level * 10, 1.0))
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r📊AFter 音声レベル: [{bar}] {level:.3f}", end="", flush=True)


async def main():
    """メイン関数"""
    try:
        # アプリケーションの作成と開始
        app = VoiceGeminiApp()
        await app.start()
    except Exception as e:
        print(f"\n❌ 致命的なエラーが発生しました: {e}")
        sys.exit(1)


def run():
    """エントリーポイント"""
    # イベントループの実行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  終了シグナルを受信しました...")
        sys.exit(0)


if __name__ == "__main__":
    run()