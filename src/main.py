"""
Voice Gemini App - メインアプリケーション

音声対話型AIアプリケーションのエントリーポイント
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import signal

from genai_processors import streams, chain

from .audio.recorder import AudioRecorder
from .audio.player import AudioPlayer
from .processors.audio_processor import AudioFileProcessor
from .processors.stt_processor import WhisperProcessor
from .processors.gemini_processor import GeminiChatProcessor
from .processors.tts_processor import GTTSProcessor
from .config import get_config
from .utils.logger import get_logger
from .utils.exceptions import VoiceGeminiError


class VoiceGeminiApp:
    """Voice Gemini App のメインクラス"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.config = get_config()
        self.logger = get_logger()
        
        # コンポーネントの初期化
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        
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
            operation="app_start",
            message="Voice Gemini App を開始します",
            version="0.1.0"
        )
        
        # 起動メッセージ
        self._print_welcome_message()
        
        # 対話ループの開始
        self.is_running = True
        try:
            await self.conversation_loop()
        except KeyboardInterrupt:
            await self.logger.info(
                operation="app_interrupted",
                message="ユーザーによって中断されました"
            )
        except Exception as e:
            await self.logger.error(
                operation="app_error",
                message="アプリケーションエラー",
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
                print("録音を開始するには Enter キーを押してください")
                print("終了するには 'q' を入力してください")
                print("="*50)
                
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    self.is_running = False
                    break
                
                # 録音の実行
                print("\n🎤 録音中... (最大30秒、Ctrl+Cで停止)")
                audio_data, audio_path = await self.recorder.record(
                    duration=self.config.audio.recording_max_seconds
                )
                self.last_audio_path = audio_path
                print(f"✅ 録音完了: {audio_path}")
                
                # パイプライン処理
                print("\n🔄 処理中...")
                await self.process_audio(audio_path)
                
            except KeyboardInterrupt:
                # 録音中の中断をキャッチ
                print("\n⚠️  録音を中断しました")
                continue
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
                await self.logger.error(
                    operation="conversation_loop_error",
                    message="対話ループでエラーが発生しました",
                    error=str(e),
                    error_type=type(e).__name__
                )
    
    async def process_audio(self, audio_path: Path):
        """音声ファイルを処理"""
        try:
            # 入力ストリームの作成
            input_stream = streams.stream_content([str(audio_path)])
            
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
                        if part.content and part.metadata and part.metadata.get('processor') == 'gemini':
                            print(f"\n🤖 Gemini: {part.content}")
                            break
                    
                    # 音声の再生
                    print("\n🔊 応答を再生中...")
                    await self.player.play_file(tts_audio_path)
                    print("✅ 再生完了")
                    
                    # 一時ファイルの削除
                    tts_audio_path.unlink(missing_ok=True)
            
        except Exception as e:
            await self.logger.error(
                operation="audio_processing_error",
                message="音声処理中にエラーが発生しました",
                error=str(e),
                error_type=type(e).__name__,
                audio_path=str(audio_path)
            )
            raise VoiceGeminiError(f"音声処理に失敗しました: {e}")
    
    async def cleanup(self):
        """クリーンアップ処理"""
        await self.logger.info(
            operation="app_cleanup",
            message="クリーンアップを実行しています"
        )
        
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
        print("\n使用方法:")
        print("  - Enter キー: 録音開始")
        print("  - Ctrl+C: 録音停止")
        print("  - 'q' または 'quit': アプリケーション終了")
        print("\n設定:")
        print(f"  - Whisperモデル: {self.config.whisper.model}")
        print(f"  - Geminiモデル: {self.config.gemini.model}")
        print(f"  - 音声言語: {self.config.tts.language}")
        print("="*60)


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
    # Ctrl+C のハンドリング
    def signal_handler(signum, frame):
        print("\n\n⚠️  終了シグナルを受信しました...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # イベントループの実行
    asyncio.run(main())


if __name__ == "__main__":
    run()