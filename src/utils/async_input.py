"""
非同期入力ユーティリティ

非同期環境でのキーボード入力を扱うヘルパー関数
"""

import asyncio
import sys
import threading
from typing import Optional


class AsyncInputReader:
    """非同期入力リーダー"""
    
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.reader_thread = None
        self.is_running = False
    
    def start(self):
        """入力読み取りスレッドを開始"""
        if self.reader_thread is None or not self.reader_thread.is_alive():
            self.is_running = True
            self.reader_thread = threading.Thread(target=self._read_input, daemon=True)
            self.reader_thread.start()
    
    def stop(self):
        """入力読み取りを停止"""
        self.is_running = False
    
    def _read_input(self):
        """バックグラウンドスレッドで入力を読み取る"""
        while self.is_running:
            try:
                # 標準入力から読み取り（タイムアウト付き）
                import select
                if sys.platform == 'win32':
                    # Windows用の処理
                    line = sys.stdin.readline().strip()
                else:
                    # Unix/Linux/macOS用の処理
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        line = sys.stdin.readline().strip()
                    else:
                        continue
                
                if line and self.is_running:
                    # 非同期キューに追加
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(
                            self.input_queue.put(line),
                            loop
                        )
                    except RuntimeError:
                        # イベントループが閉じられている場合
                        break
            except Exception:
                if not self.is_running:
                    break
    
    async def get_input(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        非同期で入力を取得
        
        Args:
            timeout: タイムアウト秒数
        
        Returns:
            入力文字列またはNone（タイムアウト時）
        """
        try:
            if timeout:
                return await asyncio.wait_for(self.input_queue.get(), timeout)
            else:
                return await self.input_queue.get()
        except asyncio.TimeoutError:
            return None
    
    async def check_for_quit(self) -> bool:
        """
        quitコマンドをチェック
        
        Returns:
            quitコマンドが入力された場合True
        """
        try:
            # ノンブロッキングで入力をチェック
            line = await self.get_input(timeout=0.1)
            if line and line.lower() in ['q', 'quit', 'exit']:
                return True
        except:
            pass
        return False