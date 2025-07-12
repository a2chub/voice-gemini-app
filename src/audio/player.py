"""
音声再生モジュール

録音した音声やTTSで生成した音声を再生します。
"""

import asyncio
from pathlib import Path
from typing import Union, Optional
import numpy as np
import sounddevice as sd
import simpleaudio as sa
import scipy.io.wavfile as wav

from ..config import get_config
from ..utils.logger import get_logger, log_execution
from ..utils.exceptions import PlaybackError, AudioError


class AudioPlayer:
    """音声再生クラス"""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        device: Optional[int] = None
    ):
        """
        音声プレイヤーの初期化
        
        Args:
            sample_rate: サンプリングレート（Hz）。Noneの場合は設定から取得
            device: 使用する音声デバイスのID。Noneの場合はデフォルトデバイス
        """
        config = get_config()
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.device = device
        self.logger = get_logger()
    
    @log_execution
    async def play_file(self, file_path: Union[str, Path], wait: bool = True):
        """
        音声ファイルを再生
        
        Args:
            file_path: 再生する音声ファイルのパス
            wait: 再生完了まで待機するか
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PlaybackError(f"音声ファイルが見つかりません: {file_path}")
        
        await self.logger.info(
            f"音声ファイルの再生を開始します",
            operation="playback_start",
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size
        )
        
        try:
            # sounddeviceを使用した再生（simpleaudioのSegmentation faultを回避）
            loop = asyncio.get_event_loop()
            
            # WAVファイルを読み込み
            sample_rate, audio_data = await loop.run_in_executor(
                None,
                lambda: wav.read(str(file_path))
            )
            
            # データタイプを正規化
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # 再生
            await loop.run_in_executor(
                None,
                lambda: sd.play(audio_data, sample_rate, device=self.device)
            )
            
            if wait:
                # 再生完了まで待機
                await loop.run_in_executor(None, sd.wait)
            
            await self.logger.info(
                "音声ファイルの再生が完了しました",
                operation="playback_complete",
                file_path=str(file_path)
            )
            
        except Exception as e:
            await self.logger.error(
                "音声再生中にエラーが発生しました",
                operation="playback_error",
                error=str(e),
                error_type=type(e).__name__,
                file_path=str(file_path)
            )
            raise PlaybackError(f"音声再生に失敗しました: {e}")
    
    @log_execution
    async def play_array(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        wait: bool = True
    ):
        """
        numpy配列の音声データを再生
        
        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート。Noneの場合はインスタンスの値を使用
            wait: 再生完了まで待機するか
        """
        sample_rate = sample_rate or self.sample_rate
        
        await self.logger.info(
            "音声データの再生を開始します",
            operation="playback_array_start",
            data_shape=audio_data.shape,
            sample_rate=sample_rate,
            duration_seconds=len(audio_data) / sample_rate
        )
        
        try:
            # sounddeviceを使用した再生
            loop = asyncio.get_event_loop()
            
            # 再生開始（非ブロッキング）
            await loop.run_in_executor(
                None,
                lambda: sd.play(audio_data, sample_rate, device=self.device)
            )
            
            if wait:
                # 再生完了まで待機
                await loop.run_in_executor(None, sd.wait)
            
            await self.logger.info(
                "音声データの再生が完了しました",
                operation="playback_array_complete"
            )
            
        except Exception as e:
            await self.logger.error(
                "音声データ再生中にエラーが発生しました",
                operation="playback_array_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise PlaybackError(f"音声データ再生に失敗しました: {e}")
    
    async def _wait_playback_async(self, play_obj):
        """simpleaudioの再生完了を非同期で待つ"""
        loop = asyncio.get_event_loop()
        while play_obj.is_playing():
            await asyncio.sleep(0.1)
    
    @log_execution
    async def play_with_volume(
        self,
        file_path: Union[str, Path],
        volume: float = 1.0,
        wait: bool = True
    ):
        """
        音量調整付きで音声ファイルを再生
        
        Args:
            file_path: 再生する音声ファイルのパス
            volume: 音量（0.0〜1.0）
            wait: 再生完了まで待機するか
        """
        if not 0.0 <= volume <= 1.0:
            raise ValueError(f"音量は0.0〜1.0の範囲で指定してください: {volume}")
        
        file_path = Path(file_path)
        
        # WAVファイルを読み込み
        loop = asyncio.get_event_loop()
        sample_rate, audio_data = await loop.run_in_executor(
            None,
            lambda: wav.read(str(file_path))
        )
        
        # 音量調整
        audio_data = audio_data.astype(np.float32) / 32768.0  # 正規化
        audio_data = audio_data * volume
        
        # 再生
        await self.play_array(audio_data, sample_rate, wait)
    
    @staticmethod
    async def list_output_devices():
        """利用可能な出力デバイスをリスト表示"""
        logger = get_logger()
        devices = sd.query_devices()
        
        output_devices = []
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'default': i == sd.default.device[1]
                })
        
        await logger.info(
            "利用可能な出力デバイス",
            operation="list_output_devices",
            devices=output_devices
        )
        
        return output_devices
    
    async def stop(self):
        """再生を停止"""
        try:
            sd.stop()
            await self.logger.info(
                "音声再生を停止しました",
                operation="playback_stop"
            )
        except Exception as e:
            await self.logger.error(
                "再生停止中にエラーが発生しました",
                operation="playback_stop_error",
                error=str(e)
            )
            raise PlaybackError(f"再生停止に失敗しました: {e}")