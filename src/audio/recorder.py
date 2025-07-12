"""
音声録音モジュール

sounddeviceを使用して音声を録音し、WAVファイルとして保存します。
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime

from ..config import get_config
from ..utils.logger import get_logger, log_execution
from ..utils.exceptions import RecordingError, AudioError


class AudioRecorder:
    """音声録音クラス"""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: int = 1,
        device: Optional[int] = None
    ):
        """
        音声レコーダーの初期化
        
        Args:
            sample_rate: サンプリングレート（Hz）。Noneの場合は設定から取得
            channels: チャンネル数（1: モノラル、2: ステレオ）
            device: 使用する音声デバイスのID。Noneの場合はデフォルトデバイス
        """
        config = get_config()
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.channels = channels
        self.device = device
        self.logger = get_logger()
        
        # デバイスの確認
        self._verify_device()
    
    def _verify_device(self):
        """音声デバイスの確認"""
        try:
            devices = sd.query_devices()
            if self.device is not None:
                device_info = sd.query_devices(self.device)
                if device_info['max_input_channels'] < self.channels:
                    raise AudioError(
                        f"デバイス {self.device} は {self.channels} チャンネルの入力をサポートしていません"
                    )
        except Exception as e:
            raise AudioError(f"音声デバイスの確認に失敗しました: {e}")
    
    @log_execution
    async def record(
        self,
        duration: float,
        output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Path]:
        """
        音声を録音する
        
        Args:
            duration: 録音時間（秒）
            output_path: 出力ファイルパス。Noneの場合は自動生成
        
        Returns:
            録音データ（numpy配列）と保存先パスのタプル
        """
        config = get_config()
        max_duration = config.audio.recording_max_seconds
        
        if duration > max_duration:
            await self.logger.warning(
                f"録音時間が最大値を超えています。{max_duration}秒に制限します。",
                operation="recording_duration_limit",
                requested_duration=duration,
                max_duration=max_duration
            )
            duration = max_duration
        
        await self.logger.info(
            f"{duration}秒間の録音を開始します",
            operation="recording_start",
            duration=duration,
            sample_rate=self.sample_rate,
            channels=self.channels
        )
        
        try:
            # 録音実行
            start_time = time.time()
            recording = await self._record_async(duration)
            recording_time = time.time() - start_time
            
            # ファイルパスの生成
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"recording_{timestamp}.wav")
            
            # WAVファイルとして保存
            await self._save_wav(recording, output_path)
            
            # メトリクスのログ
            await self.logger.info(
                "録音が完了しました",
                operation="recording_complete",
                duration=duration,
                actual_duration=recording_time,
                file_path=str(output_path),
                file_size_bytes=output_path.stat().st_size,
                sample_rate=self.sample_rate,
                channels=self.channels,
                max_amplitude=float(np.max(np.abs(recording))),
                mean_amplitude=float(np.mean(np.abs(recording)))
            )
            
            return recording, output_path
            
        except Exception as e:
            await self.logger.error(
                operation="recording_error",
                message="録音中にエラーが発生しました",
                error=str(e),
                error_type=type(e).__name__
            )
            raise RecordingError(f"録音に失敗しました: {e}")
    
    async def _record_async(self, duration: float) -> np.ndarray:
        """非同期で録音を実行"""
        loop = asyncio.get_event_loop()
        
        # 録音バッファの準備
        num_samples = int(duration * self.sample_rate)
        recording = np.zeros((num_samples, self.channels), dtype=np.float32)
        
        # 録音の実行（ブロッキング操作を非同期で実行）
        await loop.run_in_executor(
            None,
            lambda: sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                out=recording
            )
        )
        
        # 録音完了を待つ
        await loop.run_in_executor(None, sd.wait)
        
        # モノラルの場合は1次元配列に変換
        if self.channels == 1:
            recording = recording.flatten()
        
        return recording
    
    async def _save_wav(self, data: np.ndarray, path: Path):
        """WAVファイルとして保存"""
        loop = asyncio.get_event_loop()
        
        # int16形式に変換（WAVファイルの標準形式）
        data_int16 = np.int16(data * 32767)
        
        # ファイル保存（ブロッキング操作を非同期で実行）
        await loop.run_in_executor(
            None,
            lambda: wav.write(str(path), self.sample_rate, data_int16)
        )
    
    async def record_with_silence_detection(
        self,
        max_duration: float = 30,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0
    ) -> Tuple[np.ndarray, Path]:
        """
        無音検出付きで録音（音声が途切れたら自動停止）
        
        Args:
            max_duration: 最大録音時間（秒）
            silence_threshold: 無音と判定する閾値
            silence_duration: 無音が続いたら停止する時間（秒）
        
        Returns:
            録音データと保存先パスのタプル
        """
        await self.logger.info(
            operation="recording_with_silence_detection_start",
            message="無音検出付き録音を開始します",
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration
        )
        
        # TODO: 無音検出ロジックの実装
        # 現在は通常の録音として動作
        return await self.record(max_duration)
    
    @staticmethod
    async def list_devices():
        """利用可能な音声デバイスをリスト表示"""
        logger = get_logger()
        devices = sd.query_devices()
        
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default': i == sd.default.device[0]
                })
        
        await logger.info(
            operation="list_audio_devices",
            message="利用可能な入力デバイス",
            devices=input_devices
        )
        
        return input_devices