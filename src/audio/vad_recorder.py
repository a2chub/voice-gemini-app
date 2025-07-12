"""
音声アクティビティ検出（VAD）付き録音モジュール

リアルタイムで音声を検出し、話し始めたら録音を開始し、
話し終わったら自動的に録音を停止する機能を提供します。
"""

import asyncio
import time
import queue
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
import threading

from ..config import get_config
from ..utils.logger import get_logger
from ..utils.exceptions import RecordingError, AudioError


class VADRecorder:
    """音声アクティビティ検出付き録音クラス"""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: int = 1,
        device: Optional[int] = None,
        chunk_duration: float = 0.03,  # 30ms chunks
        energy_threshold: float = 0.01,
        silence_duration: float = 1.5
    ):
        """
        VAD付きレコーダーの初期化
        
        Args:
            sample_rate: サンプリングレート
            channels: チャンネル数
            device: 音声デバイスID
            chunk_duration: チャンクの長さ（秒）
            energy_threshold: 音声と判定するエネルギー閾値
            silence_duration: 無音と判定するまでの時間（秒）
        """
        config = get_config()
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.channels = channels
        self.device = device
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * chunk_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.logger = get_logger()
        
        # 録音状態
        self.is_recording = False
        self.is_voice_active = False
        self.last_voice_time = 0
        self.audio_queue = queue.Queue()
        self.recorded_chunks = []
        
        # コールバック
        self.on_voice_start: Optional[Callable] = None
        self.on_voice_end: Optional[Callable] = None
        self.on_level_update: Optional[Callable[[float], None]] = None
    
    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """音声チャンクのエネルギーを計算"""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音声入力のコールバック"""
        if status:
            print(f"Audio callback status: {status}")
        
        # 音声データをキューに追加
        self.audio_queue.put(indata.copy())
    
    async def record_with_vad(
        self,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        pre_buffer: float = 0.5,
        output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Path]:
        """
        VAD付きで録音を実行
        
        Args:
            max_duration: 最大録音時間
            min_duration: 最小録音時間
            pre_buffer: 音声検出前のバッファ時間
            output_path: 出力パスz
        
        Returns:
            録音データと保存パス
        """
        await self.logger.info(
            "VAD付き録音を開始します",
            operation="vad_recording_start",
            max_duration=max_duration,
            energy_threshold=self.energy_threshold
        )
        
        self.is_recording = True
        self.recorded_chunks = []
        self.is_voice_active = False
        
        # プレバッファ用のリングバッファ
        pre_buffer_size = int(pre_buffer / self.chunk_duration)
        pre_buffer_chunks = []
        
        # 音声ストリームを開始
        stream = sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device
        )
        
        try:
            with stream:
                start_time = time.time()
                voice_start_time = None
                
                while self.is_recording and (time.time() - start_time) < max_duration:
                    try:
                        # キューから音声データを取得（タイムアウト付き）
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        
                        # エネルギー計算
                        energy = self._calculate_energy(audio_chunk)
                        
                        # レベル更新コールバック
                        if self.on_level_update:
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.on_level_update, energy
                            )
                        
                        # 音声アクティビティ判定
                        if energy > self.energy_threshold:
                            if not self.is_voice_active:
                                # 音声開始
                                self.is_voice_active = True
                                voice_start_time = time.time()
                                
                                # プレバッファを録音データに追加
                                self.recorded_chunks.extend(pre_buffer_chunks)
                                
                                await self.logger.info(
                                    "音声を検出しました",
                                    operation="voice_detected",
                                    energy=float(energy)
                                )
                                
                                if self.on_voice_start:
                                    await asyncio.get_event_loop().run_in_executor(
                                        None, self.on_voice_start
                                    )
                            
                            self.last_voice_time = time.time()
                        
                        # 録音データの処理
                        if self.is_voice_active:
                            self.recorded_chunks.append(audio_chunk)
                            
                            # 無音判定
                            if time.time() - self.last_voice_time > self.silence_duration:
                                voice_duration = time.time() - voice_start_time
                                
                                if voice_duration >= min_duration:
                                    # 録音終了
                                    await self.logger.info(
                                        "音声が終了しました",
                                        operation="voice_ended",
                                        duration=voice_duration
                                    )
                                    
                                    if self.on_voice_end:
                                        await asyncio.get_event_loop().run_in_executor(
                                            None, self.on_voice_end
                                        )
                                    
                                    break
                                else:
                                    # 短すぎる音声は無視
                                    self.is_voice_active = False
                                    self.recorded_chunks = []
                        else:
                            # プレバッファを更新
                            pre_buffer_chunks.append(audio_chunk)
                            if len(pre_buffer_chunks) > pre_buffer_size:
                                pre_buffer_chunks.pop(0)
                    
                    except queue.Empty:
                        continue
                    except Exception as e:
                        await self.logger.error(
                            "録音処理中にエラーが発生しました",
                            operation="vad_recording_error",
                            error=str(e)
                        )
                        raise
        
        finally:
            self.is_recording = False
        
        # 録音データを結合
        if self.recorded_chunks:
            recording = np.concatenate(self.recorded_chunks)
            
            # モノラルの場合は1次元配列に変換
            if self.channels == 1:
                recording = recording.flatten()
            
            # ファイルパスの生成
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"recording_{timestamp}.wav")
            
            # 保存
            await self._save_wav(recording, output_path)
            
            await self.logger.info(
                "VAD録音が完了しました",
                operation="vad_recording_complete",
                file_path=str(output_path),
                duration=len(recording) / self.sample_rate,
                file_size_bytes=output_path.stat().st_size
            )
            
            return recording, output_path
        else:
            raise RecordingError("音声が検出されませんでした")
    
    async def _save_wav(self, data: np.ndarray, path: Path):
        """WAVファイルとして保存"""
        loop = asyncio.get_event_loop()
        
        # int16形式に変換
        data_int16 = np.int16(data * 32767)
        
        # ファイル保存
        await loop.run_in_executor(
            None,
            lambda: wav.write(str(path), self.sample_rate, data_int16)
        )
    
    def stop_recording(self):
        """録音を停止"""
        self.is_recording = False
    
    async def record_with_push_to_talk(
        self,
        output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Path]:
        """
        プッシュトゥトーク方式の録音
        Enterキーを押している間だけ録音
        """
        await self.logger.info(
            "プッシュトゥトーク録音を開始します",
            operation="push_to_talk_start"
        )
        
        # TODO: キーボード入力との連携実装
        # 現在はVAD録音で代用
        return await self.record_with_vad()