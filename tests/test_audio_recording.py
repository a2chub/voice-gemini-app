"""
音声録音機能のテスト
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.audio.recorder import AudioRecorder
from src.audio.player import AudioPlayer
from src.utils.exceptions import RecordingError, PlaybackError


class TestAudioRecorder:
    """AudioRecorderのテスト"""
    
    @pytest.mark.asyncio
    async def test_recorder_initialization(self):
        """レコーダーの初期化テスト"""
        recorder = AudioRecorder(sample_rate=16000, channels=1)
        
        assert recorder.sample_rate == 16000
        assert recorder.channels == 1
        assert recorder.device is None
    
    @pytest.mark.asyncio
    async def test_record_success(self, temp_dir, mock_logger):
        """録音成功のテスト"""
        recorder = AudioRecorder()
        recorder.logger = mock_logger
        
        # sounddeviceのモック
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait') as mock_wait:
            
            # モック音声データ
            mock_audio = np.random.rand(16000).astype(np.float32)
            mock_rec.return_value = mock_audio.reshape(-1, 1)
            
            # 録音実行
            audio_data, audio_path = await recorder.record(
                duration=1.0,
                output_path=temp_dir / "test.wav"
            )
            
            # 検証
            assert isinstance(audio_data, np.ndarray)
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"
            
            # ログの確認
            info_logs = [log for log in mock_logger.logs if log['level'] == 'info']
            assert any('recording_start' in log.get('operation', '') for log in info_logs)
            assert any('recording_complete' in log.get('operation', '') for log in info_logs)
    
    @pytest.mark.asyncio
    async def test_record_with_max_duration(self, mock_logger):
        """最大録音時間の制限テスト"""
        recorder = AudioRecorder()
        recorder.logger = mock_logger
        
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait') as mock_wait, \
             patch('src.audio.recorder.get_config') as mock_config:
            
            # 設定のモック
            mock_config.return_value.audio.recording_max_seconds = 5
            
            # 10秒の録音を要求（最大5秒に制限される）
            await recorder.record(duration=10.0)
            
            # 警告ログの確認
            warning_logs = [log for log in mock_logger.logs if log['level'] == 'warning']
            assert any('recording_duration_limit' in log.get('operation', '') for log in warning_logs)
    
    @pytest.mark.asyncio
    async def test_record_error_handling(self, mock_logger):
        """録音エラーハンドリングのテスト"""
        recorder = AudioRecorder()
        recorder.logger = mock_logger
        
        with patch('sounddevice.rec', side_effect=Exception("Device error")):
            with pytest.raises(RecordingError) as exc_info:
                await recorder.record(duration=1.0)
            
            assert "録音に失敗しました" in str(exc_info.value)
            
            # エラーログの確認
            error_logs = [log for log in mock_logger.logs if log['level'] == 'error']
            assert any('recording_error' in log.get('operation', '') for log in error_logs)
    
    @pytest.mark.asyncio
    async def test_list_devices(self, mock_logger):
        """デバイスリスト取得のテスト"""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 2, 'max_output_channels': 0},
            {'name': 'USB Microphone', 'max_input_channels': 1, 'max_output_channels': 0},
            {'name': 'Built-in Output', 'max_input_channels': 0, 'max_output_channels': 2}
        ]
        
        with patch('sounddevice.query_devices', return_value=mock_devices), \
             patch('sounddevice.default.device', (0, 2)):
            
            devices = await AudioRecorder.list_devices()
            
            # 入力デバイスのみが返される
            assert len(devices) == 2
            assert all(d['channels'] > 0 for d in devices)


class TestAudioPlayer:
    """AudioPlayerのテスト"""
    
    @pytest.mark.asyncio
    async def test_player_initialization(self):
        """プレイヤーの初期化テスト"""
        player = AudioPlayer(sample_rate=16000)
        
        assert player.sample_rate == 16000
        assert player.device is None
    
    @pytest.mark.asyncio
    async def test_play_file_success(self, sample_audio_file, mock_logger):
        """ファイル再生成功のテスト"""
        player = AudioPlayer()
        player.logger = mock_logger
        
        # simpleaudioのモック
        mock_play_obj = Mock()
        mock_play_obj.is_playing.side_effect = [True, True, False]  # 3回目で再生終了
        
        mock_wave_obj = Mock()
        mock_wave_obj.play.return_value = mock_play_obj
        
        with patch('simpleaudio.WaveObject.from_wave_file', return_value=mock_wave_obj):
            await player.play_file(sample_audio_file)
            
            # 再生メソッドが呼ばれたことを確認
            mock_wave_obj.play.assert_called_once()
            
            # ログの確認
            info_logs = [log for log in mock_logger.logs if log['level'] == 'info']
            assert any('playback_start' in log.get('operation', '') for log in info_logs)
            assert any('playback_complete' in log.get('operation', '') for log in info_logs)
    
    @pytest.mark.asyncio
    async def test_play_file_not_found(self, mock_logger):
        """存在しないファイルの再生エラーテスト"""
        player = AudioPlayer()
        player.logger = mock_logger
        
        with pytest.raises(PlaybackError) as exc_info:
            await player.play_file("non_existent.wav")
        
        assert "音声ファイルが見つかりません" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_play_array_success(self, sample_audio_data, mock_logger):
        """numpy配列の再生成功テスト"""
        player = AudioPlayer()
        player.logger = mock_logger
        
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait:
            
            await player.play_array(sample_audio_data, sample_rate=16000)
            
            # 再生メソッドが呼ばれたことを確認
            mock_play.assert_called_once()
            args, kwargs = mock_play.call_args
            np.testing.assert_array_equal(args[0], sample_audio_data)
            assert args[1] == 16000  # sample_rate
    
    @pytest.mark.asyncio
    async def test_play_with_volume(self, sample_audio_file, mock_logger):
        """音量調整付き再生のテスト"""
        player = AudioPlayer()
        player.logger = mock_logger
        
        with patch('sounddevice.play') as mock_play, \
             patch('sounddevice.wait') as mock_wait:
            
            await player.play_with_volume(sample_audio_file, volume=0.5)
            
            # 再生された音声データを確認
            args, _ = mock_play.call_args
            played_audio = args[0]
            
            # 音量が適用されていることを確認（最大振幅が0.5以下）
            assert np.max(np.abs(played_audio)) <= 0.5
    
    @pytest.mark.asyncio
    async def test_stop_playback(self, mock_logger):
        """再生停止のテスト"""
        player = AudioPlayer()
        player.logger = mock_logger
        
        with patch('sounddevice.stop') as mock_stop:
            await player.stop()
            
            mock_stop.assert_called_once()
            
            # ログの確認
            info_logs = [log for log in mock_logger.logs if log['level'] == 'info']
            assert any('playback_stop' in log.get('operation', '') for log in info_logs)