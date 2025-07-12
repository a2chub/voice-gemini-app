"""
Gemini API プロセッサー

Google Gemini APIを使用してテキスト処理を行います。
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Dict, Any, List
import google.generativeai as genai

from genai_processors import ProcessorPart

from .base import TextProcessorBase
from ..config import get_config
from ..utils.exceptions import GeminiAPIError, APIError


class GeminiChatProcessor(TextProcessorBase):
    """Gemini APIを使用した対話処理プロセッサー"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Gemini プロセッサーの初期化
        
        Args:
            model_name: 使用するGeminiモデル名
            temperature: 生成温度（0.0-1.0）
            max_tokens: 最大トークン数
            system_prompt: システムプロンプト
        """
        super().__init__("gemini_chat")
        
        config = get_config()
        
        # APIキーの設定
        genai.configure(api_key=config.gemini.api_key)
        
        # パラメータの設定
        self.model_name = model_name or config.gemini.model
        self.temperature = temperature or config.gemini.temperature
        self.max_tokens = max_tokens or config.gemini.max_tokens
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # モデルの初期化
        self._model = None
        self._chat_session = None
        
        # メトリクスの初期化
        self.add_metric('model_name', self.model_name)
        self.add_metric('temperature', self.temperature)
        self.add_metric('max_tokens', self.max_tokens)
    
    def _get_default_system_prompt(self) -> str:
        """デフォルトのシステムプロンプトを取得"""
        return (
            "あなたは親切で知識豊富なAIアシスタントです。"
            "ユーザーの質問に対して、正確で有用な回答を提供してください。"
            "回答は簡潔で分かりやすく、日本語で応答してください。"
        )
    
    async def _initialize_model(self):
        """Geminiモデルを初期化"""
        if self._model is None:
            await self.logger.info(
                operation="gemini_model_initialization",
                message=f"Geminiモデル '{self.model_name}' を初期化しています",
                model=self.model_name
            )
            
            try:
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens,
                    }
                )
                
                # チャットセッションの開始
                self._chat_session = self._model.start_chat(history=[])
                
                # システムプロンプトを送信
                if self.system_prompt:
                    await self._send_system_prompt()
                
                await self.logger.info(
                    operation="gemini_model_initialized",
                    message="Geminiモデルの初期化が完了しました"
                )
                
            except Exception as e:
                await self.logger.error(
                    operation="gemini_initialization_error",
                    message="Geminiモデルの初期化に失敗しました",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise GeminiAPIError(f"Geminiモデルの初期化に失敗しました: {e}")
    
    async def _send_system_prompt(self):
        """システムプロンプトを送信"""
        loop = asyncio.get_event_loop()
        
        try:
            # システムプロンプトの送信
            await loop.run_in_executor(
                None,
                lambda: self._chat_session.send_message(
                    f"システム設定: {self.system_prompt}"
                )
            )
        except Exception as e:
            await self.logger.warning(
                operation="system_prompt_error",
                message="システムプロンプトの送信に失敗しました",
                error=str(e)
            )
    
    async def process(
        self,
        input_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """
        テキストをGemini APIで処理
        
        Args:
            input_stream: テキストを含む入力ストリーム
        
        Yields:
            Geminiの応答を含むProcessorPart
        """
        # モデルの初期化
        await self._initialize_model()
        
        async for part in input_stream:
            # テキストの取得
            text = self._extract_text(part)
            if not text:
                continue
            
            # Gemini APIの呼び出し
            try:
                response = await self._generate_response(text)
                
                # ProcessorPartの作成
                yield ProcessorPart(
                    content=response['text'],
                    metadata={
                        'gemini_response': response,
                        'processor': 'gemini',
                        'model': self.model_name,
                        'input_text': text
                    }
                )
                
            except Exception as e:
                await self.logger.error(
                    operation="gemini_api_error",
                    message="Gemini API呼び出し中にエラーが発生しました",
                    error=str(e),
                    error_type=type(e).__name__,
                    input_text=text[:100]
                )
                raise GeminiAPIError(f"Gemini API呼び出しに失敗しました: {e}")
    
    def _extract_text(self, part: ProcessorPart) -> Optional[str]:
        """ProcessorPartからテキストを抽出"""
        # contentにテキストが含まれている場合
        if part.content:
            return part.content.strip()
        
        # metadataにtranscriptionが含まれている場合
        if part.metadata and 'transcription' in part.metadata:
            transcription = part.metadata['transcription']
            if isinstance(transcription, dict) and 'text' in transcription:
                return transcription['text'].strip()
        
        return None
    
    async def _generate_response(self, text: str) -> Dict[str, Any]:
        """Gemini APIで応答を生成"""
        loop = asyncio.get_event_loop()
        
        await self.logger.info(
            operation="gemini_api_call_start",
            message="Gemini APIを呼び出しています",
            input_preview=text[:100],
            input_length=len(text)
        )
        
        start_time = time.time()
        
        try:
            # API呼び出し（ブロッキング操作を非同期で実行）
            response = await loop.run_in_executor(
                None,
                lambda: self._chat_session.send_message(text)
            )
            
            api_latency = time.time() - start_time
            
            # 応答の処理
            result = {
                'text': response.text,
                'api_latency': api_latency,
                'prompt_tokens': getattr(response, 'prompt_token_count', 0),
                'completion_tokens': getattr(response, 'candidates_token_count', 0),
                'total_tokens': getattr(response, 'total_token_count', 0)
            }
            
            # メトリクスの更新
            self.add_metric('api_latency', api_latency)
            self.add_metric('prompt_tokens', result['prompt_tokens'])
            self.add_metric('completion_tokens', result['completion_tokens'])
            self.add_metric('total_tokens', result['total_tokens'])
            
            await self.logger.info(
                operation="gemini_api_call_complete",
                message="Gemini API呼び出しが完了しました",
                response_preview=result['text'][:100],
                response_length=len(result['text']),
                api_latency=api_latency,
                total_tokens=result['total_tokens']
            )
            
            return result
            
        except Exception as e:
            api_latency = time.time() - start_time
            self.add_metric('api_latency', api_latency)
            self.add_metric('api_error', str(e))
            
            # エラーの種類に応じた処理
            if "quota" in str(e).lower():
                raise APIError("APIクォータを超過しました", status_code=429)
            elif "invalid" in str(e).lower():
                raise APIError("無効なリクエストです", status_code=400)
            else:
                raise
    
    def reset_conversation(self):
        """会話履歴をリセット"""
        if self._model:
            self._chat_session = self._model.start_chat(history=[])
            if self.system_prompt:
                # 新しいセッションでシステムプロンプトを再送信
                asyncio.create_task(self._send_system_prompt())