#!/bin/bash

# Voice Gemini App セットアップスクリプト

echo "=========================================="
echo "Voice Gemini App セットアップを開始します"
echo "=========================================="

# Python バージョンの確認
echo "Pythonバージョンを確認しています..."
python_version=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "エラー: Python3が見つかりません。Python 3.10以上をインストールしてください。"
    exit 1
fi
echo "✓ $python_version"

# 仮想環境の作成
if [ ! -d "venv" ]; then
    echo "仮想環境を作成しています..."
    python3 -m venv venv
    echo "✓ 仮想環境を作成しました"
else
    echo "✓ 仮想環境は既に存在します"
fi

# 仮想環境の有効化
echo "仮想環境を有効化しています..."
source venv/bin/activate

# pipのアップグレード
echo "pipをアップグレードしています..."
pip install --upgrade pip

# 依存関係のインストール
echo "依存関係をインストールしています..."
pip install -r requirements.txt

# .envファイルの作成
if [ ! -f ".env" ]; then
    echo ".envファイルを作成しています..."
    cp .env.example .env
    echo "✓ .envファイルを作成しました"
    echo ""
    echo "⚠️  重要: .envファイルを編集してGEMINI_API_KEYを設定してください"
    echo "   編集コマンド: nano .env"
else
    echo "✓ .envファイルは既に存在します"
fi

# ログディレクトリの作成
mkdir -p logs
echo "✓ ログディレクトリを作成しました"

# 音声デバイスの確認
echo ""
echo "利用可能な音声デバイス:"
python -m sounddevice

echo ""
echo "=========================================="
echo "✅ セットアップが完了しました！"
echo "=========================================="
echo ""
echo "次のステップ:"
echo "1. .envファイルを編集してGEMINI_API_KEYを設定"
echo "2. python -m src.main でアプリケーションを起動"
echo ""