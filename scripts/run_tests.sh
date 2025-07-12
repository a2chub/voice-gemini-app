#!/bin/bash

# テスト実行スクリプト

echo "Voice Gemini App テストを実行します"
echo "===================================="

# 仮想環境の有効化
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# テストの実行
echo "単体テストを実行しています..."
pytest tests/ -v --tb=short

# カバレッジレポート（オプション）
if [ "$1" == "--coverage" ]; then
    echo ""
    echo "カバレッジレポートを生成しています..."
    pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo "HTMLカバレッジレポート: htmlcov/index.html"
fi

# lintチェック（オプション）
if [ "$1" == "--lint" ] || [ "$2" == "--lint" ]; then
    echo ""
    echo "コードスタイルをチェックしています..."
    
    echo "Black..."
    black --check src/ tests/
    
    echo "Flake8..."
    flake8 src/ tests/
    
    echo "mypy..."
    mypy src/
fi

echo ""
echo "テスト完了！"