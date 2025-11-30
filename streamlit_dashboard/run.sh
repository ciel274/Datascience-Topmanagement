#!/bin/bash

# SPI対策 ボトルネック発見 - Streamlit版
# 実行スクリプト

echo "🚀 SPI対策 ボトルネック発見 を起動します..."
echo ""

# 仮想環境の確認と作成
if [ ! -d "venv" ]; then
    echo "📦 仮想環境を作成中..."
    python3 -m venv venv
    echo "✅ 仮想環境を作成しました"
fi

# 仮想環境の有効化
source venv/bin/activate

# パッケージのインストール確認
echo "📚 必要なパッケージをインストール中..."
pip install -q streamlit pandas numpy plotly

# Streamlitの起動
echo ""
echo "✅ セットアップ完了！"
echo ""
echo "📊 ダッシュボードを起動します..."
echo "ブラウザで http://localhost:8501 を開いてください"
echo ""

streamlit run dashboard.py
