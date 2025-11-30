@echo off
REM SPI対策 ボトルネック発見 - Streamlit版
REM 実行スクリプト (Windows用)

echo.
echo 🚀 SPI対策 ボトルネック発見 を起動します...
echo.

REM 仮想環境の確認と作成
if not exist "venv" (
    echo 📦 仮想環境を作成中...
    python -m venv venv
    echo ✅ 仮想環境を作成しました
)

REM 仮想環境の有効化
call venv\Scripts\activate.bat

REM パッケージのインストール
echo 📚 必要なパッケージをインストール中...
pip install -q streamlit pandas numpy plotly

REM Streamlitの起動
echo.
echo ✅ セットアップ完了！
echo.
echo 📊 ダッシュボードを起動します...
echo ブラウザで http://localhost:8501 を開いてください
echo.

streamlit run dashboard.py

pause
