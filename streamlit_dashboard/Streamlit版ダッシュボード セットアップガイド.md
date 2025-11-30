# Streamlit版ダッシュボード セットアップガイド

このガイドでは、VSCode、Python、Streamlitを使用してダッシュボードを実行する方法を説明します。

## 📋 前提条件

- **Python 3.8以上** がインストール済み
- **VSCode** がインストール済み
- **インターネット接続** が可能

## 🚀 クイックスタート

### 方法1: スクリプトで自動実行（推奨）

#### macOS/Linux

```bash
cd /path/to/streamlit_dashboard
chmod +x run.sh
./run.sh
```

#### Windows

```bash
cd C:\path\to\streamlit_dashboard
run.bat
```

### 方法2: 手動で実行

#### ステップ1: 仮想環境の作成

```bash
# プロジェクトディレクトリに移動
cd streamlit_dashboard

# 仮想環境を作成
python3 -m venv venv
```

#### ステップ2: 仮想環境の有効化

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

#### ステップ3: 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install streamlit pandas numpy plotly
```

#### ステップ4: ダッシュボードを起動

```bash
streamlit run dashboard.py
```

ブラウザが自動的に開き、`http://localhost:8501` でダッシュボードが表示されます。

## 🎯 VSCodeでの実行方法

### 方法1: VSCode統合ターミナルを使用

1. VSCodeでプロジェクトフォルダを開く
2. ターミナルを開く（`Ctrl+`` または `Cmd+``）
3. 以下のコマンドを実行：

```bash
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate.bat  # Windows

streamlit run dashboard.py
```

### 方法2: VSCode Python拡張機能を使用

1. Python拡張機能をインストール
2. `dashboard.py`を開く
3. 右上の「▶」ボタンをクリック
4. ターミナルで以下を実行：

```bash
streamlit run dashboard.py
```

## 📊 ダッシュボードの使用方法

### 1. 初期設定

ダッシュボード起動後、サイドバーで以下を設定：

- **志望企業名** - 対策対象の企業（例：〇〇商社）
- **目標正答率** - SPI試験での目標（例：80%）
- **時間設定** - 解答時間の基準を選択

### 2. データ入力

サイドバーの「✏️ 学習データ入力」セクションで：

1. 日付を選択
2. 科目 → ジャンル → 単元を順に選択
3. 解答時間、正誤、ミスの原因を入力
4. 「➕ データを追加」をクリック

### 3. サンプルデータで試す

`sample_log.csv` をアップロードして、ダッシュボードの機能を試すことができます：

1. サイドバーの「📁 ファイル管理」を展開
2. 「学習ログCSV」に `sample_log.csv` をアップロード
3. ダッシュボードが自動的に更新されます

## 🔧 トラブルシューティング

### エラー: "streamlit: command not found"

**原因**: Streamlitがインストールされていない、または仮想環境が有効になっていない

**解決方法**:
```bash
# 仮想環境が有効か確認（プロンプトに (venv) が表示されるはず）
pip install streamlit
```

### エラー: "ModuleNotFoundError: No module named 'streamlit'"

**原因**: 必要なパッケージがインストールされていない

**解決方法**:
```bash
source venv/bin/activate  # 仮想環境を有効化
pip install -r requirements.txt
```

### ポート8501が既に使用されている

**原因**: 別のStreamlitプロセスが実行中

**解決方法**:
```bash
# 別のポートで実行
streamlit run dashboard.py --server.port 8502
```

### グラフが表示されない

**原因**: Plotlyがインストールされていない

**解決方法**:
```bash
pip install --upgrade plotly
```

### 日本語が文字化けする

**原因**: フォント設定の問題

**解決方法**: 
- macOS: システムに日本語フォントがインストール済み
- Windows: 「Noto Sans JP」フォントをインストール
- Linux: `sudo apt-get install fonts-noto-cjk`

## 📁 ファイル構成

```
streamlit_dashboard/
├── dashboard.py          # メインのダッシュボードアプリ
├── requirements.txt      # Python依存パッケージ
├── sample_log.csv        # サンプル学習ログ
├── run.sh               # macOS/Linux用実行スクリプト
├── run.bat              # Windows用実行スクリプト
├── README.md            # 機能説明書
├── SETUP_GUIDE.md       # このファイル
└── venv/                # 仮想環境（自動生成）
```

## 💡 ヒント

### データの永続化

Streamlitのセッション機能により、ブラウザを閉じるまでデータは保持されます。永続的に保存するには：

1. 「📋 入力済みデータ一覧」セクションでCSVをダウンロード
2. 次回起動時に「📁 ファイル管理」からアップロード

### 複数のデータセットを管理

異なる企業や時期のデータを分析する場合：

1. 各データセットを別のCSVファイルで保存
2. 必要に応じてアップロードして分析

### カスタマイズ

`dashboard.py`の先頭部分で以下をカスタマイズ可能：

- **カラー設定** - PRIMARY、ACCENT等の色コード
- **問題マスタ** - DEFAULT_MASTER_ROWSに問題を追加
- **ページ設定** - st.set_page_config()のパラメータ

## 📞 サポート

問題が解決しない場合：

1. Pythonバージョンを確認: `python --version`
2. パッケージを再インストール: `pip install --upgrade -r requirements.txt`
3. 仮想環境を再作成: `rm -rf venv && python3 -m venv venv`

## ✅ 確認チェックリスト

- [ ] Python 3.8以上がインストール済み
- [ ] 仮想環境が作成済み
- [ ] 必要なパッケージがインストール済み
- [ ] `streamlit run dashboard.py` でダッシュボードが起動
- [ ] ブラウザで `http://localhost:8501` にアクセス可能
- [ ] サイドバーで企業名と目標を入力可能
- [ ] データ入力フォームが動作
- [ ] グラフが表示される

すべてのチェックが完了したら、ダッシュボードの使用を開始できます！
