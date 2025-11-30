# VSCode + Python + Streamlit セットアップガイド

VSCodeでSPI対策ダッシュボードを開発・実行するための完全なガイドです。

## 📋 前提条件

- **Windows/macOS/Linux** - いずれでも動作
- **VSCode** - [公式サイト](https://code.visualstudio.com/)からダウンロード
- **Python 3.8以上** - [公式サイト](https://www.python.org/)からダウンロード
- **Git** - （GitHub連携時に必要）

## 🚀 ステップ1: 環境構築

### 1.1 Pythonのインストール確認

ターミナル/コマンドプロンプトで以下を実行：

```bash
python --version
```

**出力例：**
```
Python 3.11.0
```

Pythonがインストールされていない場合は、[公式サイト](https://www.python.org/)からダウンロードしてインストールしてください。

### 1.2 VSCodeのインストール

[公式サイト](https://code.visualstudio.com/)からダウンロードしてインストール

### 1.3 VSCode拡張機能のインストール

VSCodeを開き、左サイドバーの「拡張機能」アイコンをクリック

以下の拡張機能をインストール：

1. **Python** (ms-python.python)
   - 検索ボックスに「Python」と入力
   - 「Install」をクリック

2. **Pylance** (ms-python.vscode-pylance)
   - 検索ボックスに「Pylance」と入力
   - 「Install」をクリック

3. **Streamlit** (streamlit.streamlit)
   - 検索ボックスに「Streamlit」と入力
   - 「Install」をクリック

## 📁 ステップ2: プロジェクトフォルダの準備

### 2.1 プロジェクトフォルダを作成

**Windows:**
```bash
mkdir C:\Users\YourName\spi_dashboard
cd C:\Users\YourName\spi_dashboard
```

**macOS/Linux:**
```bash
mkdir ~/spi_dashboard
cd ~/spi_dashboard
```

### 2.2 VSCodeでフォルダを開く

1. VSCodeを起動
2. 「File」→「Open Folder」をクリック
3. 作成したフォルダを選択
4. 「Select Folder」をクリック

### 2.3 ファイルをコピー

以下のファイルをプロジェクトフォルダにコピー：

- `app.py` - メインアプリケーション
- `requirements.txt` - 依存パッケージ
- `README.md` - プロジェクト説明
- `data/sample_log.csv` - サンプルデータ

## 🔧 ステップ3: 仮想環境の作成

### 3.1 ターミナルを開く

VSCodeで `Ctrl+`` (バッククォート) を押してターミナルを開く

### 3.2 仮想環境を作成

```bash
python -m venv venv
```

### 3.3 仮想環境を有効化

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**確認：** プロンプトの先頭に `(venv)` が表示されればOK

```
(venv) C:\Users\YourName\spi_dashboard>
```

### 3.4 Pythonインタプリタを設定

1. VSCode左下の「Select Python Interpreter」をクリック
2. 「./venv/bin/python」 または 「.\venv\Scripts\python.exe」 を選択

## 📦 ステップ4: 依存パッケージのインストール

ターミナルで以下を実行：

```bash
pip install -r requirements.txt
```

**出力例：**
```
Successfully installed streamlit-1.28.1 pandas-2.1.3 numpy-1.26.2 plotly-5.18.0
```

## ▶️ ステップ5: Streamlitアプリを起動

ターミナルで以下を実行：

```bash
streamlit run app.py
```

**出力例：**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://169.254.0.21:8501
  External URL: http://13.212.35.196:8501
```

ブラウザが自動的に開き、ダッシュボードが表示されます。

## 🎯 ステップ6: 開発ワークフロー

### コードの編集

1. VSCodeで `app.py` を開く
2. コードを編集
3. ファイルを保存（Ctrl+S）
4. Streamlitが自動的に再読み込み
5. ブラウザで変更を確認

### デバッグ

**方法1: print()で出力**

```python
print("デバッグ情報:", variable)
```

ターミナルに出力されます。

**方法2: ブレークポイント**

1. `app.py` の行番号の左をクリック（赤い点が表示される）
2. F5キーを押してデバッグ開始
3. 実行が一時停止する

## 📝 VSCode設定（推奨）

### .vscode/settings.jsonを作成

VSCodeで以下の手順：

1. 「File」→「Preferences」→「Settings」
2. 右上の「Open Settings (JSON)」アイコンをクリック
3. 以下を追加：

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true
    }
}
```

## 🔄 毎回の起動方法

### 簡単な起動手順

1. VSCodeでプロジェクトフォルダを開く
2. ターミナルを開く（Ctrl+``）
3. 仮想環境を有効化：
   ```bash
   # Windows
   .\venv\Scripts\activate.bat
   
   # macOS/Linux
   source venv/bin/activate
   ```
4. アプリを起動：
   ```bash
   streamlit run app.py
   ```

## 💡 よくある問題と解決方法

### Q1: 「streamlit: command not found」エラー

**原因:** 仮想環境が有効になっていない

**解決方法:**
```bash
# Windows
.\venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### Q2: ブラウザが自動的に開かない

**原因:** ファイアウォール設定またはブラウザの問題

**解決方法:**
- ターミナルに表示されたURLをブラウザにコピー&ペースト
- 例：`http://localhost:8501`

### Q3: ポート8501が既に使用されている

**原因:** 別のStreamlitプロセスが実行中

**解決方法:**
```bash
streamlit run app.py --server.port 8502
```

### Q4: 「ModuleNotFoundError」エラー

**原因:** 必要なパッケージがインストールされていない

**解決方法:**
```bash
pip install -r requirements.txt
```

### Q5: テキストが見えない

**原因:** ブラウザのキャッシュ問題

**解決方法:**
- ブラウザをリロード（F5キー）
- または Ctrl+Shift+Delete でキャッシュをクリア

### Q6: グラフが表示されない

**原因:** Plotlyがインストールされていない

**解決方法:**
```bash
pip install --upgrade plotly
```

## 🎨 カスタマイズ例

### カラーを変更

`app.py` の以下の部分を編集：

```python
# ===== カラー定義 =====
PRIMARY = "#3B82F6"      # プライマリ色を変更
ACCENT = "#F97316"       # アクセント色を変更
SUCCESS = "#10B981"      # 成功色を変更
```

### 新しい問題を追加

`app.py` の `DEFAULT_MASTER_ROWS` に追加：

```python
["N-A07", "非言語", "推論", "新しい単元", 120, 85, "高", 5],
```

### ページタイトルを変更

`app.py` の以下の部分を編集：

```python
st.set_page_config(
    page_title="新しいタイトル",  # ここを変更
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## 📚 参考資料

- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [Python公式ドキュメント](https://docs.python.org/)
- [VSCode公式ドキュメント](https://code.visualstudio.com/docs)
- [Pandas公式ドキュメント](https://pandas.pydata.org/docs/)
- [Plotly公式ドキュメント](https://plotly.com/python/)

## ✅ セットアップ完了チェックリスト

- [ ] Pythonをインストール
- [ ] VSCodeをインストール
- [ ] VSCode拡張機能をインストール
- [ ] プロジェクトフォルダを作成
- [ ] ファイルをコピー
- [ ] 仮想環境を作成
- [ ] 仮想環境を有効化
- [ ] 依存パッケージをインストール
- [ ] `streamlit run app.py` を実行
- [ ] ブラウザでダッシュボードが表示される

すべてのチェックが完了したら、開発を開始できます！

## 🚀 次のステップ

1. **データを入力** - サイドバーから学習データを入力
2. **ダッシュボードを確認** - KPIカードとグラフを確認
3. **カスタマイズ** - 企業名や目標を設定
4. **GitHub連携** - コードをGitHubにプッシュ
5. **Streamlit Cloud** - アプリをデプロイ

詳細は `README.md` を参照してください。
