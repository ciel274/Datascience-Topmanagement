# SPI対策 ボトルネック発見ダッシュボード

企業レベルのSPI試験対策ダッシュボード。PythonとStreamlitでWebアプリケーションを構築しました。

## 📋 プロジェクト概要

このダッシュボードは、SPI試験の学習進捗を可視化し、ボトルネック（改善が必要な単元）を自動発見するツールです。

**主な特徴：**
- Pythonコードだけで開発（HTML/CSS/JavaScript不要）
- インタラクティブなUI（Streamlit）
- 企業レベルの洗練されたデザイン
- リアルタイムデータ分析
- CSVデータのアップロード・ダウンロード対応

## 🛠️ 技術スタック

| 技術 | 用途 |
|------|------|
| **Python 3.11+** | プログラミング言語 |
| **Streamlit** | Webアプリケーションフレームワーク |
| **Pandas** | データ処理・分析 |
| **Plotly** | インタラクティブグラフ表示 |
| **NumPy** | 数値計算 |

## 📁 プロジェクト構造

```
spi_dashboard/
├── app.py                  # メインアプリケーション
├── requirements.txt        # Python依存パッケージ
├── .gitignore             # Git除外ファイル
├── README.md              # このファイル
├── VSCODE_SETUP.md        # VSCodeセットアップガイド
├── data/
│   └── sample_log.csv     # サンプル学習ログ
└── utils/
    └── analysis.py        # データ分析ユーティリティ
```

## 🚀 クイックスタート

### 前提条件

- Python 3.8以上
- VSCode（推奨）
- Git（GitHub連携時）

### セットアップ手順

#### 1. リポジトリをクローン

```bash
git clone https://github.com/yourusername/spi_dashboard.git
cd spi_dashboard
```

#### 2. 仮想環境を作成

```bash
python -m venv venv
```

#### 3. 仮想環境を有効化

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

#### 4. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

#### 5. Streamlitアプリを起動

```bash
streamlit run app.py
```

ブラウザが自動的に開き、`http://localhost:8501` でダッシュボードが表示されます。

## 📖 使用方法

### 基本的な流れ

1. **企業・目標を設定** - サイドバーで志望企業名と目標正答率を入力
2. **学習データを入力** - 問題を解いた日時、正誤、解答時間などを記録
3. **分析結果を確認** - KPIカード、グラフ、科目別達成状況を確認
4. **データを保存** - CSVをダウンロードして保存

### サイドバーの機能

#### 🏢 企業・目標設定
- **志望企業名** - 対策対象の企業を入力
- **目標正答率** - SPI試験での目標を設定（0-100%）
- **時間設定** - 解答時間の基準を選択（標準/厳しく/緩く）

#### ✏️ 学習データ入力
- **日付** - 問題を解いた日を選択
- **科目** - 「非言語」または「言語」を選択
- **ジャンル** - 科目に応じたジャンルを選択
- **単元** - ジャンルに応じた単元を選択
- **解答時間** - 実際にかかった時間（秒）
- **正誤** - 「〇」（正解）または「✕」（不正解）
- **ミスの原因** - 不正解の場合、原因を選択
- **学習時間** - その問題に費やした学習時間（分）

#### 📁 ファイル管理
- **問題マスタCSV** - 独自の問題セットをアップロード
- **学習ログCSV** - 過去の学習データをアップロード

#### 🔍 分析期間
- **開始日** - 分析を開始する日を選択
- **終了日** - 分析を終了する日を選択

### メインエリアの表示内容

#### ⚡ アクションカード
次週の重点単元と改善理由を表示

#### 📊 主要指標（KPIカード）
- **現在の正答率** - 期間平均の正答率
- **目標との差** - 目標に対する進捗
- **時間超過率** - 目標時間を超過した割合
- **総演習数** - 累計問題数

#### 📈 分析グラフ
- **日別正答率トレンド** - 正答率の推移（折れ線グラフ）
- **達成度ゲージ** - 目標に対する進捗（円形ゲージ）
- **優先単元 Top 5** - 改善が必要な単元（横棒グラフ）
- **誤答原因分析** - ミスの原因分析（棒グラフ）

#### 📚 科目別達成状況
各科目の正答率とプログレスバー

#### 📋 入力済みデータ一覧
入力したデータの表示とCSVダウンロード

## 🎨 カスタマイズ

### カラーの変更

`app.py` の先頭部分でカラー定義を変更：

```python
PRIMARY = "#3B82F6"      # プライマリ色（青）
ACCENT = "#F97316"       # アクセント色（オレンジ）
SUCCESS = "#10B981"      # 成功色（緑）
WARNING = "#F59E0B"      # 警告色（黄）
DANGER = "#EF4444"       # 危険色（赤）
NEUTRAL = "#6B7280"      # ニュートラル色（グレー）
```

### 問題マスタの追加

`app.py` の `DEFAULT_MASTER_ROWS` に新しい問題を追加：

```python
["N-A07", "非言語", "推論", "新しい単元", 120, 85, "高", 5],
```

### CSSの変更

`app.py` の `<style>` セクションでCSSを編集

## 🐛 トラブルシューティング

### Q: ブラウザが自動的に開かない

A: ターミナルに表示されたURLをブラウザにコピー&ペーストしてください。
例：`http://localhost:8501`

### Q: 「streamlit: command not found」エラー

A: 仮想環境が有効になっていません。以下を実行：
```bash
source venv/bin/activate  # macOS/Linux
# または
.\venv\Scripts\Activate.ps1  # Windows
```

### Q: ポート8501が既に使用されている

A: 別のポートで実行：
```bash
streamlit run app.py --server.port 8502
```

### Q: テキストが見えない

A: ブラウザをリロード（F5キー）してください

### Q: グラフが表示されない

A: Plotlyを再インストール：
```bash
pip install --upgrade plotly
```

## 📊 データ形式

### 学習ログCSV

```csv
日付,問題ID,正誤,解答時間(秒),ミスの原因,学習投入時間(分)
2024-11-01,N-A01,〇,110,-,15
2024-11-01,N-A02,✕,120,時間不足,20
2024-11-02,N-A03,〇,140,-,18
```

### 問題マスタCSV

```csv
問題ID,科目,ジャンル,単元,目標解答時間(秒),目標正答率(%),難易度,出題頻度(重み)
N-A01,非言語,推論,集合の推論 (ベン図),120,85,高,5
N-A02,非言語,推論,論理的な推論 (真偽・順序),100,80,中,4
```

## 🔧 開発環境

### VSCodeの推奨拡張機能

- **Python** - ms-python.python
- **Pylance** - ms-python.vscode-pylance
- **Streamlit** - streamlit.streamlit
- **Git Graph** - mhutchie.git-graph

### VSCodeの設定

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

## 📝 ログ・デバッグ

### Streamlitのログを表示

```bash
streamlit run app.py --logger.level=debug
```

### Pythonのデバッグ

VSCodeでブレークポイントを設定して実行：
1. `app.py` の行番号の左をクリック
2. F5キーを押してデバッグ開始

## 🚀 デプロイ

### Streamlit Community Cloudへのデプロイ

1. GitHubにリポジトリをプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud) にアクセス
3. 「New app」をクリック
4. GitHubリポジトリを選択
5. `app.py` を指定
6. 「Deploy」をクリック

## 📄 ライセンス

MIT License

## 👨‍💻 作成者

SPI対策ダッシュボード開発チーム

## 🤝 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📞 サポート

問題が発生した場合は、GitHubのissueを作成してください。
