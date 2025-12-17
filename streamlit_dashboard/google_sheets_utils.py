import gspread
import pandas as pd
import streamlit as st
from google_calendar_utils import get_credentials

# ユーザーが作成したスプレッドシートのID
SPREADSHEET_ID = "1jufRuVWoBlXnp-GnT2ELgCTKeQtuLt7EeAgDE2xAml0"

class GoogleSheetsManager:
    def __init__(self):
        self.client = None
        self.spreadsheet = None

    def connect(self):
        """Google Sheets APIに接続"""
        if self.client:
            return True, None

        creds, error = get_credentials()
        if error:
            return False, error

        try:
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            return True, None
        except Exception as e:
            return False, f"スプレッドシート接続エラー: {str(e)}"

    def get_or_create_user_sheet(self, username):
        """ユーザー用のシートを取得、なければ作成"""
        success, error = self.connect()
        if not success:
            return None, error

        try:
            # ワークシートを探す
            try:
                worksheet = self.spreadsheet.worksheet(username)
            except gspread.WorksheetNotFound:
                # なければ作成（ヘッダー付き）
                worksheet = self.spreadsheet.add_worksheet(title=username, rows=100, cols=10)
                # デフォルトのヘッダー
                header = ["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"]
                worksheet.append_row(header)
            
            return worksheet, None
        except Exception as e:
            return None, f"シート取得エラー: {str(e)}"

    def get_or_create_notes_sheet(self, username):
        """ユーザー用の復習ノートシートを取得、なければ作成"""
        sheet_name = f"{username}_notes"
        success, error = self.connect()
        if not success:
            return None, error

        try:
            try:
                worksheet = self.spreadsheet.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=5)
                header = ["問題ID", "メモ", "登録日時"]
                worksheet.append_row(header)
            
            return worksheet, None
        except Exception as e:
            return None, f"ノートシート取得エラー: {str(e)}"

    def load_data(self, username):
        """ユーザーの学習データを読み込む"""
        worksheet, error = self.get_or_create_user_sheet(username)
        if error:
            return pd.DataFrame(), error

        try:
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"データ読み込みエラー: {str(e)}"

    def load_notes(self, username):
        """ユーザーの復習ノートを読み込む"""
        worksheet, error = self.get_or_create_notes_sheet(username)
        if error:
            return pd.DataFrame(), error

        try:
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"ノート読み込みエラー: {str(e)}"

    def add_data(self, username, data_row):
        """学習データを追加"""
        worksheet, error = self.get_or_create_user_sheet(username)
        if error:
            return False, error

        try:
            # data_rowは辞書型を想定、ヘッダー順にリスト化
            header = ["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"]
            row_values = [data_row.get(col, "") for col in header]
            worksheet.append_row(row_values)
            return True, None
        except Exception as e:
            return False, f"データ追加エラー: {str(e)}"

    def add_note(self, username, note_row):
        """復習ノートを追加"""
        worksheet, error = self.get_or_create_notes_sheet(username)
        if error:
            return False, error

        try:
            header = ["問題ID", "メモ", "登録日時"]
            row_values = [note_row.get(col, "") for col in header]
            worksheet.append_row(row_values)
            return True, None
        except Exception as e:
            return False, f"ノート追加エラー: {str(e)}"

    def sync_from_csv(self, username, csv_path):
        """既存のCSVデータをスプレッドシートに同期（上書きモード）"""
        try:
            df = pd.read_csv(csv_path)
            worksheet, error = self.get_or_create_user_sheet(username)
            if error:
                return False, error
            
            # 全データクリアして書き直し
            worksheet.clear()
            # ヘッダー書き込み
            worksheet.append_row(df.columns.tolist())
            # データ書き込み
            worksheet.append_rows(df.values.tolist())
            return True, None
        except Exception as e:
            return False, f"同期エラー: {str(e)}"

    def get_or_create_ranking_sheet(self):
        """ランキング用のシートを取得、なければ作成"""
        sheet_name = "Ranking"
        success, error = self.connect()
        if not success:
            return None, error

        try:
            try:
                worksheet = self.spreadsheet.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=3)
                header = ["User", "TotalStudyTime", "LastUpdated"]
                worksheet.append_row(header)
            
            return worksheet, None
        except Exception as e:
            return None, f"ランキングシート取得エラー: {str(e)}"

    def update_ranking(self, username, study_time):
        """ユーザーの学習時間をランキングに更新"""
        worksheet, error = self.get_or_create_ranking_sheet()
        if error:
            return False, error

        try:
            # 全データを取得して該当ユーザーを探す
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            from datetime import datetime
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if not df.empty and "User" in df.columns and username in df["User"].values:
                # 更新
                cell = worksheet.find(username)
                worksheet.update_cell(cell.row, 2, study_time) # TotalStudyTime
                worksheet.update_cell(cell.row, 3, now_str)    # LastUpdated
            else:
                # 新規追加
                worksheet.append_row([username, study_time, now_str])
                
            return True, None
        except Exception as e:
            return False, f"ランキング更新エラー: {str(e)}"

    def get_ranking(self):
        """ランキングデータを取得"""
        worksheet, error = self.get_or_create_ranking_sheet()
        if error:
            return pd.DataFrame(), error

        try:
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            if not df.empty:
                df["TotalStudyTime"] = pd.to_numeric(df["TotalStudyTime"], errors="coerce").fillna(0)
                df = df.sort_values("TotalStudyTime", ascending=False).reset_index(drop=True)
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"ランキング読み込みエラー: {str(e)}"
