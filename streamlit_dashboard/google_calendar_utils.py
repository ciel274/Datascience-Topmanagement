import os.path
import datetime
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets'
]

def get_credentials():
    """認証情報を取得する共通関数"""
    creds = None
    
    # ---------------------------------------------------------
    # 1. Streamlit CloudのSecrets（金庫）を確認
    # ---------------------------------------------------------
    try:
        if "token" in st.secrets:
            try:
                token_info = st.secrets["token"]
                creds = Credentials(
                    token=token_info["token"],
                    refresh_token=token_info["refresh_token"],
                    token_uri=token_info["token_uri"],
                    client_id=token_info["client_id"],
                    client_secret=token_info["client_secret"],
                    scopes=SCOPES
                )
                # トークンの有効期限切れチェック
                if not creds.valid:
                    if creds.expired and creds.refresh_token:
                        creds.refresh(Request())
            except Exception as e:
                return None, f"Secretsの設定エラー: {str(e)}"
    except Exception:
        # secrets.tomlが存在しない場合は無視してローカルファイル確認へ進む
        pass

    # ---------------------------------------------------------
    # 2. ローカルファイル（PC用）を確認 - クラウドでは無視
    # ---------------------------------------------------------
    if not creds and os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
        except Exception as e:
            return None, f"ローカル認証エラー: {str(e)}"

    # ---------------------------------------------------------
    # 3. どちらも見つからない場合
    # ---------------------------------------------------------
    if not creds or not creds.valid:
        return None, "認証情報が見つかりません。Streamlit CloudのSecrets設定を確認してください。"

    return creds, None

def get_calendar_service():
    """Google Calendar APIのサービスオブジェクトを取得する"""
    creds, error = get_credentials()
    if error:
        return None, error

    try:
        service = build('calendar', 'v3', credentials=creds)
        return service, None
    except Exception as e:
        return None, f"サービスの構築に失敗しました: {str(e)}"

def add_event_to_calendar(service, summary, start_time, end_time, description=None):
    """カレンダーに予定を追加する"""
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'Asia/Tokyo',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'Asia/Tokyo',
        },
    }

    try:
        event = service.events().insert(calendarId='primary', body=event).execute()
        return event.get('htmlLink'), None
    except Exception as e:
        return None, str(e)