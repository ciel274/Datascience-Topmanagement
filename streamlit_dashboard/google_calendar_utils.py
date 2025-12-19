import os.path
import datetime
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
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
    # 2. ローカルファイル（PC用）を確認
    # ---------------------------------------------------------
    if not creds and os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception:
            # トークンファイルが壊れている場合は削除して再作成
            os.remove('token.json')
            creds = None

    # 有効期限切れの再取得
    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # リフレッシュ失敗時は再ログイン
                creds = None

    # ---------------------------------------------------------
    # 3. 認証情報がない場合、新規ログイン（ローカルのみ）
    # ---------------------------------------------------------
    if not creds:
        if os.path.exists('credentials.json'):
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                # トークンを保存
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                return None, f"ログインフローエラー: {str(e)}"
        else:
            return None, "認証情報が見つかりません。credentials.jsonを配置するか、Streamlit CloudのSecretsを設定してください。"

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

def get_user_info(creds):
    """ユーザー情報を取得する"""
    try:
        service = build('oauth2', 'v2', credentials=creds)
        user_info = service.userinfo().get().execute()
        return user_info, None
    except Exception as e:
        return None, f"ユーザー情報の取得に失敗しました: {str(e)}"

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