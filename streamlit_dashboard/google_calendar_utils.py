import os.path
import datetime
import streamlit as st  # Streamlitの機能を使うために追加
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    """Google Calendar APIのサービスオブジェクトを取得する"""
    creds = None
    
    # ---------------------------------------------------------
    # 1. Streamlit CloudのSecrets（金庫）を確認する処理を追加
    # ---------------------------------------------------------
    if "token" in st.secrets:
        try:
            # Secretsから情報を読み込んで認証情報を作成
            token_info = st.secrets["token"]
            creds = Credentials(
                token=token_info["token"],
                refresh_token=token_info["refresh_token"],
                token_uri=token_info["token_uri"],
                client_id=token_info["client_id"],
                client_secret=token_info["client_secret"],
                scopes=SCOPES
            )
            # トークンの有効期限切れチェック（クラウド用）
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
        except Exception as e:
            # Secretsの読み込みに失敗しても、ローカルファイルの処理に進むためにここでは何もしない
            print(f"Secrets loading skipped: {e}")
            creds = None

    # ---------------------------------------------------------
    # 2. Secretsが使えなかった場合、ローカルファイルを探す（PC用）
    # ---------------------------------------------------------
    if not creds:
        # スクリプトのディレクトリを基準にパスを設定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        token_path = os.path.join(base_dir, 'token.json')
        client_secret_path = os.path.join(base_dir, 'client_secret.json')

        # token.json があれば読み込む
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        # 有効な認証情報がない場合、ログインフローを実行
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            
            if not creds:
                # ローカル実行で client_secret.json がない場合はエラー
                if not os.path.exists(client_secret_path):
                    # Secretsもなくファイルもない場合
                    return None, "認証情報が見つかりません。Streamlit CloudのSecretsを設定するか、ローカルにclient_secret.jsonを配置してください。"
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        client_secret_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    return None, f"認証フローでエラーが発生しました: {str(e)}"
            
            # 次回のために認証情報を保存（ローカル実行時のみ）
            try:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            except Exception:
                pass # クラウド環境などで書き込めない場合は無視

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