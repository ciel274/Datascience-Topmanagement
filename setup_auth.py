import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# 必要な権限（カレンダー＋スプレッドシート）
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets'
]

def main():
    creds = None
    # 既存のトークンがあれば読み込む
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # 有効なトークンがない場合は新規取得
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("トークンを更新中...")
            creds.refresh(Request())
        else:
            print("新しい認証を開始します。ブラウザが開きます...")
            # credentials.json または client_secret.json を探す
            possible_paths = [
                'streamlit_dashboard/client_secret.json',
                'client_secret.json',
                'streamlit_dashboard/credentials.json',
                'credentials.json'
            ]
            
            cred_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    cred_path = p
                    break
            
            if not cred_path:
                print("エラー: client_secret.json または credentials.json が見つかりません。")
                return

            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # トークンを保存 (streamlit_dashboardディレクトリに保存)
        token_path = 'streamlit_dashboard/token.json'
        # もしディレクトリがなければカレントに
        if not os.path.exists('streamlit_dashboard'):
            token_path = 'token.json'
            
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
        print(f"認証成功！ {token_path} を更新しました。")

if __name__ == '__main__':
    main()
