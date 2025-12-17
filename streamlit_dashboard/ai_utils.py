import google.generativeai as genai
import streamlit as st
import os
from pypdf import PdfReader

def configure_genai():
    """Gemini APIの設定"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません。"
    
    try:
        genai.configure(api_key=api_key)
        return True, None
    except Exception as e:
        return False, f"API設定エラー: {str(e)}"

def extract_text_from_pdf(file_obj):
    """PDFファイルからテキストを抽出"""
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"PDF読み込みエラー: {str(e)}"

def get_gemini_response(prompt, context="", document_content=""):
    """Geminiから応答を取得"""
    success, error = configure_genai()
    if not success:
        return f"エラー: {error}"

    try:
        model = genai.GenerativeModel('gemini-flash-lite-latest')
        
        doc_context = ""
        if document_content:
            doc_context = f"""
            【参照資料（PDF内容）】
            {document_content[:50000]}  # 文字数制限（念のため）
            """

        full_prompt = f"""
        あなたは「データサイエンス・トップマネジメント講義」の学習をサポートするAIコーチです。
        ユーザーの学習状況や質問に対して、親切かつ的確にアドバイスしてください。
        資料が提供されている場合は、その内容に基づいて回答してください。
        
        【コンテキスト（学習状況など）】
        {context}
        
        {doc_context}
        
        【ユーザーの質問】
        {prompt}
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI応答エラー: {str(e)}"

def generate_similar_problem(subject, unit, difficulty="普通"):
    """類似問題を生成"""
    success, error = configure_genai()
    if not success:
        return f"エラー: {error}"

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""
        以下の単元のSPI（適性検査）またはデータサイエンスに関する練習問題を1問作成してください。
        
        科目: {subject}
        単元: {unit}
        難易度: {difficulty}
        
        出力形式:
        【問題】
        (問題文)
        
        【選択肢】
        A. ...
        B. ...
        C. ...
        D. ...
        
        【解答と解説】
        正解: ...
        解説: ...
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"問題生成エラー: {str(e)}"
