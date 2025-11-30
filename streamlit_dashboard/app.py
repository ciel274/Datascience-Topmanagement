import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import glob
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit_antd_components as sac
import google_calendar_utils
import app_translations as tr

def t(key):
    return tr.get_text(key, st.session_state.get("language", "日本語"))



# --- 安全な再実行トリガ（環境差分を吸収） ---
def trigger_rerun():
    """
    Streamlit の再実行を安全に呼び出す。
    """
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            try:
                qp = dict(st.query_params) if hasattr(st, "query_params") else {}
                qp["_rerun"] = int(time.time())
                st.query_params = qp
            except Exception:
                if hasattr(st, "set_query_params"):
                    st.set_query_params(**qp)
    except Exception:
        return

@st.cache_resource
def train_ai_models(df):
    """
    機械学習モデルの学習（キャッシュ化）
    Random Forestを用いて正答率を予測し、重要変数を抽出する
    """
    # データが少なすぎる場合は学習しない
    if df.empty or len(df) < 5:
        return None, None, None
    
    try:
        # データ前処理
        df_ml = df.copy()
        df_ml["date_obj"] = pd.to_datetime(df_ml["日付"])
        # 基準日からの経過日数
        min_date = df_ml["date_obj"].min()
        df_ml["days_passed"] = (df_ml["date_obj"] - min_date).dt.days
        # 正誤を数値化 (1/0)
        df_ml["is_correct"] = df_ml["正誤"].apply(lambda x: 1 if x == "〇" else 0)
        
        # 欠損値処理
        df_ml = df_ml.fillna(0)
        
        # カテゴリ変数のエンコーディング
        le_subj = LabelEncoder()
        le_unit = LabelEncoder()
        
        # 文字列型に変換してからエンコード
        df_ml["subj_code"] = le_subj.fit_transform(df_ml["科目"].astype(str))
        df_ml["unit_code"] = le_unit.fit_transform(df_ml["単元"].astype(str))
        
        # 特徴量: 経過日数, 科目, 単元, 解答時間, 学習投入時間
        # ※本来はOneHotEncodingすべきだが、決定木ベースなのでLabelEncodingでも許容
        X = df_ml[["days_passed", "subj_code", "unit_code", "解答時間(秒)", "学習投入時間(分)"]]
        y = df_ml["is_correct"]
        
        # モデル学習 (Random Forest Regressor)
        # 0/1の分類ではなく、確率(正答率)として予測したいので回帰モデルを使用
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # 変数重要度
        importances = pd.DataFrame({
            "feature": ["経過日数", "科目", "単元", "解答時間", "学習時間"],
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return model, importances, (le_subj, le_unit, min_date)
        
    except Exception as e:
        st.error(f"AI学習エラー: {e}")
        return None, None, None

def generate_weekly_study_plan(df, exam_date, target_rate, current_rate):
    """
    週間学習プラン自動生成 (エビングハウス忘却曲線 + 可用時間考慮)
    """
    if exam_date is None:
        return None
    
    # 残り日数計算
    today = datetime.today().date()
    days_left = (exam_date - today).days
    
    if days_left < 0:
        return None
    
    if df.empty:
        return None

    # 1. 復習候補の特定 (エビングハウス忘却曲線: 1, 3, 7, 14, 30日前)
    review_intervals = [1, 3, 7, 14, 30]
    review_candidates = {} # date -> set(units)
    
    # 過去の学習ログから復習すべき単元を特定
    df["date_obj"] = pd.to_datetime(df["日付"]).dt.date
    
    for day in range(min(7, days_left)):
        target_date = today + timedelta(days=day)
        review_units = set()
        
        # この日(target_date)に復習すべき過去の日付を計算
        for interval in review_intervals:
            past_date = target_date - timedelta(days=interval)
            # past_dateに学習した単元を取得
            studied_on_date = df[df["date_obj"] == past_date]["単元"].unique()
            for unit in studied_on_date:
                review_units.add(unit)
        
        review_candidates[target_date] = list(review_units)

    # 2. 弱点単元の抽出
    weak_units = df.groupby("単元").agg({
        "ミス": ["sum", "count"]
    }).reset_index()
    weak_units.columns = ["単元", "ミス数", "試行回数"]
    weak_units["正答率"] = (weak_units["試行回数"] - weak_units["ミス数"]) / weak_units["試行回数"]
    weak_units["優先度"] = (1 - weak_units["正答率"]) * weak_units["試行回数"]
    weak_list = weak_units.sort_values("優先度", ascending=False)["単元"].tolist()
    
    # 3. 週間プラン生成
    weekly_plan = {}
    daily_limit_mins = st.session_state.get("daily_study_time", 60)
    unit_time_mins = 20 # 1単元あたりの想定時間
    
    for day in range(min(7, days_left)):
        date = today + timedelta(days=day)
        date_str = date.strftime("%m/%d (%a)")
        
        todays_units = []
        current_time = 0
        
        # A. 復習単元を優先的に追加
        reviews = review_candidates.get(date, [])
        for unit in reviews:
            if current_time + unit_time_mins <= daily_limit_mins:
                todays_units.append({"name": unit, "type": "復習"})
                current_time += unit_time_mins
        
        # B. 時間が余っていれば弱点単元を追加
        weak_idx = 0
        while current_time + unit_time_mins <= daily_limit_mins and weak_idx < len(weak_list):
            unit = weak_list[weak_idx]
            # まだリストになければ追加
            if not any(u["name"] == unit for u in todays_units):
                todays_units.append({"name": unit, "type": "弱点"})
                current_time += unit_time_mins
            weak_idx += 1
            
        # C. それでも時間が余っていれば、ランダムまたは次の弱点を追加
        # (ここではシンプルに弱点リストをループ)
        while current_time + unit_time_mins <= daily_limit_mins:
             if weak_idx < len(weak_list):
                unit = weak_list[weak_idx]
                if not any(u["name"] == unit for u in todays_units):
                    todays_units.append({"name": unit, "type": "学習"})
                    current_time += unit_time_mins
                weak_idx += 1
             else:
                 break # 弱点リスト一巡したら終了
        
        if todays_units:
            weekly_plan[date_str] = {
                "units": todays_units,
                "time_minutes": current_time
            }
    
    return weekly_plan

def generate_ai_advice(current_rate, target_rate, time_excess_rate, streak_days):
    """
    ルールベースAIによる学習アドバイス生成
    """
    advices = []
    
    # 1. 正答率に基づくアドバイス
    if current_rate >= target_rate:
        advices.append(("<i class='bi bi-star-fill' style='color:#fbbf24;'></i>", "素晴らしい正答率です！この調子で難易度の高い問題にも挑戦してみましょう。"))
    elif current_rate >= target_rate - 0.1:
        advices.append(("<i class='bi bi-fire' style='color:#f97316;'></i>", "目標まであと少し！苦手な単元をピンポイントで復習すれば届きます。"))
    else:
        advices.append(("<i class='bi bi-lightbulb-fill' style='color:#f59e0b;'></i>", "まずは基礎固めから。正答率の低い単元を重点的に見直しましょう。"))
        
    # 2. 解答時間に基づくアドバイス
    if time_excess_rate > 0.3:
        advices.append(("<i class='bi bi-stopwatch' style='color:#6b7280;'></i>", "少し時間がかかっているようです。問題を解くスピードを意識してみましょう。"))
    elif time_excess_rate < 0.1:
        advices.append(("<i class='bi bi-lightning-charge-fill' style='color:#eab308;'></i>", "解答スピードは完璧です！ケアレスミスにだけ注意してください。"))
        
    # 3. 継続日数に基づくアドバイス
    if streak_days >= 3:
        advices.append(("<i class='bi bi-calendar-check-fill' style='color:#ef4444;'></i>", f"{streak_days}日連続学習中！習慣化の達人ですね。"))
    elif streak_days == 0:
        advices.append(("<i class='bi bi-megaphone-fill' style='color:#3b82f6;'></i>", "今日はまだ学習記録がありません。1問だけでも解いてみませんか？"))
        
    # ランダムに1つ、または状況に合わせて結合して返す
    # ここではメインのアドバイス（正答率）とサブアドバイスを組み合わせる
    main_icon, main_text = advices[0]
    
    if len(advices) > 1:
        sub_icon, sub_text = advices[1] if len(advices) > 1 else ("", "")
        return f"**AIコーチ**: {main_icon} {main_text}  \n{sub_icon} {sub_text}"
    else:
        return f"**AIコーチ**: {main_icon} {main_text}"

def generate_calendar_heatmap(df, year, month, exam_date=None, weekly_plan=None):
    """
    学習カレンダーヒートマップを生成（強化版）
    - 単月表示
    - 未来の学習予定表示
    - 試験日ハイライト
    """
    try:
        from datetime import datetime, timedelta
        import calendar as cal
        import pandas as pd # pandas import added for df_copy = pd.DataFrame()
        
        # 日付列を確実にdatetime型に変換
        df_copy = df.copy() if not df.empty else pd.DataFrame()
        if not df_copy.empty:
            df_copy["日付"] = pd.to_datetime(df_copy["日付"], errors='coerce')
            df_copy = df_copy.dropna(subset=["日付"])
            df_copy["日付"] = df_copy["日付"].dt.date
        
        # 日別に集計
        daily_stats_dict = {}
        if not df_copy.empty:
            daily_stats = df_copy.groupby(
                "日付"
            ).agg({
                "問題ID": "count",
                "正誤": lambda x: (x == "〇").mean(),
                "学習投入時間(分)": "sum"
            }).reset_index()
            
            daily_stats.columns = ["日付", "問題数", "正答率", "学習時間"]
            daily_stats_dict = daily_stats.set_index("日付").to_dict('index')
        
        # 週間プランから未来の予定を取得
        future_plan_dict = {}
        if weekly_plan:
            for day_key, units in weekly_plan.items():
                try:
                    # day_keyがすでにdatetime.dateオブジェクトの場合
                    if isinstance(day_key, datetime.date):
                        date_obj = day_key
                    else:
                        # 文字列の場合（"01/23 (Mon)"形式）
                        month_day_str = day_key.split(' ')[0]
                        current_year = datetime.now().year
                        date_obj = datetime.strptime(f"{current_year}/{month_day_str}", "%Y/%m/%d").date()
                    
                    # unitsが辞書でunitsキーを持つ場合
                    if isinstance(units, dict) and "units" in units:
                        future_plan_dict[date_obj] = len(units["units"])
                    # unitsがリストの場合
                    elif isinstance(units, list):
                        future_plan_dict[date_obj] = len(units)
                    # その他の場合は単に存在フラグとして1を設定
                    elif units:
                        future_plan_dict[date_obj] = 1
                except Exception as e:
                    # エラーは無視して次へ
                    pass
        
        # CSSを定義
        css = """
        <style>
        .calendar-single {
            background: white;
            border-radius: 12px;
            padding: 8px 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            max-width: 100%;
            margin: 0 auto;
            font-family: "Source Sans Pro", sans-serif;
        }
        .calendar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .calendar-title {
            font-size: 1.3rem;
            font-weight: 800;
            color: #1f2937;
        }
        .calendar-nav {
            display: flex;
            gap: 8px;
        }
        .calendar-nav-btn {
            background: #f3f4f6;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: 600;
            color: #374151;
            transition: all 0.2s;
        }
        .calendar-nav-btn:hover {
            background: #e5e7eb;
        }
        .calendar-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        .calendar-weekday {
            font-size: 0.85rem;
            font-weight: 700;
            color: #6b7280;
            text-align: center;
            padding: 12px 8px;
            border-bottom: 2px solid #e5e7eb;
        }
        .calendar-day {
            aspect-ratio: 1;
            text-align: center;
            vertical-align: middle;
            font-size: 0.9rem;
            cursor: pointer;
            position: relative;
            border: 1px solid #f3f4f6;
            padding: 4px;
            box-sizing: border-box;
        }
        .calendar-day-content {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            padding: 8px 4px;
            box-sizing: border-box;
            transition: all 0.2s;
        }
        .calendar-day-content:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .calendar-day-empty {
            background: #fafafa;
        }
        .calendar-day-number {
            font-weight: 600;
            color: #1f2937;
            font-size: 1rem;
            line-height: 1;
            margin-bottom: 4px;
        }
        .calendar-day-indicator {
            font-size: 0.75rem;
            margin-top: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
        }
        /* 過去の学習データ（緑系） */
        .study-level-0 { background: #f9fafb; }
        .study-level-1 { background: #d1fae5; }
        .study-level-2 { background: #6ee7b7; }
        .study-level-3 { background: #34d399; }
        .study-level-4 { background: #10b981; color: white; }
        
        /* 未来の予定（青系） */
        .future-plan { 
            background: #eff6ff; 
            box-shadow: inset 0 0 0 2px #3b82f6;
        }
        .future-no-plan { background: #f9fafb; }
        
        /* 試験日（赤系） */
        .exam-date { 
            background: linear-gradient(135deg, #fecaca 0%, #ef4444 100%);
            box-shadow: inset 0 0 0 3px #dc2626;
            color: white;
            font-weight: 900;
            position: relative;
            overflow: hidden;
        }
        .exam-badge {
            position: absolute;
            top: 0;
            right: 0;
            background: #dc2626;
            color: white;
            font-size: 0.55rem;
            padding: 1px 4px;
            border-bottom-left-radius: 4px;
            font-weight: 700;
            line-height: 1.2;
        }
        </style>
        """
        
        # HTMLカレンダーを生成
        month_cal = cal.monthcalendar(year, month)
        month_name = f"{year}年{month}月"
        today = datetime.today().date()
        
        html = f'''
        <div class="calendar-single">
            <table class="calendar-table">
                <tr>
        '''
        
        # 曜日ヘッダー
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        for wd in weekdays:
            html += f'<th class="calendar-weekday">{wd}</th>'
        html += "</tr>"
        
        # 各週
        for week in month_cal:
            html += "<tr>"
            for day in week:
                if day == 0:
                    # 空白セル
                    html += '<td class="calendar-day"><div class="calendar-day-content calendar-day-empty"></div></td>'
                else:
                    date = datetime(year, month, day).date()
                    
                    # 試験日かチェック
                    is_exam_date = (exam_date and date == exam_date)
                    
                    # 過去 vs 未来
                    is_past = date < today
                    is_today = date == today
                    is_future = date > today
                    
                    tooltip = ""
                    css_class = ""
                    indicator = ""
                    badge = "" # Initialize badge
                    
                    if is_exam_date:
                        # 試験日
                        css_class = "exam-date"
                        tooltip = f"{date.strftime('%Y年%m月%d日')}: 🎯試験日"
                        badge = '<span class="exam-badge">試験</span>'
                    elif is_past or is_today:
                        # 過去/今日 - 学習データを表示
                        if date in daily_stats_dict:
                            study_time = daily_stats_dict[date]["学習時間"]
                            problems = int(daily_stats_dict[date]["問題数"])
                            accuracy = daily_stats_dict[date]["正答率"] * 100
                            
                            # 色レベルを決定
                            if study_time == 0:
                                level = 0
                            elif study_time <= 30:
                                level = 1
                            elif study_time <= 60:
                                level = 2
                            elif study_time <= 90:
                                level = 3
                            else:
                                level = 4
                            
                            css_class = f"study-level-{level}"
                            tooltip = f"{date.strftime('%Y年%m月%d日')}: {problems}問, 正答率{accuracy:.0f}%, {int(study_time)}分"
                            # 絵文字をBootstrap Iconに変更
                            indicator = '<i class="bi bi-check-lg"></i>' if problems > 0 else ""
                        else:
                            css_class = "study-level-0"
                            tooltip = f"{date.strftime('%Y年%m月%d日')}: 学習なし"
                        badge = ""
                    else:
                        # 未来 - 週間プランを表示
                        # 日付をキーとして検索
                        plan_count = future_plan_dict.get(date, 0)
                        
                        if plan_count > 0:
                            css_class = "future-plan"
                            tooltip = f"{date.strftime('%Y年%m月%d日')}: 📝学習予定 {plan_count}単元"
                            # 絵文字をBootstrap Iconに変更
                            indicator = f'<i class="bi bi-pencil-fill" style="color:#3b82f6; font-size:0.7rem;"></i> <span style="color:#3b82f6;">{plan_count}</span>'
                        else:
                            css_class = "future-no-plan"
                            tooltip = f"{date.strftime('%Y年%m月%d日')}: 予定なし"
                        badge = ""
                    
                    html += f'''
                    <td class="calendar-day" title="{tooltip}">
                        <div class="calendar-day-content {css_class}">
                            {badge}
                            <span class="calendar-day-number">{day}</span>
                            <div class="calendar-day-indicator">{indicator}</div>
                        </div>
                    </td>
                    '''
            
            html += "</tr>"
        
        html += '''
            </table>
        </div>
        '''
        
        return css, html
        
    except Exception as e:
        import streamlit as st # streamlit import added for st.error
        st.error(f"カレンダーヒートマップの生成エラー: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def generate_detailed_insights(df, current_rate, target_rate, exam_date=None):
    """
    統計分析とパターン認識で、具体的かつ実用的なアドバイスを提供
    """
    if df.empty:
        return []
    
    insights = []
    
    # 1. 学習パターン分析（時間帯・曜日）
    if "日付" in df.columns:
        df["hour"] = pd.to_datetime(df["日付"]).dt.hour
        df["dayofweek"] = pd.to_datetime(df["日付"]).dt.dayofweek
        
        # 時間帯別正答率
        hourly_stats = df.groupby("hour")["ミス"].agg(["sum", "count"])
        hourly_stats["accuracy"] = (hourly_stats["count"] - hourly_stats["sum"]) / hourly_stats["count"]
        
        if len(hourly_stats) >= 2:
            best_hour = hourly_stats["accuracy"].idxmax()
            worst_hour = hourly_stats["accuracy"].idxmin()
            
            if hourly_stats.loc[best_hour, "accuracy"] - hourly_stats.loc[worst_hour, "accuracy"] > 0.15:
                time_label = "朝型" if best_hour < 12 else "午後型" if best_hour < 18 else "夜型"
                insights.append({
                    "category": "学習パターン",
                    "icon": "clock-history",
                    "priority": "high",
                    "message": f"あなたは**{time_label}学習者**です。{best_hour}時台の正答率が最も高いです（{hourly_stats.loc[best_hour, 'accuracy']:.1%}）。重要な学習はこの時間帯に集中させましょう。"
                })
    
    # 2. 弱点の具体的指摘
    unit_stats = df.groupby("単元")["ミス"].agg(["sum", "count"])
    unit_stats["accuracy"] = (unit_stats["count"] - unit_stats["sum"]) / unit_stats["count"]
    unit_stats = unit_stats[unit_stats["count"] >= 3]  # 3問以上のデータがある単元のみ
    
    if not unit_stats.empty:
        weak_units = unit_stats[unit_stats["accuracy"] < 0.5].sort_values("accuracy")
        
        if not weak_units.empty:
            worst_unit = weak_units.index[0]
            worst_accuracy = weak_units.iloc[0]["accuracy"]
            
            # 単元別の具体的アドバイス（ルールベース）
            unit_advice = {
                "推論": "前提→結論の論理構造を意識し、命題の真偽を慎重に判断しましょう",
                "数的推理": "公式の丸暗記より、問題のパターン認識を優先しましょう",
                "判断推理": "図やテーブルを必ず描き、視覚的に整理しましょう",
                "資料解釈": "計算ミスを減らすため、概数で当たりをつける習慣を",
                "英語": "文法より読解スピードを優先。1文1秒ペースを目標に"
            }
            
            advice = unit_advice.get(worst_unit, "基礎問題を繰り返し、パターンを体に染み込ませましょう")
            
            insights.append({
                "category": "弱点分析",
                "icon": "exclamation-triangle",
                "priority": "high",
                "message": f"**{worst_unit}**が最大の弱点です（正答率{worst_accuracy:.1%}）。{advice}"
            })
    
    # 3. ペース分析
    if exam_date:
        today = datetime.today().date()
        days_left = (exam_date - today).days
        
        if days_left > 0:
            gap = target_rate - current_rate
            required_daily_improvement = gap / days_left if days_left > 0 else 0
            
            if gap > 0.2 and days_left < 30:
                insights.append({
                    "category": "進捗管理",
                    "icon": "speedometer",
                    "priority": "urgent",
                    "message": f"⚠️ **要注意**: 残り{days_left}日で{gap:.1%}の改善が必要です。1日あたり{required_daily_improvement:.2%}のペースで向上が必要です。集中学習を推奨します。"
                })
            elif gap > 0 and days_left >= 30:
                insights.append({
                    "category": "進捗管理",
                    "icon": "graph-up",
                    "priority": "medium",
                    "message": f"残り{days_left}日で目標達成可能です。現在のペースを維持しながら、弱点補強を進めましょう。"
                })
            elif gap <= 0:
                insights.append({
                    "category": "進捗管理",
                    "icon": "trophy",
                    "priority": "low",
                    "message": "🎉 **目標達成済み**！現在の実力を維持しつつ、難易度の高い問題にチャレンジしましょう。"
                })
    
    # 4. 比較分析（直近1週間 vs 前週）
    if "日付" in df.columns and len(df) >= 10:
        df["date_obj"] = pd.to_datetime(df["日付"]).dt.date
        today = datetime.today().date()
        week_ago = today - timedelta(days=7)
        two_weeks_ago = today - timedelta(days=14)
        
        this_week = df[df["date_obj"] >= week_ago]
        last_week = df[(df["date_obj"] >= two_weeks_ago) & (df["date_obj"] < week_ago)]
        
        if not this_week.empty and not last_week.empty:
            this_week_rate = (this_week["正誤"] == "〇").sum() / len(this_week)
            last_week_rate = (last_week["正誤"] == "〇").sum() / len(last_week)
            improvement = this_week_rate - last_week_rate
            
            if improvement > 0.05:
                insights.append({
                    "category": "成長記録",
                    "icon": "arrow-up-circle",
                    "priority": "medium",
                    "message": f"📈 **素晴らしい成長**！先週比+{improvement:.1%}の改善です。この調子で継続しましょう。"
                })
            elif improvement < -0.05:
                insights.append({
                    "category": "成長記録",
                    "icon": "arrow-down-circle",
                    "priority": "medium",
                    "message": f"先週比-{abs(improvement):.1%}の低下が見られます。休息が必要かもしれません。無理せず、基礎の復習に戻りましょう。"
                })
    
    # 5. 時間管理分析
    if "解答時間(秒)" in df.columns and "目標時間" in df.columns:
        time_excess = (df["解答時間(秒)"] - df["目標時間"]).mean()
        
        if time_excess > 10:
            insights.append({
                "category": "時間管理",
                "icon": "hourglass-split",
                "priority": "medium",
                "message": f"平均{time_excess:.0f}秒超過しています。「速さより正確さ」から「スピード重視」にシフトする時期かもしれません。"
            })
        elif time_excess < -5:
            insights.append({
                "category": "時間管理",
                "icon": "lightning",
                "priority": "low",
                "message": "解答スピードは十分です。ケアレスミス防止のための見直し時間を確保しましょう。"
            })
    
    return insights

def generate_roadmap(exam_date, current_rate, target_rate):
    """
    試験日からの逆算ロードマップ（ガントチャート）生成
    """
    if exam_date is None:
        return None
    
    today = datetime.today().date()
    days_left = (exam_date - today).days
    
    if days_left <= 0:
        return None
        
    # フェーズ計算ロジック
    # 基礎固め: 全体の40% (進捗が遅れていれば50%に拡大)
    # 応用演習: 全体の40%
    # 直前対策: 全体の20%
    
    base_ratio = 0.4
    if current_rate < target_rate - 0.2: # 目標より20%以上低い場合
        base_ratio = 0.5 # 基礎期間を延長
        
    base_days = int(days_left * base_ratio)
    practice_days = int(days_left * (0.8 - base_ratio))
    final_days = days_left - base_days - practice_days
    
    # データフレーム作成
    data = [
        dict(Task="基礎固め期", Start=today, Finish=today + timedelta(days=base_days), Phase="Foundation"),
        dict(Task="応用演習期", Start=today + timedelta(days=base_days), Finish=today + timedelta(days=base_days + practice_days), Phase="Practice"),
        dict(Task="直前対策期", Start=today + timedelta(days=base_days + practice_days), Finish=exam_date, Phase="Final")
    ]
    
    df_gantt = pd.DataFrame(data)
    
    # Plotly Expressでガントチャート作成
    fig = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Phase",
                      color_discrete_map={"Foundation": "#60A5FA", "Practice": "#34D399", "Final": "#F87171"},
                      height=150) # 高さを抑える
    
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None, tickformat="%m/%d")
    
    # レイアウト調整
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#374151"),
        bargap=0.2
    )
    
    return fig


def generate_study_roadmap_detailed(df, df_master):
    """
    難易度別学習ロードマップの生成
    基礎(低)→標準(中)→応用(高)の順で習熟度を可視化し、次のステップを提案
    """
    if df.empty or df_master.empty:
        return None, None, None
    
    # 難易度列が存在するかチェック（なければデフォルト補完）
    if "難易度" not in df_master.columns:
        # st.warning("マスターデータに「難易度」列がありません。全て「中」として扱います。")
        df_master = df_master.copy()
        df_master["難易度"] = "中"
    
    try:
        # DFとマスタをマージして難易度情報を取得
        if "難易度" in df.columns:
            df_merged = df.copy()
        else:
            df_merged = df.merge(df_master[["問題ID", "難易度", "科目", "単元"]], on="問題ID", how="left")
        
        # 難易度列がNaNの行を除外
        df_merged = df_merged[df_merged["難易度"].notna()]
        
        if df_merged.empty:
            return None, None, None
        
        # 難易度別の統計を計算
        difficulty_stats = {}
        for diff in ["低", "中", "高"]:
            diff_data = df_merged[df_merged["難易度"] == diff]
            if not diff_data.empty:
                total = len(diff_data)
                correct = (diff_data["正誤"] == "〇").sum()
                accuracy = correct / total if total > 0 else 0
                
                # マスタデータから該当難易度の総問題数を取得
                master_diff = df_master[df_master["難易度"] == diff]
                total_problems_in_master = len(master_diff)
                coverage = (len(diff_data["問題ID"].unique()) / total_problems_in_master * 100) if total_problems_in_master > 0 else 0
                
                # その難易度の主な単元（問題数が多い順トップ5）
                top_units = master_diff["単元"].value_counts().head(5).index.tolist()
                
                difficulty_stats[diff] = {
                    "solved": len(diff_data["問題ID"].unique()),
                    "total": total_problems_in_master,
                    "accuracy": accuracy,
                    "coverage": coverage,
                    "attempts": total,
                    "units": top_units
                }
            else:
                # データがない場合
                master_diff = df_master[df_master["難易度"] == diff]
                total_problems_in_master = len(master_diff)
                top_units = master_diff["単元"].value_counts().head(5).index.tolist()
                
                difficulty_stats[diff] = {
                    "solved": 0,
                    "total": total_problems_in_master,
                    "accuracy": 0,
                    "coverage": 0,
                    "attempts": 0,
                    "units": top_units
                }
        
        # 現在のフェーズを判定
        current_phase = "基礎固め"
        next_recommendations = []
        
        # 基礎(低)が80%以上の正答率かつ70%以上のカバレッジなら標準へ
        if difficulty_stats["低"]["accuracy"] >= 0.8 and difficulty_stats["低"]["coverage"] >= 70:
            # 標準(中)が80%以上の正答率かつ70%以上のカバレッジなら応用へ
            if difficulty_stats["中"]["accuracy"] >= 0.8 and difficulty_stats["中"]["coverage"] >= 70:
                current_phase = "応用演習"
                next_recommendations = [
                    "応用問題を継続して解きましょう",
                    "高難易度問題の正答率向上を目指しましょう",
                    "解答時間の短縮にも意識を向けましょう"
                ]
            else:
                current_phase = "標準演習"
                # 未着手の標準問題を推薦
                unsolved_medium = df_master[
                    (df_master["難易度"] == "中") & 
                    (~df_master["問題ID"].isin(df["問題ID"].unique()))
                ]
                if not unsolved_medium.empty:
                    top_units = unsolved_medium["単元"].value_counts().head(3).index.tolist()
                    next_recommendations = [
                        f"次は「{top_units[0]}」に挑戦しましょう",
                        "標準問題の正答率80%を目指しましょう",
                        f"現在のカバー率: {difficulty_stats['中']['coverage']:.0f}%"
                    ]
                else:
                    next_recommendations = [
                        "標準問題をもう一度復習しましょう",
                        "正答率80%を安定させることが目標です"
                    ]
        else:
            current_phase = "基礎固め"
            # 未着手の基礎問題を推薦
            unsolved_low = df_master[
                (df_master["難易度"] == "低") & 
                (~df_master["問題ID"].isin(df["問題ID"].unique()))
            ]
            if not unsolved_low.empty:
                top_units = unsolved_low["単元"].value_counts().head(3).index.tolist()
                next_recommendations = [
                    f"まずは「{top_units[0]}」から始めましょう",
                    "基礎問題の正答率80%を目指しましょう",
                    f"現在のカバー率: {difficulty_stats['低']['coverage']:.0f}%"
                ]
            else:
                next_recommendations = [
                    "基礎問題を復習して定着度を上げましょう",
                    "正答率80%を安定させることが重要です"
                ]
        
        # ビジュアライゼーション用データ作成
        roadmap_data = {
            "phase": ["基礎固め", "標準演習", "応用演習"],
            "progress": [
                difficulty_stats["低"]["coverage"],
                difficulty_stats["中"]["coverage"],
                difficulty_stats["高"]["coverage"]
            ],
            "units": [
                difficulty_stats["低"]["units"],
                difficulty_stats["中"]["units"],
                difficulty_stats["高"]["units"]
            ],
            "accuracy": [
                difficulty_stats["低"]["accuracy"] * 100,
                difficulty_stats["中"]["accuracy"] * 100,
                difficulty_stats["高"]["accuracy"] * 100
            ],
            "status": [
                "完了" if difficulty_stats["低"]["accuracy"] >= 0.8 and difficulty_stats["低"]["coverage"] >= 70 else "進行中" if difficulty_stats["低"]["attempts"] > 0 else "未着手",
                "完了" if difficulty_stats["中"]["accuracy"] >= 0.8 and difficulty_stats["中"]["coverage"] >= 70 else "進行中" if difficulty_stats["中"]["attempts"] > 0 else "未着手",
                "完了" if difficulty_stats["高"]["accuracy"] >= 0.8 and difficulty_stats["高"]["coverage"] >= 70 else "進行中" if difficulty_stats["高"]["attempts"] > 0 else "未着手"
            ]
        }
        
        return roadmap_data, current_phase, next_recommendations
        
    except Exception as e:
        st.error(f"ロードマップ生成エラー: {e}")
        return None, None, None

def generate_sankey_diagram(df):
    """
    学習フローのSankey Diagram生成
    科目 → 単元 → 正誤結果 の3層フロー可視化
    """
    if df.empty or len(df) < 5:
        return None
    
    # データ集計: 科目 → 単元 → 正誤
    flow_data = df.groupby(["科目", "単元", "正誤"]).size().reset_index(name="count")
    
    # ノード定義
    subjects = df["科目"].unique().tolist()
    units = df["単元"].unique().tolist()
    results = ["正解", "不正解"]
    
    # ノードリスト作成（科目 → 単元 → 結果の順）
    node_labels = subjects + units + results
    node_colors = []
    
    # 科目の色（青系）
    subject_colors = ["#3B82F6", "#6366F1", "#8B5CF6"]
    for i in range(len(subjects)):
        node_colors.append(subject_colors[i % len(subject_colors)])
    
    # 単元の色（グレー系）
    for _ in units:
        node_colors.append("#9CA3AF")
    
    # 結果の色（正解=緑、不正解=赤）
    node_colors.append("#10B981")  # 正解
    node_colors.append("#EF4444")  # 不正解
    
    # リンク定義
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # 科目 → 単元 のリンク
    for subject in subjects:
        subject_idx = node_labels.index(subject)
        subject_data = df[df["科目"] == subject]
        
        for unit in subject_data["単元"].unique():
            unit_idx = node_labels.index(unit)
            count = len(subject_data[subject_data["単元"] == unit])
            
            sources.append(subject_idx)
            targets.append(unit_idx)
            values.append(count)
            link_colors.append("rgba(59, 130, 246, 0.3)")  # 薄い青
    
    # 単元 → 正誤 のリンク
    for unit in units:
        unit_idx = node_labels.index(unit)
        unit_data = df[df["単元"] == unit]
        
        # 正解数
        correct_count = len(unit_data[unit_data["正誤"] == "〇"])
        if correct_count > 0:
            sources.append(unit_idx)
            targets.append(node_labels.index("正解"))
            values.append(correct_count)
            link_colors.append("rgba(16, 185, 129, 0.4)")  # 薄い緑
        
        # 不正解数
        incorrect_count = len(unit_data[unit_data["正誤"] == "✕"])
        if incorrect_count > 0:
            sources.append(unit_idx)
            targets.append(node_labels.index("不正解"))
            values.append(incorrect_count)
            link_colors.append("rgba(239, 68, 68, 0.4)")  # 薄い赤
    
    # Sankey図作成
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=2),
            label=node_labels,
            color=node_colors,
            hovertemplate='%{label}: %{value}問<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}: %{value}問<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="学習フローの可視化",
            font=dict(size=18, color="#111827", weight="bold"),
            x=0.5,
            xanchor="center"
        ),
        font=dict(size=14, color="#000000", weight="bold"), # 文字色を完全な黒に、サイズアップ
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def generate_weekly_report(df):
    """
    週報レポート生成（過去7日間の学習サマリー）
    """
    if df.empty:
        return "データがありません。"
    
    today = datetime.today().date()
    week_ago = today - timedelta(days=7)
    
    # 過去7日間のデータ
    df["date_obj"] = pd.to_datetime(df["日付"]).dt.date
    df_week = df[df["date_obj"] >= week_ago].copy()
    
    if df_week.empty:
        return "過去7日間のデータがありません。"
    
    # 集計
    total_problems = len(df_week)
    total_time = df_week["学習投入時間(分)"].sum() if "学習投入時間(分)" in df_week.columns else 0
    accuracy = (1 - df_week["ミス"].mean()) * 100
    
    # 最も頑張った単元
    top_unit = df_week.groupby("単元").size().idxmax() if not df_week.empty else "N/A"
    top_count = df_week.groupby("単元").size().max() if not df_week.empty else 0
    
    # 継続日数
    study_days = df_week["date_obj"].nunique()
    
    report = f"""
### <i class="bi bi-bar-chart-fill"></i> **{st.session_state.current_user}さんの週報レポート**
期間: {week_ago.strftime('%Y/%m/%d')} 〜 {today.strftime('%Y/%m/%d')}

---

### <i class="bi bi-graph-up"></i> 今週の成果
- **学習日数**: {study_days}日
- **総演習問題数**: {total_problems}問
- **総学習時間**: {total_time:.0f}分 ({total_time/60:.1f}時間)
- **平均正答率**: {accuracy:.1f}%

### <i class="bi bi-trophy-fill"></i> 最重点単元
**{top_unit}** を {top_count}問 演習しました！

### <i class="bi bi-chat-quote-fill"></i> AIコーチからの総評
"""
    
    # 簡易的な総評ロジック
    if accuracy >= 80:
        report += "素晴らしい！この調子で継続しましょう。"
    elif accuracy >= 60:
        report += "着実に力をつけています。弱点を意識して復習を！"
    else:
        report += "基礎固めが必要です。焦らずコツコツ進めましょう。"
    
    report += f"\n\n### <i class='bi bi-bullseye'></i> 来週の目標\n正答率 **{min(100, accuracy + 5):.0f}%** を目指して、復習を強化しましょう！\n"
    
    return report

def predict_with_prophet(df, target_rate, exam_date):
    """
    Prophet時系列予測 - より精密な正答率予測
    トレンド + 季節性を考慮した予測を提供
    """
    try:
        from prophet import Prophet
    except ImportError:
        return None, "Prophetがインストールされていません"
    
    if df.empty or len(df) < 10:
        return None, "予測には最低10件のデータが必要です"
    
    if exam_date is None:
        return None, "試験日が設定されていません"
    
    # データ準備
    df_prophet = df.copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["日付"])
    
    # 日別正答率を計算
    daily_accuracy = df_prophet.groupby("ds").apply(
        lambda x: (x["正誤"] == "〇").sum() / len(x)
    ).reset_index()
    daily_accuracy.columns = ["ds", "y"]
    
    if len(daily_accuracy) < 2:
        return None, "予測には最低2日分のデータが必要です"
    
    # Prophetモデル構築
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True if len(daily_accuracy) >= 7 else False,
        yearly_seasonality=False,
        interval_width=0.8  # 80%信頼区間
    )
    
    model.fit(daily_accuracy)
    
    # 未来予測（試験日まで）
    future_dates = model.make_future_dataframe(periods=(exam_date - datetime.today().date()).days)
    forecast = model.predict(future_dates)
    
    # 試験日の予測値
    exam_datetime = pd.Timestamp(exam_date)
    exam_prediction = forecast[forecast["ds"] == exam_datetime]
    
    if exam_prediction.empty:
        # 試験日がデータ範囲外の場合、最も近い日付を使用
        exam_prediction = forecast.iloc[-1]
        predicted_rate = exam_prediction["yhat"]
    else:
        predicted_rate = exam_prediction["yhat"].values[0]
    
    # 予測値をグラフ用に整形
    forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_display.columns = ["日付", "予測正答率", "下限", "上限"]
    
    return {
        "forecast": forecast_display,
        "predicted_rate": predicted_rate,
        "model": model,
        "actual_data": daily_accuracy
    }, None

def generate_pdf_report(report_text, user_name):
    """
    週報レポートをPDF化
    """
    try:
        from fpdf import FPDF
        import io
        
        class PDF(FPDF):
            def header(self):
                # ヘッダー
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'SPI Learning Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                # フッター
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        
        # レポート本文（Markdown記号を削除）
        clean_text = report_text.replace("**", "").replace("###", "").replace("##", "").replace("*", "")
        
        # 行ごとに分割して追加
        for line in clean_text.split("\n"):
            if line.strip():
                # 日本語を含む場合はエンコーディング処理
                try:
                    # Latin-1でエンコード可能な文字のみ使用
                    safe_line = line.encode('latin-1', 'ignore').decode('latin-1')
                    pdf.multi_cell(0, 5, safe_line)
                except:
                    # エンコードエラーの場合はスキップ
                    pdf.multi_cell(0, 5, "[Japanese text]")
            else:
                pdf.ln(2)
        
        # バイナリデータとして返す
        pdf_output = io.BytesIO()
        pdf_data = pdf.output(dest='S').encode('latin-1')
        pdf_output.write(pdf_data)
        pdf_output.seek(0)
        
        return pdf_output
        
    except ImportError:
        return None

def generate_excel_report(df, user_name):
    """
    学習ログと統計をExcel形式で出力
    """
    try:
        import io
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        wb = Workbook()
        
        # シート1: 生データ
        ws1 = wb.active
        ws1.title = "学習ログ"
        
        # ヘッダースタイル
        header_fill = PatternFill(start_color="3B82F6", end_color="3B82F6", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        # データフレームをExcelに書き込み
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                cell = ws1.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 1:  # ヘッダー行
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
        
        # 列幅自動調整
        for column in ws1.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws1.column_dimensions[column_letter].width = adjusted_width
        
        # シート2: 統計サマリー
        ws2 = wb.create_sheet("統計サマリー")
        
        if not df.empty:
            # 基本統計
            total_problems = len(df)
            correct_count = (df["正誤"] == "〇").sum()
            accuracy = correct_count / total_problems
            
            stats_data = [
                ["指標", "値"],
                ["総問題数", total_problems],
                ["正解数", correct_count],
                ["正答率", f"{accuracy:.1%}"],
                ["平均解答時間", f"{df['解答時間(秒)'].mean():.1f}秒"],
                ["総学習時間", f"{df['学習投入時間(分)'].sum():.0f}分"]
            ]
            
            for r_idx, row in enumerate(stats_data, 1):
                for c_idx, value in enumerate(row, 1):
                    cell = ws2.cell(row=r_idx, column=c_idx, value=value)
                    if r_idx == 1:
                        cell.fill = header_fill
                        cell.font = header_font
            
            ws2.column_dimensions['A'].width = 20
            ws2.column_dimensions['B'].width = 15
        
        # バイナリデータとして返す
        excel_output = io.BytesIO()
        wb.save(excel_output)
        excel_output.seek(0)
        
        return excel_output
        
    except ImportError:
        return None

# ページ設定
st.set_page_config(
    page_title="SPI対策 Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Bootstrap Icons & Custom CSS ---
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
/* アイコンバッジ */
.icon-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background-color: #eff6ff; /* 薄い青 */
    color: #3b82f6; /* 青 */
    margin-right: 10px;
    font-size: 1.1rem;
    flex-shrink: 0;
}
/* チャートタイトル用ラッパー */
.chart-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    font-weight: 700;
    font-size: 1.1rem;
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# ===== テーマ定義 & カラー設定 =====
if "theme" not in st.session_state:
    st.session_state.theme = "Blue"

THEMES = {
    "Blue": {
        "PRIMARY": "#3B82F6", "ACCENT": "#F97316", "SUCCESS": "#10B981", 
        "WARNING": "#F59E0B", "DANGER": "#EF4444", "NEUTRAL": "#6B7280", "BACKGROUND": "#F8FAFC"
    },
    "Green": {
        "PRIMARY": "#059669", "ACCENT": "#D97706", "SUCCESS": "#3B82F6", 
        "WARNING": "#F59E0B", "DANGER": "#EF4444", "NEUTRAL": "#6B7280", "BACKGROUND": "#F0FDF4"
    },
    "Orange": {
        "PRIMARY": "#EA580C", "ACCENT": "#0284C7", "SUCCESS": "#10B981", 
        "WARNING": "#F59E0B", "DANGER": "#EF4444", "NEUTRAL": "#6B7280", "BACKGROUND": "#FFF7ED"
    },
    "Dark": {
        "PRIMARY": "#60A5FA", "ACCENT": "#FB923C", "SUCCESS": "#34D399", 
        "WARNING": "#FBBF24", "DANGER": "#F87171", "NEUTRAL": "#9CA3AF", "BACKGROUND": "#0F172A"
    },
}

# テーマ取得（キーエラー対策）
current_theme = st.session_state.theme
if current_theme not in THEMES:
    current_theme = "Blue"
    st.session_state.theme = "Blue"

tm = THEMES[current_theme]
PRIMARY = tm["PRIMARY"]
ACCENT = tm["ACCENT"]
SUCCESS = tm["SUCCESS"]
WARNING = tm["WARNING"]
DANGER = tm["DANGER"]
BACKGROUND = tm["BACKGROUND"]
DANGER = tm["DANGER"]
NEUTRAL = tm["NEUTRAL"]
BACKGROUND = tm["BACKGROUND"]

# ===== 問題マスタ（全30問） =====
MASTER_COLUMNS = ["問題ID", "科目", "ジャンル", "単元", "目標解答時間(秒)", "目標正答率(%)", "難易度", "出題頻度(重み)"]
DEFAULT_MASTER_ROWS = [
    ["N-A01", "非言語", "推論", "集合の推論 (ベン図)", 120, 85, "高", 4],
    ["N-A02", "非言語", "推論", "論理的な推論 (真偽・順序)", 100, 80, "高", 5],
    ["N-A03", "非言語", "推論", "対戦・リーグ戦の推論", 150, 75, "高", 5],
    ["N-A04", "非言語", "推論", "命題・三段論法", 90, 90, "低", 3],
    ["N-A05", "非言語", "推論", "領域 (座標平面)", 180, 65, "高", 3],
    ["N-A06", "非言語", "推論", "物の流れ", 160, 70, "高", 4],
    ["N-B01", "非言語", "計算・文章題", "比と割合の計算", 60, 95, "低", 5],
    ["N-B02", "非言語", "計算・文章題", "濃度算", 100, 80, "中", 5],
    ["N-B03", "非言語", "計算・文章題", "割引・割増計算", 90, 85, "中", 3],
    ["N-B04", "非言語", "計算・文章題", "損益算", 110, 90, "中", 5],
    ["N-B05", "非言語", "計算・文章題", "仕事算", 90, 80, "中", 4],
    ["N-B06", "非言語", "計算・文章題", "速度算", 130, 75, "高", 5],
    ["N-B07", "非言語", "計算・文章題", "料金の割引", 100, 70, "中", 5],
    ["N-C01", "非言語", "確率・場合", "確率 (基礎)", 70, 70, "低", 5],
    ["N-C02", "非言語", "確率・場合", "場合の数", 120, 65, "高", 5],
    ["N-D01", "非言語", "図表の読み取り", "グラフ・表の計算", 150, 75, "高", 3],
    ["N-D02", "非言語", "図表の読み取り", "増加率の把握", 140, 80, "中", 4],
    ["N-D03", "非言語", "図表の読み取り", "複数情報の読み取り", 180, 70, "高", 3],
    ["N-E01", "非言語", "特殊算・その他", "植木算・年齢算", 90, 85, "低", 3],
    ["N-E02", "非言語", "特殊算・その他", "集合の計算", 100, 80, "中", 3],
    ["N-E03", "非言語", "特殊算・その他", "分割払い", 110, 75, "中", 3],
    ["N-E04", "非言語", "特殊算・その他", "不定方程式", 120, 60, "低", 1],
    ["L-A01", "言語", "語彙知識", "二語の関係", 15, 95, "低", 4],
    ["L-A02", "言語", "語彙知識", "熟語の成り立ち", 20, 90, "中", 4],
    ["L-A03", "言語", "語彙知識", "語句の定義", 25, 85, "中", 5],
    ["L-B01", "言語", "文法・表現", "語句の用法", 30, 80, "中", 4],
    ["L-B02", "言語", "文法・表現", "空欄補充", 40, 75, "中", 4],
    ["L-B03", "言語", "文法・表現", "文の並べ替え", 100, 70, "高", 5],
    ["L-C01", "言語", "文章読解", "長文読解", 480, 70, "高", 5],
    ["L-C02", "言語", "文章読解", "論理的読解", 180, 65, "高", 4],
]
df_master_default = pd.DataFrame(DEFAULT_MASTER_ROWS, columns=MASTER_COLUMNS)

# ===== セッション初期化 =====
if "df_log_manual" not in st.session_state:
    st.session_state.df_log_manual = pd.DataFrame(columns=["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"])
if "target_rate_user" not in st.session_state:
    st.session_state.target_rate_user = 80
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "time_policy" not in st.session_state:
    st.session_state.time_policy = "標準"
if "subj" not in st.session_state:
    st.session_state.subj = None
if "gen" not in st.session_state:
    st.session_state.gen = None
if "uni" not in st.session_state:
    st.session_state.uni = None
if "keep_input_open" not in st.session_state:
    st.session_state.keep_input_open = True
if "expander_open" not in st.session_state:
    st.session_state.expander_open = st.session_state.keep_input_open
if "exam_date" not in st.session_state:
    st.session_state.exam_date = None
if "language" not in st.session_state:
    st.session_state.language = "日本語"
if "current_user" not in st.session_state:
    st.session_state.current_user = "デフォルトユーザー"
if "user_data_dir" not in st.session_state:
    st.session_state.user_data_dir = "user_data"
if "daily_study_time" not in st.session_state:
    st.session_state.daily_study_time = 60
if "plan_completion" not in st.session_state:
    st.session_state.plan_completion = {}
if "df_notes" not in st.session_state:
    st.session_state.df_notes = pd.DataFrame(columns=["問題ID", "メモ", "登録日時"])
if "display_mode" not in st.session_state:
    st.session_state.display_mode = "システム設定"

# ===== 高品質CSS (Glassmorphism & Modern UI) =====

# ダークモード用スタイル定義
dark_css = """
    /* ルート変数定義 */
    /* ルート変数定義 (Streamlit変数の強制オーバーライド) */
    :root {
        /* カスタム変数 */
        --primary: #60a5fa;
        --accent: #fb923c;
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f87171;
        --neutral: #9ca3af;
        --background: #0f172a;
        --surface: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border-color: #334155;

        /* Streamlit標準変数のオーバーライド */
        --primary-color: #60a5fa !important;
        --background-color: #0f172a !important;
        --secondary-background-color: #1e293b !important;
        --text-color: #f1f5f9 !important;
        --font: "sans serif" !important;
    }

    /* アプリ全体の背景とテキスト */
    html, body, .stApp {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }

    /* サイドバーの背景 */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #1e293b !important;
    }

    /* ============================================
       ダークモード修正 (完結編・最終調整 V4)
       ============================================ */
    
    /* 0. ブラウザネイティブのダークモード有効化 */
    :root {
        color-scheme: dark;
    }
    
    /* 1. アプリ全体とサイドバー */
    .stApp {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }
    
    /* 2. 入力フォームの徹底修正（属性セレクタ使用） */
    .stApp input[type="text"],
    .stApp input[type="number"],
    .stApp input[type="date"],
    .stApp input[type="password"],
    .stApp input[type="email"],
    .stApp textarea,
    .stApp select {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
        caret-color: #f1f5f9 !important;
    }
    
    /* BaseWebコンテナの修正 */
    .stApp div[data-baseweb="input"],
    .stApp div[data-baseweb="base-input"],
    .stApp div[data-baseweb="textarea"],
    .stApp div[data-baseweb="select"] > div {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #f1f5f9 !important;
    }

    /* サイドバー入力の強制オーバーライド (base_cssの特異性に対抗) */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stDateInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="popover"] {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] [role="option"] {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    
    /* 3. ボタン（計測開始・停止など）の修正 */
    .stApp button {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    
    .stApp button:hover {
        border-color: #60a5fa !important;
        color: #60a5fa !important;
    }
    
    /* Primaryボタン（データ追加など）は赤色を維持 */
    .stApp button[kind="primary"] {
        background-color: #ef4444 !important;
        border-color: #ef4444 !important;
        color: white !important;
    }
    .stApp button[kind="primary"]:hover {
        background-color: #dc2626 !important;
    }
    
    /* 4. 数値入力のステップボタン（+/-）の修正 */
    .stApp [data-baseweb="spin-button-group"] {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    
    .stApp [data-baseweb="spin-button-group"] > div {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    
    /* 5. ファイルアップローダーの修正 */
    .stApp [data-testid="stFileUploaderDropzone"] {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #f1f5f9 !important;
    }
    
    .stApp [data-testid="stFileUploaderDropzone"] div,
    .stApp [data-testid="stFileUploaderDropzone"] span,
    .stApp [data-testid="stFileUploaderDropzone"] small {
        color: #f1f5f9 !important;
    }
    
    /* 6. Expander (details/summaryタグ) */
    .stApp details {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
    
    .stApp summary {
        background-color: transparent !important;
        color: #f1f5f9 !important;
    }
    
    .stApp summary:hover {
        color: #60a5fa !important;
    }
    
    .stApp [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
        color: #f1f5f9 !important;
    }
    
    /* 7. プレースホルダー */
    .stApp ::placeholder {
        color: #94a3b8 !important;
        opacity: 0.7 !important;
    }
    
    /* 8. タブバー (SAC/Ant Design 強制オーバーライド - ダークモード) */
    /* 重要: base_cssのライトモードスタイルを確実に上書きするため、transparentではなく実際の暗い色を指定 */
    .stApp .stTabs,
    .stApp .ant-tabs,
    .stApp .ant-tabs-top {
        background-color: transparent !important;
    }
    
    /* これらの要素はbase_cssで白(#ffffff)に設定されているため、暗い色で上書き */
    .stApp .ant-tabs-nav,
    .stApp .ant-tabs-nav-wrap,
    .stApp .ant-tabs-nav-list {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }

    /* タブのコンテナ */
    .stApp div[data-baseweb="tab-list"],
    .stApp .ant-tabs-nav-operations {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 4px !important;
        border-radius: 8px !important;
        gap: 4px !important;
    }
    
    /* 個別のタブボタン */
    .stApp button[data-baseweb="tab"],
    .stApp .ant-tabs-tab {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border-radius: 6px !important;
        border: none !important;
        margin: 0 !important;
    }
    
    /* アクティブなタブ */
    .stApp button[data-baseweb="tab"][aria-selected="true"],
    .stApp .ant-tabs-tab-active {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* タブボタン内のテキスト */
    .stApp .ant-tabs-tab-btn {
        color: inherit !important;
    }


    /* SAC Divider Fix (Ant Design Divider) - Nuclear (No .stApp) */
    html body .ant-divider,
    html body .ant-divider *,
    html body div[class*="ant-divider"],
    html body div[class*="ant-divider"] * {
        border-top-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    html body .ant-divider-inner-text,
    html body .ant-divider-inner-text *,
    html body div[class*="ant-divider-inner-text"] {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }

    /* タブバーの背景色を強力に上書き - Nuclear (No .stApp) */
    html body .ant-tabs-nav,
    html body .ant-tabs-nav *,
    html body div[class*="ant-tabs-nav"],
    html body div[class*="ant-tabs-nav"] * {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border-color: #334155 !important;
    }
    
    html body .ant-tabs-tab,
    html body div[class*="ant-tabs-tab"] {
        background-color: transparent !important;
    }
    
    html body .ant-tabs-tab-active,
    html body div[class*="ant-tabs-tab-active"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* 水平線 (hr) */
    .stApp hr {
        border-color: #334155 !important;
        opacity: 1 !important;
    }
    
    /* 9. アラート・カード類 */
    .stApp [data-testid="stAlert"] {
        background-color: rgba(30, 41, 59, 0.95) !important;
        border: 1px solid #3b82f6 !important;
        color: #f1f5f9 !important;
    }
    
    .metric-card, .action-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid #334155 !important;
    }
    
    .metric-value, .action-unit, .action-title, .metric-label {
        color: #f1f5f9 !important;
    }
    
    /* 10. ドロップダウンメニュー（ポータル） */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
    }
    
    li[role="option"] {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #334155 !important;
    }
    
    /* 11. テキストカラー強制 */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp p, .stApp label, .stApp span, .stApp div, .stApp li {
        color: #f1f5f9 !important;
    }
    
    /* 例外: Primaryボタンのテキスト */
    .stApp button[kind="primary"] span {
        color: white !important;
    }
    
    /* 例外: カレンダーの日付 */
    .stApp div[data-baseweb="calendar"] button {
        color: #f1f5f9 !important;
    }
    .stApp div[data-baseweb="calendar"] button:hover {
        background-color: #3b82f6 !important;
    }
"""

base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700;900&display=swap');

:root {
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --success: {SUCCESS};
  --warning: {WARNING};
  --danger: {DANGER};
  --neutral: {NEUTRAL};
  --background: {BACKGROUND};
}

* {
    font-family: 'Inter', 'Noto Sans JP', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background-color: var(--background);
    background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(249, 115, 22, 0.05) 0px, transparent 50%);
}

/* ヘッダー */
.header {
    position: sticky; top: 0; z-index: 20;
    padding: 16px 0;
    margin-bottom: 24px;
}
.title-wrap { display:flex; align-items:center; gap:16px; }
.logo { 
    width:48px; height:48px; border-radius:12px; 
    display:flex; align-items:center; justify-content:center; 
    background: linear-gradient(135deg, var(--primary), #1e40af);
    color:#fff; font-weight:800; font-size: 24px;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
}

/* アクションカード */
.action-card {
    background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 15px -3px rgba(249, 115, 22, 0.1);
    display:flex; gap:20px; align-items:flex-start;
    position: relative; overflow: hidden;
}
.action-card::before {
    content: ''; position: absolute; top: 0; right: 0; width: 100px; height: 100px;
    background: var(--accent); opacity: 0.05; border-radius: 0 0 0 100%;
}
.action-icon {
    width:64px; height:64px; border-radius:16px;
    background: var(--accent); color:white;
    display:flex; align-items:center; justify-content:center;
    font-size:28px; flex-shrink:0;
    box-shadow: 0 4px 6px -1px rgba(249, 115, 22, 0.3);
}
.action-content { flex:1; z-index: 1; }
.action-title { color: #1f2937; font-weight:800; font-size:1.1rem; margin:0; }
.priority-badge {
    background: var(--danger); color: white; padding: 4px 12px;
    border-radius: 999px; font-size: 0.75rem; font-weight: 700;
    margin-left: 12px; display: inline-block;
}
.action-unit {
    font-size: 1.8rem; font-weight: 900; color: #111827;
    margin: 12px 0 8px 0; letter-spacing: -0.02em;
}

/* メトリックカード (Glassmorphism) */
.kpi-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:20px; margin-top:24px; }
.metric-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-label { color: var(--neutral); font-size: 0.85rem; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 2.2rem; font-weight: 900; color: #0f172a; line-height: 1; }
.metric-sub { font-size: 0.8rem; color: var(--neutral); margin-top: 8px; font-weight: 500; }

/* チャートコンテナ */
.chart-container {
    margin-top: 24px;
}

/* バッジ */
.badge-container { display: flex; align-items: center; gap: 8px; }
.badge {
    background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
    border: 1px solid #f59e0b;
    color: #b45309;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 700;
    box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    display: inline-flex; align-items: center;
    white-space: nowrap;
}

/* チャートタイトル */
.chart-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 12px;
}


/* ============================================
   ライトモード用 SAC/Ant Design タブスタイル
   ============================================ */

/* SAC/Ant Designタブバーの明示的なライトモードスタイル */
.stApp .stTabs,
.stApp .ant-tabs,
.stApp .ant-tabs-top {
    background-color: transparent !important;
}

.stApp .ant-tabs-nav,
.stApp .ant-tabs-nav-wrap,
.stApp .ant-tabs-nav-list {
    background-color: #ffffff;
    background: #ffffff;
    border:1px solid #e5e7eb;
    border-radius: 8px;
    padding: 4px;
}

.stApp .ant-tabs-tab,
.stApp button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #6b7280 !important;
    border-radius: 6px !important;
}

.stApp .ant-tabs-tab-active,
.stApp button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #3b82f6 !important;
    color: white !important;
}

.stApp .ant-tabs-tab-btn {
    color: inherit !important;
}

/* レスポンシブ */
@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .container { padding: 1rem; }
  .header .container > div { flex-direction: column; align-items: flex-start; gap: 12px; }
  .badge-container { flex-wrap: wrap; }
}

/* ============================================
   サイドバーのフォーム要素 - 統一デザインシステム
   ============================================ */

/* サイドバー全体の背景 */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

/* 統一されたフォーム要素スタイル */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input,
[data-testid="stSidebar"] .stDateInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

/* ホバー状態 - 統一 */
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stTextInput > div > div > input:hover,
[data-testid="stSidebar"] .stNumberInput > div > div > input:hover,
[data-testid="stSidebar"] .stDateInput > div > div > input:hover,
[data-testid="stSidebar"] .stTextArea > div > div > textarea:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 1px 3px 0 rgba(59, 130, 246, 0.1) !important;
}

/* フォーカス状態 - 統一 */
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus,
[data-testid="stSidebar"] .stNumberInput > div > div > input:focus,
[data-testid="stSidebar"] .stDateInput > div > div > input:focus,
[data-testid="stSidebar"] .stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
}

/* ドロップダウンオプション - 統一 */
[data-testid="stSidebar"] [role="listbox"],
[data-testid="stSidebar"] [role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

[data-testid="stSidebar"] [role="option"]:hover {
    background-color: #eff6ff !important;
}

/* セレクトボックスの内部テキスト - 統一 */
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* ラベル - 統一されたコントラスト */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextArea label {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    margin-bottom: 6px !important;
    display: block !important;
}

/* キャプション - 統一 */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}

/* プレースホルダー - 統一 */
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ボタン - 統一 */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"] {
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* カスタムラベル - 統一 */
.input-label {
    color: #1e293b;
    font-weight: 600;
    margin-bottom: 6px;
    font-size: 0.875rem;
    display: block;
}
"""

# CSSの組み立て
mode = st.session_state.get("display_mode", "システム設定")
current_theme_name = st.session_state.get("theme", "Blue")
final_css = base_css

# テーマがDarkの場合、またはモードがダークモードの場合
if current_theme_name == "Dark" or mode == "ダークモード":
    final_css += f"\n{dark_css}\n"
# システム設定の場合は、テーマがDarkの場合のみダークモードCSSを適用（Blueなどはライトモード固定）
elif mode == "システム設定" and current_theme_name == "Dark":
    final_css += f"\n{dark_css}\n"

final_css += "</style>"
final_css = final_css.replace("{PRIMARY}", PRIMARY).replace("{ACCENT}", ACCENT).replace("{SUCCESS}", SUCCESS).replace("{WARNING}", WARNING).replace("{DANGER}", DANGER).replace("{BACKGROUND}", BACKGROUND)

st.markdown(final_css, unsafe_allow_html=True)

# ===== ユーザー別ファイルパス定義 =====
user_log_path = f"{st.session_state.user_data_dir}/{st.session_state.current_user}.csv"
user_notes_path = f"{st.session_state.user_data_dir}/{st.session_state.current_user}_notes.csv"

# ===== サイドバー =====
# ===== サイドバー =====
st.sidebar.markdown(f'<div class="chart-header"><i class="bi bi-sliders icon-badge"></i>{t("settings_title")}</div>', unsafe_allow_html=True)

# 1. 企業・目標設定
expanded_settings = not bool(st.session_state.company_name)
with st.sidebar.expander(t("company_goal_settings"), expanded=expanded_settings):
    company = st.text_input(t("target_company"), value=st.session_state.company_name, placeholder=t("target_company_placeholder"), key="comp_input")
    st.session_state.company_name = company
    
    target = st.slider(t("target_accuracy"), 0, 100, st.session_state.target_rate_user, 5, key="target_slider")
    st.session_state.target_rate_user = target
    
    study_time = st.slider(t("daily_study_time"), 10, 180, st.session_state.daily_study_time, 10, key="time_slider")
    st.session_state.daily_study_time = study_time
    
    time_policy = st.selectbox(t("time_policy"), ["標準", "厳しく(-10%)", "緩く(+10%)"], 
                             index=["標準", "厳しく(-10%)", "緩く(+10%)"].index(st.session_state.time_policy), format_func=t, key="time_select")
    st.session_state.time_policy = time_policy

    # 試験日設定（サイドバーに追加）
    st.markdown("---")
    st.caption(t("exam_date_caption"))
    current_exam_date = st.session_state.exam_date if st.session_state.exam_date else datetime.today()
    new_exam_date = st.date_input(t("exam_date"), value=current_exam_date, key="sidebar_exam_date")
    
    if new_exam_date != st.session_state.exam_date:
        st.session_state.exam_date = new_exam_date
        trigger_rerun()

time_factor = {"標準": 1.0, "厳しく(-10%)": 0.9, "緩く(+10%)": 1.1}[st.session_state.time_policy]

# マスタデータの準備（ファイル管理より先にデフォルト読み込み）
if "df_master" not in st.session_state:
    st.session_state.df_master = df_master_default.copy()

# 2. 学習データ入力
expanded_flag = st.session_state.get("expander_open", st.session_state.get("keep_input_open", True))
with st.sidebar.expander(t("input_data_title"), expanded=expanded_flag):
    st.markdown(f"<p class='input-label'>{t('date')}</p>", unsafe_allow_html=True)
    dt = st.date_input(t("date"), datetime.today(), label_visibility="collapsed", key="dt_input")
    
    # マスタデータ使用
    df_master_use = st.session_state.df_master
    
    subjs = sorted(df_master_use["科目"].unique().tolist())
    # セッションステートからインデックスを復元
    subj_idx = subjs.index(st.session_state.subj) if st.session_state.subj in subjs else 0
    sel_subj = st.selectbox(t("subject"), subjs, index=subj_idx, label_visibility="collapsed", key="s1")
    
    # 科目変更時のみリセット
    if st.session_state.subj != sel_subj:
        st.session_state.subj = sel_subj
        st.session_state.gen = None
        st.session_state.uni = None
        # trigger_rerun() # 即時反映のため（必要なら）
    
    gens = ["選択"] + sorted(df_master_use[df_master_use["科目"] == st.session_state.subj]["ジャンル"].unique().tolist())
    gen_idx = gens.index(st.session_state.gen) if st.session_state.gen in gens else 0
    sel_gen = st.selectbox(t("genre"), gens, index=gen_idx, label_visibility="collapsed", key="g1")
    
    if st.session_state.gen != sel_gen:
        st.session_state.gen = sel_gen
        st.session_state.uni = None
    
    if st.session_state.gen and st.session_state.gen != "選択":
        unis = sorted(df_master_use[(df_master_use["科目"] == st.session_state.subj) & 
                                (df_master_use["ジャンル"] == st.session_state.gen)]["単元"].unique().tolist())
    else:
        unis = []
    
    unis = [t("select")] + unis
    uni_idx = unis.index(st.session_state.uni) if st.session_state.uni in unis else 0
    sel_uni = st.selectbox(t("unit"), unis, index=uni_idx, label_visibility="collapsed", key="u1")
    
    if st.session_state.uni != sel_uni:
        st.session_state.uni = sel_uni
    
    ids = df_master_use[(df_master_use["科目"] == st.session_state.subj) & 
                    (df_master_use["ジャンル"] == st.session_state.gen) & 
                    (df_master_use["単元"] == st.session_state.uni)]["問題ID"].tolist() if (
                    st.session_state.uni and st.session_state.uni != t("select")) else []
    
    pid = ids[0] if ids else ""
    st.caption(f"{t('problem_id')}: **{pid or t('not_selected')}**")
    
    # タイマー機能
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        def start_timer():
            st.session_state.timer_start_time = time.time()
            st.toast(t("timer_toast_start"), icon="⏱️")

        st.button(t("timer_start"), use_container_width=True, on_click=start_timer)

    with col_t2:
        def stop_timer():
            if st.session_state.get("timer_start_time"):
                elapsed = int(time.time() - st.session_state.timer_start_time)
                st.session_state.timer_elapsed = elapsed
                st.session_state.timer_start_time = None
                st.toast(t("timer_toast_stop").format(elapsed), icon="✅")
            else:
                st.toast(t("timer_toast_warn"), icon="⚠️")

        st.button(t("timer_stop"), use_container_width=True, on_click=stop_timer)
    
    col1, col2 = st.columns(2)
    with col1:
        # タイマー結果があればそれをデフォルトに
        def_at = st.session_state.get("timer_elapsed", 60)
        at = st.number_input(t("answer_time"), min_value=0, max_value=600, value=def_at, step=5, key="at_input")
    with col2:
        cor = st.selectbox(t("result"), ["〇", "✕"], format_func=t, key="cor_select")
    
    cau = st.selectbox(t("miss_reason"), ["-", "理解不足", "知識不足", "時間不足", "ケアレス"], format_func=t, key="cau_select")
    stm = st.number_input(t("study_time_min"), min_value=0, max_value=180, value=10, step=5, key="stm_input")
    
    # 復習メモ欄
    memo = st.text_area(t("memo"), placeholder=t("memo_placeholder"), height=80, key="memo_input")
    
    def add_data_callback(current_pid):
        if not current_pid:
            st.toast(t("toast_error_id"), icon="⚠️")
            return
        
        # 入力値の取得
        input_dt = st.session_state.dt_input
        input_cor = st.session_state.cor_select
        input_at = st.session_state.at_input
        input_cau = st.session_state.cau_select
        input_stm = st.session_state.stm_input
        input_memo = st.session_state.memo_input
        
        new_entry = {
            "日付": input_dt.strftime("%Y-%m-%d"),
            "問題ID": current_pid,
            "正誤": input_cor,
            "解答時間(秒)": input_at,
            "ミスの原因": input_cau,
            "学習投入時間(分)": input_stm
        }
        
        # ログデータ（CSV）への保存
        # 既存のCSVがある場合は読み込んで追記、なければ新規作成
        if os.path.exists(user_log_path):
            try:
                df_current = pd.read_csv(user_log_path)
                df_new = pd.concat([df_current, pd.DataFrame([new_entry])], ignore_index=True)
                df_new.to_csv(user_log_path, index=False)
            except Exception as e:
                st.error(f"ログ保存エラー: {e}")
        else:
            # CSVがない場合は新規作成
            pd.DataFrame([new_entry]).to_csv(user_log_path, index=False)

        # セッションステートも更新（フォールバック用）
        st.session_state.df_log_manual = pd.concat(
            [st.session_state.df_log_manual, pd.DataFrame([new_entry])],
            ignore_index=True
        )
        
        # メモ保存
        if input_memo and input_memo.strip():
            note_entry = {
                "問題ID": current_pid,
                "メモ": input_memo.strip(),
                "登録日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.df_notes = pd.concat(
                [st.session_state.df_notes, pd.DataFrame([note_entry])],
                ignore_index=True
            )
            st.session_state.df_notes.to_csv(user_notes_path, index=False)
        
        # st.toast("データを追加しました", icon="✅")
        st.session_state.show_success_toast = True
        st.session_state.expander_open = True

    st.button(t("add_data_btn"), type="primary", use_container_width=True, key="add_btn", on_click=add_data_callback, args=(pid,))

# 3. ユーザー管理
with st.sidebar.expander(t("user_management"), expanded=False):
    # user_dataディレクトリ作成
    if not os.path.exists(st.session_state.user_data_dir):
        os.makedirs(st.session_state.user_data_dir)
    
    # 既存ユーザーの取得
    user_files = glob.glob(f"{st.session_state.user_data_dir}/*.csv")
    existing_users = [os.path.basename(f).replace(".csv", "") for f in user_files]
    
    # デフォルトユーザーが存在しない場合は追加
    if t("default_user") not in existing_users:
        existing_users.insert(0, t("default_user"))
    
    # ユーザー選択
    selected_user = st.selectbox(
        t("select_user"),
        options=[t("create_new_user")] + existing_users,
        index=(existing_users.index(st.session_state.current_user) + 1) 
              if st.session_state.current_user in existing_users else 1,
        format_func=lambda x: t("create_new") if x == t("create_new_user") else x
    )
    
    if selected_user == t("create_new_user"):
        new_user = st.text_input(t("new_user_name"), placeholder=t("new_user_placeholder"))
        if st.button(t("create_user_btn")) and new_user:
            if new_user not in existing_users:
                st.session_state.current_user = new_user
                # 空のCSVを作成
                empty_df = pd.DataFrame(columns=["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"])
                empty_df.to_csv(f"{st.session_state.user_data_dir}/{new_user}.csv", index=False)
                st.success(t("user_created").format(new_user))
                trigger_rerun()
            else:
                st.error(t("user_exists"))
    elif selected_user != st.session_state.current_user:
        st.session_state.current_user = selected_user
        trigger_rerun()
    
    st.markdown(f"**{t('current_user')}:** {st.session_state.current_user}")

# 4. ファイルアップロードセクション
st.sidebar.markdown(f'<div class="chart-header" style="font-size:0.9rem; margin-bottom:8px;"><i class="bi bi-folder icon-badge" style="width:24px; height:24px; font-size:0.9rem;"></i>{t("file_management")}</div>', unsafe_allow_html=True)
with st.sidebar.expander(t("file_details"), expanded=False):
    st.markdown(f"<p class='input-label'>{t('master_csv')}</p>", unsafe_allow_html=True)
    master_file = st.file_uploader(t("master_csv"), type=["csv"], key="master", label_visibility="collapsed")
    
    st.markdown(f"<p class='input-label'>{t('log_csv')}</p>", unsafe_allow_html=True)
    log_file = st.file_uploader(t("log_csv"), type=["csv"], key="log", label_visibility="collapsed")

# マスタ読み込み処理（ファイル管理の後で更新）
if master_file:
    try:
        st.session_state.df_master = pd.read_csv(master_file)
        with st.sidebar:
            sac.alert(t("master_loaded"), icon='check-circle', color='success', size='sm')
    except:
        with st.sidebar:
            sac.alert(t("master_failed"), icon='x-circle', color='error', size='sm')
else:
    # アップロードがない場合はデフォルト（初期化済み）
    pass

# ログデータの取得

if log_file:
    # アップロードされたファイルが前回と同じかチェック（ファイル名やサイズで簡易判定）
    file_id = f"{log_file.name}_{log_file.size}"
    if st.session_state.get("processed_log_file") != file_id:
        try:
            df_upload = pd.read_csv(log_file)
            
            # 既存データがある場合はマージする
            if os.path.exists(user_log_path):
                try:
                    df_current = pd.read_csv(user_log_path)
                    # 共通のカラムを持つ場合のみ連結、あるいは単純連結
                    df_log = pd.concat([df_current, df_upload], ignore_index=True)
                except:
                    df_log = df_upload
            else:
                df_log = df_upload
            
            # ユーザーのファイルに保存（マージ結果）
            df_log.to_csv(user_log_path, index=False)
            st.session_state.processed_log_file = file_id
            # マニュアル入力用DFも更新
            st.session_state.df_log_manual = df_log.copy()
            with st.sidebar:
                sac.alert(t("log_merged"), icon='check-circle', color='success', size='sm')
        except:
            df_log = st.session_state.df_log_manual.copy()
            with st.sidebar:
                sac.alert(t("log_failed"), icon='x-circle', color='error', size='sm')
    else:
        # 既に処理済みの場合は、保存されたファイル（最新の状態）を読み込む
        if os.path.exists(user_log_path):
            try:
                df_log = pd.read_csv(user_log_path)
                # マニュアル入力用DFも同期
                st.session_state.df_log_manual = df_log.copy()
            except:
                df_log = st.session_state.df_log_manual.copy()
        else:
             df_log = st.session_state.df_log_manual.copy()
else:
    # ユーザーのファイルが存在すれば読み込み
    if os.path.exists(user_log_path):
        try:
            df_log = pd.read_csv(user_log_path)
            # マニュアル入力用DFも同期
            st.session_state.df_log_manual = df_log.copy()
        except:
            df_log = st.session_state.df_log_manual.copy()
    else:
        df_log = st.session_state.df_log_manual.copy()

# ノートデータの取得（ユーザー別）
if os.path.exists(user_notes_path):
    try:
        st.session_state.df_notes = pd.read_csv(user_notes_path)
    except:
        st.session_state.df_notes = pd.DataFrame(columns=["問題ID", "メモ", "登録日時"])
else:
    # ファイルが存在しなければ空のDataFrame
    st.session_state.df_notes = pd.DataFrame(columns=["問題ID", "メモ", "登録日時"])

# マスタデータ変数をローカル変数にセット（後続処理用）
df_master = st.session_state.df_master

# ===== メインコンテンツ =====
# ===== メインコンテンツ =====
badges_html = ""
df = pd.DataFrame()
df_all = pd.DataFrame()

# 変数初期化 (AIコーチなどで使用するため)
att = 0
cor_r = 0.0
tgt_r = st.session_state.target_rate_user / 100
te = 0.0
streak = 0
prediction_text = t("data_insufficient")
prediction_color = "#6B7280"
prediction_sub = t("keep_studying")
bd = pd.DataFrame(columns=["日", "正答率", "ミス", "count", "sum"]) # 初期化

try:
    # データ処理
    df_log["日付"] = pd.to_datetime(df_log["日付"], errors="coerce")
    df_log["解答時間(秒)"] = pd.to_numeric(df_log["解答時間(秒)"], errors="coerce").fillna(0)
    df_log["学習投入時間(分)"] = pd.to_numeric(df_log["学習投入時間(分)"], errors="coerce").fillna(0)
    df_log["ミス"] = (df_log["正誤"] == "✕").astype(int)
    df = pd.merge(df_log, df_master, on="問題ID", how="left")
    df["目標時間"] = df["目標解答時間(秒)"] * time_factor
    
    # カレンダー用（全期間データ）
    df_all = df.copy()

    # 分析期間の選択
    st.sidebar.markdown(f'<div class="chart-header"><i class="bi bi-search icon-badge"></i>{t("analysis_period")}</div>', unsafe_allow_html=True)
    mind = df["日付"].min()
    maxd = df["日付"].max()
    defs = maxd - timedelta(days=7) if pd.notnull(maxd) else datetime.today() - timedelta(days=7)
    sd = st.sidebar.date_input(t("start_date"), defs if pd.notnull(defs) else datetime.today(), key="sd_input")
    ed = st.sidebar.date_input(t("end_date"), maxd if pd.notnull(maxd) else datetime.today(), key="ed_input")

    # 期間フィルタ
    if not df.empty:
        mask = (df["日付"].dt.date >= sd) & (df["日付"].dt.date <= ed)
        df = df.loc[mask]

    # KPI計算
    if not df.empty:
        att = len(df)
        cor_r = 1 - df["ミス"].mean()
        # tgt_r は初期化済み
        
        # 時間超過率
        df["時間超過"] = (df["解答時間(秒)"] > df["目標時間"]).astype(int)
        te = df["時間超過"].mean()
        
        # 集計
        agg = df.groupby("単元").agg({
            "ミス": ["sum", "count"],
            "解答時間(秒)": "mean",
            "目標時間": "mean"
        }).reset_index()
        agg.columns = ["単元", "ミス数", "試行回数", "平均解答時間", "目標時間"]
        agg["正答率"] = (agg["試行回数"] - agg["ミス数"]) / agg["試行回数"]
        
        # 優先度スコア (正答率が低い & 試行回数が多い & 時間がかかる)
        agg["優先度"] = (1 - agg["正答率"]) * 2 + (agg["平均解答時間"] / agg["目標時間"] - 1).clip(0, 1)
        
        # 科目ごとの正答率
        cr = df.groupby("科目")["ミス"].agg(["sum", "count"]).reset_index()
        cr["正答率"] = (cr["count"] - cr["sum"]) / cr["count"]
        
        # 単元ごとの正答率をマージ
        # agg = agg.merge(cr[["単元", "正答率"]], on="単元").sort_values("優先度", ascending=False) # 単元はaggにあるのでマージ不要、科目正答率をどうするか
        # 修正: 科目ごとの正答率は別途表示用。aggは単元別。
        agg = agg.sort_values("優先度", ascending=False)
        
        cs = df.groupby("科目")["ミス"].agg(["sum", "count"]).reset_index()
        cs["正答率"] = (cs["count"] - cs["sum"]) / cs["count"]
        
        bd = df.copy()
        bd["日"] = bd["日付"].dt.date
        bd = bd.groupby("日")["ミス"].agg(["sum", "count"]).reset_index()
        bd["正答率"] = (bd["count"] - bd["sum"]) / bd["count"]
    else:
        agg = pd.DataFrame()
        cs = pd.DataFrame()
        # bd は初期化済み

    # ===== バッジ判定ロジック =====
    badges = []

    # 1. 初心者 (10問以上)
    if att >= 10:
        badges.append(f"<i class='bi bi-egg-fill'></i> {t('beginner_badge')}")

    # 2. 継続日数 (Streak)
    if not df.empty:
        dates = sorted(df["日付"].dropna().dt.date.unique())
        if len(dates) > 0:
            # streak は初期化済み (0)
            streak = 1
            # 最新の日付が今日か昨日かを確認
            last_d = dates[-1]
            today_d = datetime.today().date()
            
            # もし最新データが昨日より前なら、継続は途切れている（ただし今日はまだやってないだけかもしれないので0にはしないが、連続記録としてはストップ）
            # ここではシンプルに「最新の連続記録」を計算
            
            for i in range(2, len(dates) + 1):
                if (last_d - dates[-i]).days == 1:
                    streak += 1
                    last_d = dates[-i]
                else:
                    break
            
            # 今日か昨日学習していれば継続中とみなす
            if (today_d - dates[-1]).days <= 1:
                badges.append(f"<i class='bi bi-fire'></i> {t('streak_badge').format(streak=streak)}")
            else:
                # 途切れている場合
                badges.append(f"<i class='bi bi-clock-history'></i> {t('last_streak_badge').format(streak=streak)}")

    # 3. 推論マスター (推論ジャンルの正答率80%以上 & 5問以上)
    if not df.empty:
        genre_stats = df.groupby("ジャンル")["ミス"].agg(["sum", "count"])
        genre_stats["acc"] = (genre_stats["count"] - genre_stats["sum"]) / genre_stats["count"]
        for g_name, row in genre_stats.iterrows():
            if row["count"] >= 5 and row["acc"] >= 0.8:
                badges.append(f"<i class='bi bi-trophy-fill'></i> {g_name}{t('master_suffix')}")

    # 4. スピードスター (平均解答時間が目標の80%以下 & 正答率80%以上)
    if att >= 10 and cor_r >= 0.8:
        avg_time = df["解答時間(秒)"].mean()
        avg_target = df["目標時間"].mean()
        if avg_target > 0 and avg_time <= avg_target * 0.8:
            badges.append(f"<i class='bi bi-lightning-fill'></i> {t('speedster_badge')}")

    # バッジHTML生成（最大3個まで）
    display_badges = badges[:3]  # 最初の3つのみ表示
    for b in display_badges:
        badges_html += f"<span class='badge'>{b}</span>"
    
    # 4つ以上ある場合は「+N」を表示
    if len(badges) > 3:
        remaining = len(badges) - 3
        badges_html += f"<span class='badge' style='background: #e5e7eb; color: #6b7280; border-color: #9ca3af;'>+{remaining}</span>"

except Exception as e:
    st.error(f"{t('data_processing_error')}: {e}")


# ===== ヘッダー (Data Loaded) =====
title_text = t("app_title")
company_val = st.session_state.get('company_name', '')
if not company_val:
    company_val = t("target_company") if st.session_state.language == "English" else t('target_company') # Fallback or just use t()
    company_val = t("target_company") # Simplified
target_lbl = t("goal_label")
policy_val = st.session_state.get('time_policy',t('standard'))

# カウントダウン
countdown_html = ""
if st.session_state.exam_date:
    days_left = (pd.to_datetime(st.session_state.exam_date) - pd.to_datetime(datetime.today().date())).days
    if days_left >= 0:
        lbl = t("days_left")
        unit = t("days_unit")
        bg_col = "#ef4444" if days_left <= 7 else "#3b82f6"
        countdown_html = f"<div style='background:{bg_col}; color:white; padding:2px 10px; border-radius:6px; font-weight:bold; font-size:0.8rem; display:flex; align-items:center; gap:4px;'><span>{lbl}</span><span style='font-size:1rem;'>{days_left}</span><span>{unit}</span></div>"

st.markdown(
    f"<div class='header'><div class='container'>"
    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
    f"<div class='title-wrap'>"
    f"<div class='logo'><i class='bi bi-journal-text'></i></div>"
    f"<div>"
    f"<div style='display:flex; align-items:center; gap:12px;'>"
    f"<h1 style='color:#1e293b; margin:0; font-size:1.8rem; font-weight:800;'>{title_text}</h1>"
    f"{countdown_html}"
    f"</div>"
    f"<p style='color:#64748b; margin:4px 0 0 0; font-weight:500;'>{company_val} | {target_lbl} {st.session_state.get('target_rate_user',80)}% | {policy_val}</p>"
    f"</div></div>"
    f"<div class='badge-container'>{badges_html}</div>"
    f"</div></div></div>",
    unsafe_allow_html=True
)

if not df.empty:
    cau = df[df["ミス"] == 1]["ミスの原因"].value_counts().reset_index()
    cau.columns = [t("cause"), t("count")]

    # --- 3. 合格ライン到達予測 (Linear Regression) ---
    prediction_text = t("data_insufficient")
    prediction_sub = t("min_3_days_data")
    prediction_color = NEUTRAL

    if len(bd) >= 3:
        x = np.arange(len(bd))
        y = bd["正答率"].values
        if np.std(y) == 0:
            prediction_text = t("no_change")
            prediction_sub = t("accuracy_constant")
        else:
            z = np.polyfit(x, y, 1)
            slope = z[0]
            
            if cor_r >= tgt_r:
                prediction_text = t("achieved_exclamation")
                prediction_sub = t("goal_cleared")
                prediction_color = SUCCESS
            elif slope <= 0.001: # ほぼ横ばいか減少
                prediction_text = t("no_improvement")
                prediction_sub = t("review_study_method")
                prediction_color = DANGER
            else:
                intercept = z[1]
                days_needed = (tgt_r - intercept) / slope
                current_day = len(bd) - 1
                days_remaining = days_needed - current_day
                
                if days_remaining <= 0:
                     prediction_text = t("close_to_achieving")
                     prediction_sub = t("almost_there")
                     prediction_color = SUCCESS
                elif days_remaining > 365:
                    prediction_text = t("over_1_year")
                    prediction_sub = t("speed_up_needed")
                    prediction_color = WARNING
                else:
                    pred_date = datetime.today() + timedelta(days=int(days_remaining))
                    prediction_text = pred_date.strftime("%Y/%m/%d")
                    prediction_sub = t("predicted_achievement_date")
                    prediction_color = PRIMARY
    
else:
    cau = pd.DataFrame(columns=[t("cause"), t("count")])
    prediction_text = t("data_insufficient")
    prediction_sub = t("no_data")
    prediction_color = NEUTRAL

# ===== アクション & メニュー (2カラムレイアウト) =====
if not df.empty:
    ac1, ac2 = st.columns(2)

    with ac1:
        tu = agg.iloc[0] if not agg.empty else None
        if tu is not None:
            top_unit_accuracy = tu["正答率"]
            tc = cau.iloc[0][t("cause")] if not cau.empty else t("unknown")
            rsn = f"{t('accuracy_rate')}{top_unit_accuracy:.0%}。" + (t("time_shortage_issue") if te > 0.3 else f"「{tc}」{t('main_cause_review_field')}")
            
            st.markdown(f"""
            <div class="action-card" style="height: 100%;">
              <div class="action-icon"><i class="bi bi-lightning-charge-fill"></i></div>
              <div class="action-content">
                <div class="action-header">
                  <div class="action-title">{t('next_week_focus_unit')}</div>
                  <div class="priority-badge">{t('highest_priority')}</div>
                </div>
                <div class="action-unit">{tu['単元']}</div>
                <div class="action-reason">{rsn}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with ac2:
        # 1. 本日の学習メニュー提案
        st.markdown(f"""
        <div class="action-card" style="height: 100%; border-color: {PRIMARY}; background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);">
          <div class="action-icon" style="background: {PRIMARY}; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);"><i class="bi bi-calendar-event-fill"></i></div>
          <div class="action-content">
            <div class="action-header">
              <div class="action-title">{t('todays_study_menu')}</div>
              <div class="priority-badge" style="background: {PRIMARY};">{t('recommended')}</div>
            </div>
            <div style="margin-top: 12px;">
        """, unsafe_allow_html=True)
        
        if not agg.empty:
            top_3 = agg.head(3)
            for i, row in top_3.iterrows():
                # 優先度に応じて問題数を提案 (例: 優先度1.0 -> 3問, 0.5 -> 2問)
                q_count = max(1, min(5, int(row["優先度"] * 4)))
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px dashed #e5e7eb; padding-bottom:4px;">
                    <span style="font-weight:700; color:#374151;">{i+1}. {row['単元']}</span>
                    <span style="font-weight:800; color:{PRIMARY};">{q_count}{t('questions_unit')}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div>{t('cannot_propose_no_data')}</div>", unsafe_allow_html=True)
            
        st.markdown("""
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ===== KPIカード =====
st.markdown("<div class='container'>", unsafe_allow_html=True)

# モダンなタブナビゲーション (SAC) - テーマに応じて色を変更
current_theme_name = st.session_state.get("theme", "Blue")
is_dark_mode = (current_theme_name == "Dark") or (st.session_state.get("display_mode") == "ダークモード")

# SACタブの背景色を強制的に変更するCSS
if is_dark_mode:
    sac_tab_css = """
    <style>
    /* SAC タブのダークモード強制スタイル - Nuclear (No .stApp dependency) */
    html body div[class*="ant-tabs-nav"],
    html body div[class*="ant-tabs-nav"] * {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border-color: #334155 !important;
    }
    html body div[class*="ant-tabs-tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
    }
    html body div[class*="ant-tabs-tab-active"],
    html body div[class*="ant-tabs-tab-active"] * {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* SAC Divider Fix - Nuclear (No .stApp dependency) */
    html body div[class*="ant-divider"],
    html body div[class*="ant-divider"] * {
        border-top-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    html body div[class*="ant-divider-inner-text"],
    html body div[class*="ant-divider-inner-text"] * {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }
    </style>
    """
    st.markdown(sac_tab_css, unsafe_allow_html=True)

tab_selection = sac.tabs([
    sac.TabsItem(label=t("tab_dashboard"), icon='bar-chart-fill'),
    sac.TabsItem(label=t("tab_data_list"), icon='table'),
    sac.TabsItem(label=t("tab_ai_analysis"), icon='robot'),
    sac.TabsItem(label=t("tab_review_notes"), icon='journal-bookmark-fill'),
    sac.TabsItem(label=t("tab_settings"), icon='gear-fill'),
], align='center', size='lg', color='blue')


if tab_selection == t("tab_dashboard"):
    if df_all.empty:
        sac.alert(t("sidebar_input_prompt"), icon='info-circle', color='info')
    else:
        # st.markdown("### 📊 主要指標") # Removed
        
        # AIコーチ
        advice_text = generate_ai_advice(cor_r, tgt_r, te, streak)
        sac.alert(advice_text, icon='lightbulb', color='info', size='sm')

        # KPI ストリップ (Unified Design)
        st.markdown("""
        <style>
        .stats-strip {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 16px;
            padding: 20px 0;
            display: flex;
            align-items: center;
            justify-content: space-evenly;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-top: 16px;
            margin-bottom: 24px;
        }
        .stat-item {
            flex: 1;
            text-align: center;
            border-right: 1px solid rgba(0,0,0,0.06);
            padding: 0 12px; /* 少し詰める */
        }
        .stat-item:last-child { border-right: none; }
        .stat-label { 
            color: var(--neutral); font-size: 0.8rem; font-weight: 600; 
            margin-bottom: 4px; letter-spacing: 0.03em;
        }
        .stat-value { 
            font-size: 1.8rem; font-weight: 900; line-height: 1.1; 
            margin-bottom: 2px;
        }
        .stat-sub { 
            font-size: 0.7rem; color: var(--neutral); font-weight: 500; 
        }
        </style>
        """, unsafe_allow_html=True)

        # 値の計算
        col_cor = SUCCESS if cor_r >= tgt_r else DANGER
        gap = cor_r - tgt_r
        col_gap = SUCCESS if gap >= 0 else DANGER
        col_time = DANGER if te > 0.3 else SUCCESS

        st.markdown(f"""
        <div class="stats-strip">
            <div class="stat-item">
                <div class="stat-label">{t("current_accuracy")}</div>
                <div class="stat-value" style="color:{col_cor}">{cor_r:.0%}</div>
                <div class="stat-sub">{t("period_average")}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">{t("gap_to_goal")}</div>
                <div class="stat-value" style="color:{col_gap}">{gap:+.0%}</div>
                <div class="stat-sub">{t("achieved") if gap>=0 else t("not_achieved")}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">{t("forecast")}</div>
                <div class="stat-value" style="color:{prediction_color}; font-size: 1.6rem;">{prediction_text}</div>
                <div class="stat-sub">{prediction_sub}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">{t("time_excess_rate")}</div>
                <div class="stat-value" style="color:{col_time}">{te:.0%}</div>
                <div class="stat-sub">{t("over_target_time")}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">{t("total_exercises")}</div>
                <div class="stat-value" style="color:var(--primary)">{att}</div>
                <div class="stat-sub">{t("total_problems")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===== 学習カレンダーヒートマップ =====
        # ===== 学習カレンダー =====
        with st.expander(t("study_calendar"), expanded=True):
            # セッションステートで表示月を管理
            if "calendar_year" not in st.session_state:
                st.session_state.calendar_year = datetime.now().year
            if "calendar_month" not in st.session_state:
                st.session_state.calendar_month = datetime.now().month
            
            # 月間ナビゲーション
            c_nav1, c_nav2, c_nav3 = st.columns([1, 5, 1])
            with c_nav1:
                if st.button(t("prev_month"), key="prev_month"):
                    if st.session_state.calendar_month == 1:
                        st.session_state.calendar_month = 12
                        st.session_state.calendar_year -= 1
                    else:
                        st.session_state.calendar_month -= 1
                    trigger_rerun()
                    
            with c_nav3:
                if st.button(t("next_month"), key="next_month"):
                    if st.session_state.calendar_month == 12:
                        st.session_state.calendar_month = 1
                        st.session_state.calendar_year += 1
                    else:
                        st.session_state.calendar_month += 1
                    trigger_rerun()
            
            with c_nav2:
                st.markdown(f"<div style='text-align: center; font-size: 1.1rem; font-weight: 700; padding: 8px;'>{st.session_state.calendar_year}{t('year')}{st.session_state.calendar_month}{t('month')}</div>", unsafe_allow_html=True)
            
            # 週間プランからカレンダー用のデータを生成
            weekly_plan_for_calendar = {}
            if st.session_state.exam_date:
                # カレンダー表示月の日付範囲を計算
                # 簡易的にその月の1日から末日までを対象とするが、
                # generate_weekly_study_planは現在日からのプランを返すため、
                # そのまま渡してカレンダー側でフィルタリングする
                weekly_plan_data = generate_weekly_study_plan(
                    df_all, 
                    st.session_state.exam_date, 
                    tgt_r, 
                    cor_r
                )
                if weekly_plan_data:
                    weekly_plan_for_calendar = weekly_plan_data
            
            # カレンダー表示
            result = generate_calendar_heatmap(
                df_all,
                st.session_state.calendar_year,
                st.session_state.calendar_month,
                exam_date=st.session_state.exam_date,
                weekly_plan=weekly_plan_for_calendar
            )
        
            if result and result[0] and result[1]:
                css, html = result
                # CSSとHTMLを結合してcomponentsで表示
                full_html = css + html
                import streamlit.components.v1 as components
                components.html(full_html, height=400, scrolling=False)
                
                # カレンダー下の余白を減らす
                st.markdown("<div style='margin-top: -80px;'></div>", unsafe_allow_html=True)
                
                # カレンダー下に統計情報を表示
                col1, col2, col3 = st.columns(3)
                
                # 連続学習日数の計算
                if not df_all.empty:
                    df_with_date = df_all.copy()
                    df_with_date["日付"] = pd.to_datetime(df_with_date["日付"]).dt.date
                    unique_dates = sorted(df_with_date["日付"].unique(), reverse=True)
                    
                    current_streak = 0
                    max_streak = 0
                    temp_streak = 0
                    
                    if unique_dates:
                        # 現在の連続日数
                        today = datetime.today().date()
                        if unique_dates[0] == today or (len(unique_dates) > 1 and unique_dates[0] == today - timedelta(days=1)):
                            current_date = unique_dates[0]
                            current_streak = 1
                            for i in range(1, len(unique_dates)):
                                if unique_dates[i] == current_date - timedelta(days=1):
                                    current_streak += 1
                                    current_date = unique_dates[i]
                                else:
                                    break
                        
                        # 最長連続日数
                        for i in range(len(unique_dates)):
                            if i == 0:
                                temp_streak = 1
                            elif unique_dates[i-1] - unique_dates[i] == timedelta(days=1):
                                temp_streak += 1
                            else:
                                max_streak = max(max_streak, temp_streak)
                                temp_streak = 1
                        max_streak = max(max_streak, temp_streak)
                    
                    # 今月の統計
                    today = datetime.today()
                    this_month_data = df_with_date[
                        (pd.to_datetime(df_with_date["日付"]).dt.month == today.month) &
                        (pd.to_datetime(df_with_date["日付"]).dt.year == today.year)
                    ]
                    study_days_this_month = len(this_month_data["日付"].unique())
                else:
                    current_streak = 0
                    max_streak = 0
                    study_days_this_month = 0
                
                # 統計情報をカスタムスタイルで表示
                st.markdown("""
                <style>
                .calendar-stat {
                    text-align: center;
                    padding: 12px;
                    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                    border-radius: 8px;
                    border: 1px solid #e5e7eb;
                }
                .calendar-stat-icon {
                    font-size: 1.5rem;
                    color: #667eea;
                    margin-bottom: 4px;
                }
                .calendar-stat-value {
                    font-size: 1.8rem;
                    font-weight: 800;
                    color: #1f2937;
                    margin: 4px 0;
                }
                .calendar-stat-label {
                    font-size: 0.8rem;
                    color: #6b7280;
                    font-weight: 600;
                }
                </style>
                """, unsafe_allow_html=True)
                
                col1_html = f"""
                <div class="calendar-stat">
                    <i class="bi bi-fire calendar-stat-icon"></i>
                    <div class="calendar-stat-value">{current_streak}{t('days_unit')}</div>
                    <div class="calendar-stat-label">{t('current_streak_study')}</div>
                </div>
                """
                
                col2_html = f"""
                <div class="calendar-stat">
                    <i class="bi bi-calendar-check calendar-stat-icon"></i>
                    <div class="calendar-stat-value">{study_days_this_month}{t('days_unit')}</div>
                    <div class="calendar-stat-label">{t('study_days_this_month')}</div>
                </div>
                """
                
                col3_html = f"""
                <div class="calendar-stat">
                    <i class="bi bi-trophy calendar-stat-icon"></i>
                    <div class="calendar-stat-value">{max_streak}{t('days_unit')}</div>
                    <div class="calendar-stat-label">{t('longest_streak_record')}</div>
                </div>
                """
                
                with col1:
                    st.markdown(col1_html, unsafe_allow_html=True)
                with col2:
                    st.markdown(col2_html, unsafe_allow_html=True)
                with col3:
                    st.markdown(col3_html, unsafe_allow_html=True)
            else:
                st.info(t("cannot_display_data"))

        # ===== 学習ロードマップ =====
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-header'><i class='bi bi-signpost-split icon-badge'></i>{t('study_roadmap')}</div>", unsafe_allow_html=True)
        
        roadmap_data, current_phase, recommendations = generate_study_roadmap_detailed(df, st.session_state.df_master)
        
        if roadmap_data and current_phase and recommendations:
            # 現在のフェーズを強調表示
            phase_colors = {
                t("basic_consolidation"): "#3B82F6",
                t("standard_practice"): "#8B5CF6",
                t("advanced_practice"): "#EC4899"
            }
            current_color = phase_colors.get(current_phase, "#6B7280")
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {current_color}15 0%, {current_color}05 100%);
                border-left: 4px solid {current_color};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            ">
                <div style="font-size: 0.9rem; color: #64748b; font-weight: 600;">{t('current_phase')}</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {current_color}; margin-top: 4px;">
                    {current_phase}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 進捗バーを3つ表示
            col1, col2, col3 = st.columns(3)
            
            for idx, (col, phase) in enumerate([(col1, t("basic_consolidation")), (col2, t("standard_practice")), (col3, t("advanced_practice"))]):
                with col:
                    progress = roadmap_data["progress"][idx]
                    accuracy = roadmap_data["accuracy"][idx]
                    status = roadmap_data["status"][idx]
                    
                    # ステータスに応じた色とアイコン
                    if status == t("completed"):
                        status_color = "#10B981"
                        status_icon = '<i class="bi bi-check-circle-fill" style="color:#10B981;"></i>'
                        status_text_color = "#10B981"
                    elif status == t("in_progress"):
                        status_color = "#F59E0B"
                        status_icon = '<i class="bi bi-arrow-repeat" style="color:#F59E0B;"></i>'
                        status_text_color = "#F59E0B"
                    else:
                        status_color = "#9CA3AF"
                        status_icon = '<i class="bi bi-pause-circle" style="color:#9CA3AF;"></i>'
                        status_text_color = "#9CA3AF"
                    
                    units_list = "<br>".join([f"・{u}" for u in roadmap_data["units"][idx]])
                    
                    st.markdown(f"""
                    <style>
                    .roadmap-card {{
                        position: relative;
                        background: white;
                        border: 1px solid {status_color}40;
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        height: 100%;
                        cursor: help;
                        transition: transform 0.2s;
                    }}
                    .roadmap-card:hover {{
                        transform: translateY(-2px);
                    }}
                    .roadmap-tooltip {{
                        visibility: hidden;
                        width: 220px;
                        background-color: #333;
                        color: #fff;
                        text-align: left;
                        border-radius: 6px;
                        padding: 10px;
                        position: absolute;
                        z-index: 1;
                        bottom: 110%;
                        left: 50%;
                        transform: translateX(-50%);
                        opacity: 0;
                        transition: opacity 0.3s;
                        font-size: 0.8rem;
                        line-height: 1.4;
                        pointer-events: none;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                    .roadmap-tooltip::after {{
                        content: "";
                        position: absolute;
                        top: 100%;
                        left: 50%;
                        margin-left: -5px;
                        border-width: 5px;
                        border-style: solid;
                        border-color: #333 transparent transparent transparent;
                    }}
                    .roadmap-card:hover .roadmap-tooltip {{
                        visibility: visible;
                        opacity: 1;
                    }}
                    </style>
                    
                    <div class="roadmap-card">
                        <div class="roadmap-tooltip">
                            <strong>{t('main_units')}</strong><br>
                            {units_list}
                        </div>
                        <div style="font-size: 1.5rem; margin-bottom: 8px;">{status_icon}</div>
                        <div style="font-weight: 700; font-size: 1rem; color: #1f2937; margin-bottom: 8px;">
                            {phase}
                        </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 12px;">
                            {t('coverage_rate')}: {progress:.0f}%
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 8px;">
                            {t('accuracy_rate')}: {accuracy:.0f}%
                        </div>
                        <div style="
                            background: #e5e7eb;
                            border-radius: 999px;
                            height: 6px;
                            overflow: hidden;
                            margin-top: 12px;
                        ">
                            <div style="
                                background: {status_color};
                                height: 100%;
                                width: {progress}%;
                                border-radius: 999px;
                                transition: width 0.3s ease;
                            "></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 次のステップ推薦
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 20px;
            ">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px; color:#1e293b; font-weight:700;">
                    <i class="bi bi-lightbulb-fill" style="color:#f59e0b;"></i> {t('next_steps')}
                </div>
                <ul style="margin:0; padding-left:20px; color:#475569;">
                    {''.join([f'<li style="margin-bottom:8px;">{rec}</li>' for rec in recommendations])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(t("roadmap_no_data"))

        # ===== 週間学習プラン =====
        if st.session_state.exam_date:
            weekly_plan = generate_weekly_study_plan(
                df_all, 
                st.session_state.exam_date, 
                tgt_r, 
                cor_r
            )
            
            # DEBUG: 原因調査用
            # st.write(f"DEBUG: Exam Date: {st.session_state.exam_date}")
            # st.write(f"DEBUG: DF All Empty: {df_all.empty}")
            # if not df_all.empty:
            #    st.write(f"DEBUG: DF All Len: {len(df_all)}")
            # st.write(f"DEBUG: Plan Result: {weekly_plan}")
            
            if weekly_plan:
                sac.divider(label=t('weekly_study_plan'), icon='calendar-week', align='center')
                
                # CSSでチェックボックスの余白を詰める & Expanderのスタイル調整
                st.markdown("""
                <style>
                .compact-checkbox {
                    margin-bottom: -10px !important;
                }
                .compact-checkbox label {
                    font-size: 0.8rem !important;
                    padding-top: 0px !important;
                    padding-bottom: 0px !important;
                    min-height: 0px !important;
                }
                div[data-testid="stExpander"] {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                    border: 1px solid #e5e7eb;
                }
                div[data-testid="stExpander"] details summary {
                    padding-top: 8px !important;
                    padding-bottom: 8px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # 今日の日付文字列を取得（比較用）
                today_str = datetime.today().strftime("%m/%d (%a)")
                
                # ページネーション設定
                DAYS_PER_PAGE = 3
                if "plan_page_idx" not in st.session_state:
                    st.session_state.plan_page_idx = 0
                
                plan_items = list(weekly_plan.items())
                total_days = len(plan_items)
                
                # インデックス範囲の調整 (ボタン処理前)
                start_idx = st.session_state.plan_page_idx * DAYS_PER_PAGE
                if start_idx >= total_days:
                    st.session_state.plan_page_idx = 0
                    start_idx = 0
                end_idx = min(start_idx + DAYS_PER_PAGE, total_days)

                # ナビゲーションボタン (上部)
                col_prev, col_info, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.session_state.plan_page_idx > 0:
                        if st.button(t("prev_schedule"), key="plan_prev"):
                            st.session_state.plan_page_idx -= 1
                            # 再計算
                            start_idx = st.session_state.plan_page_idx * DAYS_PER_PAGE
                            end_idx = min(start_idx + DAYS_PER_PAGE, total_days)
                
                with col_next:
                    if end_idx < total_days:
                        if st.button(t("next_schedule"), key="plan_next"):
                            st.session_state.plan_page_idx += 1
                            # 再計算
                            start_idx = st.session_state.plan_page_idx * DAYS_PER_PAGE
                            end_idx = min(start_idx + DAYS_PER_PAGE, total_days)
                
                # 表示用アイテム更新
                current_items = plan_items[start_idx:end_idx]
                
                # 表示
                if current_items:
                    plan_cols = st.columns(len(current_items))
                    for idx, (day_str, plan_data) in enumerate(current_items):
                        with plan_cols[idx]:
                                # 今日かどうかでExpanderの開閉を制御
                                is_today = (day_str == today_str)
                                
                                # Expanderのラベルを作成
                                label = f"**{day_str}**"
                                
                                with st.expander(label, expanded=is_today):
                                    st.markdown(f"""
                                    <div style="text-align:center; font-size:0.75rem; color:#6B7280; margin-bottom:8px; font-weight:600;">
                                        <i class="bi bi-clock"></i> {plan_data['time_minutes']}{t('minutes_unit')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # チェックボックス
                                    for unit_info in plan_data['units']:
                                        unit_name = unit_info["name"]
                                        unit_type = unit_info["type"]
                                        
                                        # ユニークキー: 日付_単元
                                        chk_key = f"plan_{day_str}_{unit_name}"
                                        
                                        # レイアウト: チェックボックス + カレンダー追加ボタン
                                        c_chk, c_btn = st.columns([1, 0.25])
                                        
                                        # 初期値
                                        is_done = st.session_state.plan_completion.get(chk_key, False)
                                        
                                        # ラベル装飾
                                        label_text = f"~~{unit_name}~~" if is_done else f"{unit_name}"
                                        
                                        with c_chk:
                                            # CSSクラス適用のためのコンテナ
                                            st.markdown('<div class="compact-checkbox">', unsafe_allow_html=True)
                                            
                                            def toggle_plan_status(k):
                                                st.session_state.plan_completion[k] = st.session_state[k]

                                            st.checkbox(
                                                f"{unit_type} {label_text}", 
                                                value=is_done, 
                                                key=chk_key,
                                                on_change=toggle_plan_status,
                                                args=(chk_key,)
                                            )
                                            
                                            st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        with c_btn:
                                            # カレンダー追加ポップオーバー
                                            try:
                                                with st.popover("", icon=":material/calendar_month:", help=t("add_to_google_calendar")):
                                                    st.markdown(f"**{unit_name}** {t('add_to_calendar')}")
                                                    sch_time = st.time_input(t("start_time"), value=datetime.strptime("20:00", "%H:%M").time(), key=f"time_{chk_key}")
                                                    sch_dur = st.number_input(t("study_duration_min"), value=30, step=10, key=f"dur_{chk_key}")
                                                    
                                                    if st.button(t("register"), key=f"btn_{chk_key}", type="primary"):
                                                        service, error = google_calendar_utils.get_calendar_service()
                                                        if error:
                                                            st.error(error)
                                                        else:
                                                            try:
                                                                current_year = datetime.now().year
                                                                month_day = day_str.split(' ')[0]
                                                                date_str = f"{current_year}/{month_day}"
                                                                date_obj = datetime.strptime(date_str, "%Y/%m/%d").date()
                                                                if date_obj < datetime.now().date() - timedelta(days=300):
                                                                    date_obj = date_obj.replace(year=current_year + 1)
                                                                
                                                                start_dt = datetime.combine(date_obj, sch_time)
                                                                end_dt = start_dt + timedelta(minutes=sch_dur)
                                                                
                                                                summary = f"📖 {t('study')}: {unit_name}"
                                                                description = f"{t('study_unit')}: {unit_name}\n{t('type')}: {unit_type}"
                                                                
                                                                link, err = google_calendar_utils.add_event_to_calendar(service, summary, start_dt, end_dt, description)
                                                                if link:
                                                                    st.success(t("registered_success"))
                                                                elif err:
                                                                    st.error(f"{t('error')}: {err}")
                                                            except Exception as e:
                                                                st.error(f"{t('error')}: {e}")
                                            except AttributeError:
                                                # st.popoverが使えない古いバージョンの場合（フォールバック）
                                                st.caption("📅")
                                        




        # ===== 逆算ロードマップ =====
        if st.session_state.exam_date:
            roadmap_fig = generate_roadmap(st.session_state.exam_date, cor_r, tgt_r)
            if roadmap_fig:
                sac.divider(label=t('roadmap_to_pass'), icon='map', align='center')
                st.plotly_chart(roadmap_fig, use_container_width=True, config={'displayModeBar': False})

        # ===== グラフ =====
        sac.divider(label=t('analysis_graphs'), icon='graph-up', align='center')
        
        m1, m2 = st.columns([2, 1])

        with m1:
            st.markdown(f'<div class="chart-header"><i class="bi bi-graph-up icon-badge"></i>{t("daily_accuracy_trend")}</div>', unsafe_allow_html=True)
            bd = bd.sort_values("日").reset_index(drop=True)
            bd["日_label"] = pd.to_datetime(bd["日"]).dt.day.astype(str) + t("day_suffix")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bd["日_label"],
                y=(bd["正答率"] * 100),
                mode='lines+markers',
                line=dict(color=PRIMARY, width=3),
                marker=dict(size=10, color=PRIMARY, line=dict(color='white', width=2)),
                name=t("accuracy_rate"),
                hovertemplate=f'<b>%{{x}}</b><br>{t("accuracy_rate")}：%{{y:.0f}}%<extra></extra>'
            ))
            last_rate = bd["正答率"].iloc[-1] if len(bd) > 0 else cor_r
            target_color = SUCCESS if last_rate >= tgt_r else DANGER
            target_y = tgt_r * 100
            fig.update_layout(shapes=[
                dict(type="line", xref="x", x0=bd["日_label"].iloc[0], x1=bd["日_label"].iloc[-1],
                     yref="y", y0=target_y, y1=target_y,
                     line=dict(color=target_color, width=2, dash="dash"))
            ])
            fig.update_layout(
                template='simple_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=30, b=60),
                xaxis=dict(showgrid=True, gridcolor='#E6EEF8', tickfont=dict(color='#374151'), zeroline=False),
                yaxis=dict(range=[0, 100], tickmode='array', tickvals=[0, 25, 50, 75, 100],
                           showgrid=True, gridcolor='#E6EEF8', gridwidth=1, zeroline=False),
                hovermode='x unified',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with m2:
            currentRate_pct = int(round(cor_r * 100))
            targetRate_pct = int(round(tgt_r * 100))
            circumference = 2 * np.pi * 45
            dash = (currentRate_pct / 100.0) * circumference
            remaining = circumference - dash
            svg = f"""
            <div class="metric-card" style="display:flex; align-items:center; justify-content:center;">
              <div class="flex flex-col items-center">
                <div class="relative" style="width:128px; height:128px;">
                  <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                    <circle cx="50" cy="50" r="45" fill="none" stroke="var(--border)" stroke-width="8" />
                    <circle cx="50" cy="50" r="45" fill="none" stroke="{SUCCESS if cor_r >= tgt_r else DANGER}" stroke-width="8"
                            stroke-dasharray="{dash:.2f} {remaining:.2f}" stroke-linecap="round" />
                  </svg>
                  <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                    <span style="font-size:1.5rem; font-weight:800; color:var(--card-foreground);">{currentRate_pct}%</span>
                    <span style="font-size:0.75rem; color:var(--muted-foreground);">/ {targetRate_pct}%</span>
                  </div>
                </div>
              </div>
            </div>
            """
            st.markdown(svg, unsafe_allow_html=True)

        # ===== 下部グラフ =====
        b1, b2 = st.columns(2)

        with b1:
            st.markdown(f'<div class="chart-header"><i class="bi bi-list-check icon-badge"></i>{t("top_5_priority_units")}</div>', unsafe_allow_html=True)
            t5 = agg.head(5).reset_index(drop=True)
            if not t5.empty:
                max_v = max(t5["優先度"].max(), 1.0)
                pad = max_v * 0.18
                x_max = max_v + pad
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=t5["単元"],
                    x=[x_max] * len(t5),
                    orientation='h',
                    marker=dict(color='rgba(234,239,243,0.95)'),
                    hoverinfo='none',
                    showlegend=False
                ))
                fig.add_trace(go.Bar(
                    y=t5["単元"],
                    x=t5["優先度"],
                    orientation='h',
                    marker=dict(color=PRIMARY, line=dict(color='rgba(0,0,0,0.06)', width=0)),
                    hovertemplate=f'%{{y}}<br>{t("priority")}：%{{x:.2f}}<extra></extra>',
                    name=t('priority')
                ))
                fig.update_layout(
                    template='simple_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    barmode='overlay',
                    height=360,
                    margin=dict(l=140, r=20, t=10, b=20),
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor='rgba(14,30,37,0.06)', range=[0, x_max], zeroline=False),
                    yaxis=dict(autorange='reversed', tickfont=dict(size=14, color='#374151'), dtick=1)
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with b2:
            st.markdown(f'<div class="chart-header"><i class="bi bi-pie-chart icon-badge"></i>{t("incorrect_answer_cause_analysis")}</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=cau[t("cause")],
                y=cau[t("count")],
                marker=dict(color=ACCENT, line=dict(color='rgba(0,0,0,0.06)', width=1)),
                hovertemplate=f'%{{x}}<br>{t("count")}：%{{y}}<extra></extra>'
            ))
            max_y = max(cau[t("count")].max() if not cau.empty else 1, 1)
            fig.update_layout(
                template='simple_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=360,
                margin=dict(l=20, r=20, t=10, b=40),
                showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(size=12, color='#374151')),
                yaxis=dict(showgrid=True, gridcolor='rgba(14,30,37,0.06)', zeroline=False,
                           tickmode='auto', range=[0, max_y * 1.12], tickfont=dict(size=12, color='#6B7280'))
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ===== AI時系列予測（Prophet） =====
        # Prophet予測（試験日が設定されている場合のみ）
        if st.session_state.get("exam_date") is not None and len(bd) >= 5:
            prophet_result, error_msg = predict_with_prophet(df, tgt_r, st.session_state.exam_date)
            
            if prophet_result:
                st.markdown("---")
                st.markdown(f'<div class="chart-header"><i class="bi bi-graph-up-arrow icon-badge"></i>{t("ai_time_series_prediction_prophet")}</div>', unsafe_allow_html=True)
                st.caption(t("prophet_desc"))
                
                col_p1, col_p2 = st.columns([1, 2])
                
                with col_p1:
                    predicted_rate = prophet_result["predicted_rate"]
                    # 0-1の範囲にクリップ
                    predicted_rate = max(0, min(1, predicted_rate))
                    
                    st.metric(
                        t("exam_day_predicted_accuracy"),
                        f"{predicted_rate:.1%}",
                        delta=f"{(predicted_rate - cor_r):.1%}"
                    )
                    
                    if predicted_rate >= tgt_r:
                        sac.alert(t("goal_achievement_likely"), icon='check-circle', color='success', size='sm')
                    else:
                        gap = tgt_r - predicted_rate
                        sac.alert(f"⚠️ {t('goal_shortage').format(gap=gap):.1%}", icon='exclamation-circle', color='warning', size='sm')
                
                with col_p2:
                    # 予測グラフ（実績 + 予測）
                    forecast_df = prophet_result["forecast"]
                    actual_df = prophet_result["actual_data"]
                    
                    fig_prophet = go.Figure()
                    
                    # 実績データ
                    fig_prophet.add_trace(go.Scatter(
                        x=actual_df["ds"],
                        y=actual_df["y"],
                        mode='markers',
                        name=t('actual_results'),
                        marker=dict(size=8, color=PRIMARY)
                    ))
                    
                    # 予測ライン
                    fig_prophet.add_trace(go.Scatter(
                        x=forecast_df["日付"],
                        y=forecast_df["予測正答率"],
                        mode='lines',
                        name=t('prediction'),
                        line=dict(color=ACCENT, width=2)
                    ))
                    
                    # 信頼区間
                    fig_prophet.add_trace(go.Scatter(
                        x=forecast_df["日付"],
                        y=forecast_df["上限"],
                        mode='lines',
                        name=t('upper_bound'),
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig_prophet.add_trace(go.Scatter(
                        x=forecast_df["日付"],
                        y=forecast_df["下限"],
                        mode='lines',
                        name=t('lower_bound'),
                        fill='tonexty',
                        fillcolor='rgba(249, 115, 22, 0.2)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    # 目標ライン
                    fig_prophet.add_hline(
                        y=tgt_r,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=t("goal")
                    )
                    
                    fig_prophet.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
                        xaxis_title=t("date"),
                        yaxis_title=t("accuracy_rate"),
                        legend=dict(orientation="h", yanchor="top", y=-0.2),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_prophet, use_container_width=True, config={'displayModeBar': False})
            elif error_msg:
                sac.alert(f"{t('prophet_prediction')}: {error_msg}", icon='info-circle', color='info', size='sm')

        # --- 詳細分析（ヒートマップ・散布図） ---
        sac.divider(label=t('detailed_analysis'), icon='search', align='center')
        
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            st.markdown(f'<div class="chart-header"><i class="bi bi-grid-3x3 icon-badge"></i>{t("accuracy_by_field")}</div>', unsafe_allow_html=True)
            heatmap_data = df.groupby(["科目", "ジャンル"])["ミス"].agg(["sum", "count"]).reset_index()
            heatmap_data["正答率"] = (heatmap_data["count"] - heatmap_data["sum"]) / heatmap_data["count"]
            heatmap_matrix = heatmap_data.pivot(index="ジャンル", columns="科目", values="正答率")
            
            fig_heat = px.imshow(
                heatmap_matrix,
                labels=dict(x=t("subject"), y=t("genre"), color=t("accuracy_rate")),
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                color_continuous_scale="RdBu", # Changed back to RdBu for visibility (Red=Low, Blue=High)
                zmin=0, zmax=1,
                aspect="auto",
                text_auto='.0%' # Show values
            )
            fig_heat.update_traces(xgap=3, ygap=3)
            fig_heat.update_layout(
                template='simple_white',
                height=320, 
                margin=dict(l=0,r=0,t=30,b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_colorbar=dict(title=t("accuracy_rate"), tickformat=".0%")
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with c_h2:
            st.markdown(f'<div class="chart-header"><i class="bi bi-crosshair icon-badge"></i>{t("weakness_analysis_4_quadrants")}</div>', unsafe_allow_html=True)
            unit_stats = df.groupby("単元").agg({
                "解答時間(秒)": "mean",
                "ミス": ["sum", "count"],
                "科目": "first"
            }).reset_index()
            unit_stats.columns = ["単元", "平均解答時間", "ミス数", "試行回数", "科目"]
            unit_stats["正答率"] = (unit_stats["試行回数"] - unit_stats["ミス数"]) / unit_stats["試行回数"]
            
            # 平均値を計算（象限の基準）
            avg_time = unit_stats["平均解答時間"].mean()
            avg_acc = unit_stats["正答率"].mean()
            max_time = unit_stats["平均解答時間"].max()
            
            fig_scatter = px.scatter(
                unit_stats,
                x="平均解答時間",
                y="正答率",
                size="試行回数",
                color="科目",
                hover_name="単元",
                color_discrete_sequence=[PRIMARY, ACCENT, SUCCESS],
                opacity=0.85
            )
            
            # 境界線
            fig_scatter.add_hline(y=avg_acc, line_dash="dash", line_color="#9ca3af")
            fig_scatter.add_vline(x=avg_time, line_dash="dash", line_color="#9ca3af")
            
            # 象限ラベル（アノテーション）
            # 左上 (速い・高い): 理想
            fig_scatter.add_annotation(x=avg_time*0.5, y=min(1.0, avg_acc + 0.1), text=t("ideal"), showarrow=False, font=dict(color=SUCCESS, size=11, weight="bold"))
            # 右上 (遅い・高い): 慎重/要反復
            fig_scatter.add_annotation(x=avg_time + (max_time-avg_time)*0.5, y=min(1.0, avg_acc + 0.1), text=t("needs_repetition"), showarrow=False, font=dict(color=WARNING, size=11, weight="bold"))
            # 左下 (速い・低い): ケアレスミス
            fig_scatter.add_annotation(x=avg_time*0.5, y=max(0.0, avg_acc - 0.1), text=t("careless_mistake"), showarrow=False, font=dict(color=ACCENT, size=11, weight="bold"))
            # 右下 (遅い・低い): 基礎不足
            fig_scatter.add_annotation(x=avg_time + (max_time-avg_time)*0.5, y=max(0.0, avg_acc - 0.1), text=t("needs_review"), showarrow=False, font=dict(color=DANGER, size=11, weight="bold"))
            
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color='white')))
            fig_scatter.update_layout(
                template='simple_white',
                height=320, 
                margin=dict(l=0,r=0,t=30,b=0), 
                yaxis=dict(range=[-0.05, 1.05], tickformat=".0%", title=t("accuracy_rate")),
                xaxis=dict(title=t("avg_answer_time_sec")),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # ===== 科目別達成状況 =====
        sac.divider(label=t('subject_achievement_status'), icon='stack', align='center')
        if cs.empty:
            st.info(t("no_subject_data"))
        else:
            cols_display = st.columns(len(cs))
            for i, row in enumerate(cs.itertuples()):
                subj_name = row.科目
                with cols_display[i]:
                    key_btn = f"subj_btn_{i}_{subj_name}"
                    clicked = st.button(subj_name, key=key_btn)
                    if clicked:
                        if st.session_state.get("selected_subject") == subj_name:
                            st.session_state.selected_subject = None
                        else:
                            st.session_state.selected_subject = subj_name

                    r = row.正答率
                    delta = r - tgt_r
                    if r >= 1.0:
                        value_col = PRIMARY
                    elif r >= tgt_r:
                        value_col = SUCCESS
                    else:
                        value_col = DANGER
                    delta_col = SUCCESS if delta > 0 else (DANGER if delta < 0 else "#000")
                    width = min(max(int(r * 100), 0), 100)
                    
                    html = f'''
                    <div style="text-align:center; margin-bottom:16px; cursor:pointer;">
                        <div style="font-size:0.9rem; color:#6B7280; margin-bottom:4px;">{subj_name}</div>
                        <div style="font-size:1.5rem; font-weight:700; color:{value_col}; line-height:1;">{r:.0%}</div>
                        <div style="font-size:0.75rem; color:{delta_col}; margin-bottom:8px;">{delta:+.0%}</div>
                        <div style="background-color:#E5E7EB; height:4px; border-radius:2px; width:100%; overflow:hidden;">
                            <div style="background-color:{value_col}; height:100%; width:{width}%;"></div>
                        </div>
                    </div>
                    '''
                    st.markdown(html, unsafe_allow_html=True)

            sel = st.session_state.get("selected_subject", None)
            if sel:
                sac.divider(label=f'<i class="bi bi-search"></i> {sel} {t("unit_accuracy_rate")}', icon='search', align='left')
                units = df[df["科目"] == sel].groupby("単元")["ミス"].agg(["sum", "count"]).reset_index()
                if units.empty:
                    st.info(t("no_data_for_subject"))
                else:
                    units["正答率"] = (units["count"] - units["sum"]) / units["count"]
                    units = units.sort_values("正答率", ascending=False).reset_index(drop=True)
                    st.dataframe(units[[t("unit"), t("accuracy_rate"), t("count")]].rename(columns={t("count"): t("attempts")}), use_container_width=True)

                    fig_units = go.Figure(go.Bar(
                        x=units["正答率"],
                        y=units["単元"],
                        orientation="h",
                        marker=dict(color=PRIMARY, line=dict(color='rgba(0,0,0,0.06)', width=1))
                    ))
                    fig_units.update_layout(
                        template="simple_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=max(200, 40 * len(units)),
                        margin=dict(l=0, r=10, t=10, b=10),
                        xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor="#E6EEF8"),
                        yaxis=dict(tickfont=dict(size=13, color="#111827"))
                    )
                    st.plotly_chart(fig_units, use_container_width=True, config={"displayModeBar": False})
                    if st.button(t("close"), key=f"close_subj_{sel}"):
                        st.session_state.selected_subject = None
                        trigger_rerun()
if tab_selection == t("tab_data_list"):
    st.markdown(f"### 📋 {t('tab_data_list')}")
    sac.divider(label=t('data_download'), icon='download', align='center')
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        csv_log = st.session_state.df_log_manual.to_csv(index=False).encode('utf-8-sig')
        st.download_button(t("learning_log_csv"), data=csv_log, file_name=f"study_log_{st.session_state.current_user}.csv", mime="text/csv", use_container_width=True)
        
    with col_dl2:
        if not st.session_state.df_notes.empty:
            csv_notes = st.session_state.df_notes.to_csv(index=False).encode('utf-8-sig')
            st.download_button(t("review_notes_csv"), data=csv_notes, file_name=f"review_notes_{st.session_state.current_user}.csv", mime="text/csv", use_container_width=True)
        else:
            st.button(t("review_notes_none"), disabled=True, use_container_width=True)
            
    with col_dl3:
        if not agg.empty:
            csv_agg = agg.to_csv(index=False).encode('utf-8-sig')
            st.download_button(t("unit_summary_csv"), data=csv_agg, file_name=f"unit_stats_{st.session_state.current_user}.csv", mime="text/csv", use_container_width=True)
        else:
            st.button(t("unit_summary_none"), disabled=True, use_container_width=True)
    
    with st.expander(t("entered_data_list"), expanded=True):
        if not st.session_state.df_log_manual.empty:
            # --- 新機能: データエディタで直接編集 ---
            sac.alert(t("edit_cell_instruction"), icon='pencil-square', color='info', size='sm')
            
            # 日付カラムを datetime に変換してエディタに渡す
            df_editor = st.session_state.df_log_manual.copy()
            if "日付" in df_editor.columns:
                df_editor["日付"] = pd.to_datetime(df_editor["日付"], errors="coerce")

            edited_df = st.data_editor(
                df_editor,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "日付": st.column_config.DateColumn(t("date"), format="YYYY-MM-DD"),
                    "正誤": st.column_config.SelectboxColumn(t("result"), options=["〇", "✕"]),
                    "ミスの原因": st.column_config.SelectboxColumn(t("miss_reason"), options=["-", "理解不足", "知識不足", "時間不足", "ケアレス"]),
                }
            )
            
            # 編集があった場合、日付を文字列に戻して保存
            if not edited_df.equals(df_editor):
                edited_df["日付"] = edited_df["日付"].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "")
                st.session_state.df_log_manual = edited_df
                st.success(t("changes_saved"))
                trigger_rerun()
            
            csv = st.session_state.df_log_manual.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=t("download_csv"),
                data=csv,
                file_name=f"spi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    st.markdown("---")
    uploaded = st.file_uploader(t("replace_data_csv"), type=["csv"], key="tab2_upload")
    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            required = ["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"]
            missing = [c for c in required if c not in df_new.columns]
            if missing:
                st.error(t("missing_csv_columns").format(columns=', '.join(missing)))
            else:
                st.session_state.df_log_manual = df_new[required].copy()
                st.success(t("session_data_replaced"))
                trigger_rerun()
        except Exception as e:
            st.error(t("csv_read_failed").format(error=e))
    
    st.markdown("---")
    with st.expander(t("dangerous_operations")):
        if st.button(t("delete_all_data"), type="primary"):
            st.session_state.df_log_manual = pd.DataFrame(columns=["日付", "問題ID", "正誤", "解答時間(秒)", "ミスの原因", "学習投入時間(分)"])
            st.success(t("all_logs_deleted"))
            trigger_rerun()
            
        else:
            st.info(t("no_input_data_prompt"))

if tab_selection == t("tab_ai_analysis"):
    sac.divider(label=t('ai_analysis_report'), icon='robot', align='center')
    st.write(t("ai_analysis_desc"))
    
    if df.empty or len(df) < 5:
        sac.alert(t("ai_analysis_min_data"), icon='exclamation-triangle', color='warning')
    else:
        # 詳細インサイト表示（Phase 2: ルールベースAI強化）
        st.markdown ("---")
        sac.divider(label=t('personalized_learning_analysis'), icon='person-check-fill', align='left')
        st.caption(t("personalized_learning_analysis_desc"))
        
        insights = generate_detailed_insights(df, cor_r, tgt_r, st.session_state.get("exam_date"))
        
        if insights:
            # 優先度別に色分け
            priority_colors = {
                "urgent": "error",
                "high": "warning",
                "medium": "info",
                "low": "success"
            }
            
            for insight in insights:
                icon = insight.get("icon", "info-circle")
                priority = insight.get("priority", "medium")
                color = priority_colors.get(priority, "info")
                message = insight.get("message", "")
                
                sac.alert(
                    message,
                    icon=icon,
                    color=color,
                    banner=True if priority == "urgent" else False,
                    closable=False
                )
        else:
            sac.alert(t("data_accumulation_needed"), icon='info-circle', color='info')
        
        st.markdown("---")
        
        # モデル学習
        with st.spinner(t("ai_model_training")):
            model_acc, importances, encoders = train_ai_models(df)
            
        if model_acc is None:
            st.error(t("model_training_failed"))
        else:
            le_subj, le_unit, min_date = encoders
            
            # 1. 未来予測
            sac.divider(label=t('accuracy_prediction_simulation'), icon='graph-up-arrow', align='left')
            col_ai1, col_ai2 = st.columns([1, 2])
            with col_ai1:
                target_date = st.date_input(t("prediction_date"), value=datetime.today() + timedelta(days=7))
                days_future = (pd.to_datetime(target_date) - min_date).days
                
                # 予測用ダミーデータ作成（平均的な学習条件で予測）
                avg_time = df["解答時間(秒)"].mean()
                avg_study = df["学習投入時間(分)"].mean()
                
                X_pred = []
                # 学習データに含まれるユニークな科目・単元のペアを取得
                unique_pairs = df[["科目", "単元"]].drop_duplicates()
                
                for _, row in unique_pairs.iterrows():
                    try:
                        s_c = le_subj.transform([row["科目"]])[0]
                        u_c = le_unit.transform([row["単元"]])[0]
                        X_pred.append([days_future, s_c, u_c, avg_time, avg_study])
                    except:
                        pass
                
                if X_pred:
                    pred_accs = model_acc.predict(X_pred)
                    final_pred = np.mean(pred_accs)
                    
                    st.metric(t("predicted_accuracy"), f"{final_pred:.1%}", delta=f"{(final_pred - cor_r):.1%}")
                    if final_pred >= tgt_r:
                        sac.alert(t("goal_achievement_likely"), icon='check-circle', color='success', size='sm')
                    else:
                        sac.alert(t("goal_not_achieved"), icon='exclamation-circle', color='warning', size='sm')
            
            with col_ai2:
                # 予測推移グラフ（向こう30日）
                future_days = range(days_future, days_future + 30)
                future_preds = []
                for d in future_days:
                    # 各日の予測（全単元平均）
                    X_d = [[d, x[1], x[2], x[3], x[4]] for x in X_pred]
                    preds = model_acc.predict(X_d)
                    future_preds.append(np.mean(preds))
                
                fig_pred = px.line(x=[min_date + timedelta(days=d) for d in future_days], y=future_preds, 
                                   labels={"x": t("date"), "y": t("predicted_accuracy")}, title=t("30_day_growth_prediction"))
                fig_pred.add_hline(y=tgt_r, line_dash="dash", line_color="red", annotation_text=t("goal"))
                fig_pred.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_pred, use_container_width=True)

            # 2. 要因分析
            sac.divider(label=t('performance_factor_analysis'), icon='bar-chart-steps', align='left')
            st.caption(t("performance_factor_analysis_desc"))
            fig_imp = px.bar(importances, x="importance", y="feature", orientation="h", 
                             title=t("impact_on_accuracy"), labels={"importance": t("importance"), "feature": t("factor")})
            fig_imp.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # 3. AIレコメンド
            sac.divider(label=t('recommended_curriculum'), icon='journal-check', align='left')
            st.caption(t("recommended_curriculum_desc"))
            
            # 全単元の現在の予測正答率を計算
            current_days = (datetime.today() - min_date).days
            recs = []
            for _, row in unique_pairs.iterrows():
                try:
                    s_c = le_subj.transform([row["科目"]])[0]
                    u_c = le_unit.transform([row["単元"]])[0]
                    # 今日の予測
                    p = model_acc.predict([[current_days, s_c, u_c, avg_time, avg_study]])[0]
                    recs.append({t("subject"): row["科目"], t("unit"): row["単元"], t("predicted_accuracy"): p})
                except:
                    pass
            
            df_recs = pd.DataFrame(recs)
            # 成長ゾーン (40% - 75%)
            df_growth = df_recs[(df_recs[t("predicted_accuracy")] >= 0.4) & (df_recs[t("predicted_accuracy")] <= 0.75)].sort_values(t("predicted_accuracy"))
            
            if not df_growth.empty:
                for i, row in df_growth.head(3).iterrows():
                    sac.alert(f"**{row[t('subject')]} - {row[t('unit')]}** ({t('predicted_accuracy')}: {row[t('predicted_accuracy')]:.1%})", icon='fire', color='info')
            else:
                sac.alert(t("no_growth_zone_units"), icon='check2-circle', color='success')
            
            # 4. 学習フロー可視化（Sankey Diagram）
            st.markdown("---")
            sac.divider(label=t('learning_flow_visualization'), icon='diagram-3', align='left')
            st.caption(t("learning_flow_visualization_desc"))
            
            sankey_fig = generate_sankey_diagram(df)
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True, config={'displayModeBar': False})
                
                # インサイト表示
                correct_rate = (df["正誤"] == "〇").sum() / len(df)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                            padding: 16px; border-radius: 12px; border-left: 4px solid {PRIMARY}; margin-top: 16px;">
                    <div style="font-weight: 600; color: #1f2937; margin-bottom: 8px;">
                        <i class="bi bi-lightbulb-fill" style="color: {PRIMARY}; margin-right: 8px;"></i>
                        {t('flow_analysis_insights')}
                    </div>
                    <div style="color: #374151; font-size: 0.9rem;">
                        • {t('overall_accuracy_rate')}: <strong>{correct_rate:.1%}</strong><br>
                        • {t('thick_flow_explanation')}<br>
                        • {t('green_red_flow_explanation')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sac.alert(t("data_insufficient_sankey"), icon='info-circle', color='info')

if tab_selection == t("tab_review_notes"):
    sac.divider(label=t('review_notes_title'), icon='journal-bookmark', align='center')
    st.write(t("review_notes_desc"))
    
    if st.session_state.df_notes.empty:
        sac.alert(t("no_notes_yet"), icon='info-circle', color='info')
    else:
        # 検索機能
        st.markdown(f'<div style="margin-bottom:8px; font-weight:600; color:#374151;"><i class="bi bi-search" style="margin-right:6px; color:#3b82f6;"></i>{t("search_problem_id_or_memo")}</div>', unsafe_allow_html=True)
        search_query = st.text_input(t("search"), placeholder=t("enter_keyword"), label_visibility="collapsed")
        
        # フィルタリング
        df_notes_display = st.session_state.df_notes.copy()
        if search_query:
            mask = (df_notes_display["問題ID"].astype(str).str.contains(search_query, case=False, na=False)) | \
                   (df_notes_display["メモ"].astype(str).str.contains(search_query, case=False, na=False))
            df_notes_display = df_notes_display[mask]
        
        st.markdown(f"**{t('total_notes').format(count=len(df_notes_display))}**")
        
        # 表示
        for idx, row in df_notes_display.iterrows():
            with st.expander(f"**{row['問題ID']}** - {row['登録日時']}", expanded=False):
                st.markdown(row['メモ'])
                
                # 削除ボタン
                def delete_note(idx_to_drop):
                    st.session_state.df_notes = st.session_state.df_notes.drop(idx_to_drop).reset_index(drop=True)
                    st.session_state.df_notes.to_csv(user_notes_path, index=False)
                    # sac.alertはrerunしないと消えないため、st.toastを使うか、rerunなしでUI更新を待つ
                    st.toast(t("deleted"), icon="✅")

                st.button(t("delete"), key=f"del_note_{idx}", on_click=delete_note, args=(idx,))

if tab_selection == t("tab_settings"):
    sac.divider(label=t('settings'), icon='gear', align='center')
    st.write(t("settings_desc"))
    
    # 言語設定
    sac.divider(label=t('language_settings_title'), icon='translate', align='left')
    lang = st.selectbox(
        t("display_language"),
        ["日本語", "English", "簡体字"],
        index=["日本語", "English", "簡体字"].index(st.session_state.get("language", "日本語")), key="lang_select"
    )
    if st.session_state.language != lang:
        st.session_state.language = lang
        trigger_rerun()

    # 試験日設定
    sac.divider(label=t('exam_date_settings'), icon='calendar-event', align='left')
    st.caption(t("exam_date_countdown_desc"))
    edate = st.date_input(t("exam_date"), value=st.session_state.exam_date if st.session_state.exam_date else None, key="exam_date_input")
    if st.session_state.exam_date != edate:
        st.session_state.exam_date = edate
        trigger_rerun()

    # テーマ設定
    sac.divider(label=t('theme_color'), icon='palette', align='left')
    theme_keys = list(THEMES.keys())
    try:
        current_index = theme_keys.index(st.session_state.theme)
    except ValueError:
        current_index = 0
        st.session_state.theme = theme_keys[0]
        
    th = st.selectbox(t("select_theme"), theme_keys, index=current_index, key="theme_select")
    if st.session_state.theme != th:
        st.session_state.theme = th
        trigger_rerun()
    
    # 表示モード設定
    st.markdown("---")
    sac.divider(label=t('display_mode'), icon='moon-stars', align='left')
    st.caption(t("dark_mode_settings_desc"))
    
    display_modes = [t("light_mode"), t("dark_mode"), t("system_setting")]
    current_mode = st.session_state.get("display_mode", t("system_setting"))
    
    try:
        mode_index = display_modes.index(current_mode)
    except ValueError:
        mode_index = 2  # デフォルト: システム設定
    
    selected_mode = st.selectbox(
        t("select_display_mode"),
        display_modes,
        index=mode_index,
        key="display_mode_select"
    )
    
    if st.session_state.get("display_mode") != selected_mode:
        st.session_state.display_mode = selected_mode
        trigger_rerun()

    st.markdown("---")
    keep = st.checkbox(t("keep_input_form_open"), value=st.session_state.get("keep_input_open", True), key="keep_input_open_checkbox")
    st.session_state.keep_input_open = keep
    
    # 週報レポート生成
    sac.divider(label=t('weekly_report_generation'), icon='file-earmark-text', align='left')
    st.caption(t("weekly_report_desc"))
    
    if st.button(t("generate_report"), type="primary", use_container_width=True):
        report = generate_weekly_report(df)
        st.markdown(report, unsafe_allow_html=True)
        
        # ダウンロードボタン
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.text_area(t("copy_for_clipboard"), value=report, height=200, key="weekly_report_copy")
        
        with col_dl2:
            # PDF出力
            pdf_data = generate_pdf_report(report, st.session_state.current_user)
            if pdf_data:
                st.download_button(
                    label=t("download_pdf"),
                    data=pdf_data,
                    file_name=f"weekly_report_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info(t("additional_libs_for_pdf"))
        
        with col_dl3:
            # Excel出力（学習データ）
            excel_data = generate_excel_report(df, st.session_state.current_user)
            if excel_data:
                st.download_button(
                    label=t("download_excel"),
                    data=excel_data,
                    file_name=f"learning_data_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("Excel出力には追加ライブラリが必要です")
    
    st.markdown("---")
    st.write("今後の機能予定:")
    st.write("- 自動学習プラン提案")
    st.write("- ユーザー別トラッキング")

st.markdown("</div>", unsafe_allow_html=True)

# ===== カスタム通知（トースト）の表示 =====
if st.session_state.get("show_success_toast", False):
    import time
    toast_id = int(time.time() * 1000)
    st.markdown(f"""
    <style>
    @keyframes slideInFadeOut {{
        0% {{ transform: translateX(100%); opacity: 0; }}
        10% {{ transform: translateX(0); opacity: 1; }}
        80% {{ transform: translateX(0); opacity: 1; }}
        100% {{ transform: translateX(100%); opacity: 0; visibility: hidden; }}
    }}
    .custom-toast-{toast_id} {{
        position: fixed;
        top: 100px;
        right: 20px;
        background-color: #ffffff;
        border-left: 5px solid #10b981;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 10000;
        animation: slideInFadeOut 4s forwards;
    }}
    .toast-icon {{
        color: #10b981;
        font-size: 1.5rem;
    }}
    .toast-message {{
        color: #1f2937;
        font-weight: 600;
        font-size: 1rem;
    }}
    </style>
    <div class="custom-toast-{toast_id}">
        <i class="bi bi-check-circle-fill toast-icon"></i>
        <span class="toast-message">データを追加しました</span>
    </div>
    """, unsafe_allow_html=True)
    # 一度表示したらフラグを下ろす
    st.session_state.show_success_toast = False
