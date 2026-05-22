import streamlit as st
import datetime
from pathlib import Path
import base64


def render_header(page_title: str, page_subtitle: str):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Locate local FNB logo file and parse to base64
    ROOT_DIR = Path(__file__).resolve().parents[1]
    logo_path = ROOT_DIR / "assets" / "fnb_logo.png"

    logo_html = ""
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            logo_html = f'<div style="background-color: white; padding: 6px; border-radius: 8px; display: inline-block; margin-right: 18px; vertical-align: middle;"><img src="data:image/png;base64,{encoded_string}" style="width: 55px; height: auto; display: block;" /></div>'
        except Exception:
            pass

    # Pure, pristine HTML template string
    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: transparent;
            }}
            .banner-card {{
                background-color: #00A3C4 !important;
                border-left: 12px solid #FFB800 !important;
                padding: 22px 28px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-sizing: border-box;
                width: 100%;
            }}
            .left-panel {{
                display: flex;
                align-items: center;
            }}
            .title-text {{
                color: #FFFFFF !important;
                font-size: 2.1rem;
                font-weight: 700;
                margin: 0;
                padding: 0;
                line-height: 1.1;
                letter-spacing: -0.5px;
            }}
            .subtitle-text {{
                color: rgba(255, 255, 255, 0.95);
                font-size: 1.05rem;
                font-weight: 400;
                margin: 6px 0 0 0;
                padding: 0;
            }}
            .right-panel {{
                text-align: right;
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 6px;
                min-width: 240px;
            }}
            .status-badge {{
                background-color: rgba(255, 255, 255, 0.25);
                color: #FFFFFF;
                padding: 5px 14px;
                border-radius: 20px;
                font-size: 0.82rem;
                font-weight: 600;
                border: 1px solid rgba(255, 255, 255, 0.35);
                display: inline-block;
                letter-spacing: 0.5px;
            }}
            .meta-text {{
                color: rgba(255, 255, 255, 0.88);
                font-size: 0.8rem;
                line-height: 1.4;
                font-weight: 400;
                margin: 0;
            }}
        </style>
    </head>
    <body>
        <div class="banner-card">
            <div class="left-panel">
                {logo_html}
                <div>
                    <h1 class="title-text">{page_title}</h1>
                    <p class="subtitle-text">{page_subtitle}</p>
                </div>
            </div>
            <div class="right-panel">
                <div class="status-badge">● System Active</div>
                <p class="meta-text">
                    Server Time: {current_time}<br>
                    
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    # Renders inside a clean HTML component boundary to protect visuals from Streamlit theme resets
    st.components.v1.html(raw_html, height=135, scrolling=False)
