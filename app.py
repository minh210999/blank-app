import streamlit as st
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import os
import time
import urllib.parse


st.set_page_config(page_title="Art Ancestry", layout="wide")


def load_secrets():
    # Prefer st.secrets but fall back to environment variables if absent
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    try:
        google_key = st.secrets.get("GOOGLE_API_KEY", "").strip()
    except Exception:
        google_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    try:
        google_cx = st.secrets.get("GOOGLE_CX", "").strip()
    except Exception:
        google_cx = (os.environ.get("GOOGLE_CX") or "").strip()
    # NOTE: This file is an earlier draft. The active application lives in
    # `streamlit_app.py` and should be started with `streamlit run`.
    return openai_key if openai_key else None, google_key if google_key else None, google_cx if google_cx else None


SYSTEM_PROMPT = """
You are an expert Art Historian specializing in Comparative Historical Analysis. 
Your task is to analyze an uploaded artwork and compare it to its specific historical predecessor from the same country or region. For example if the uploaded work is a 17th-century Dutch painting, identify the most directly influential Dutch work from the late 16th century.

### THE PROTOCOL (5-STEP FRAMEWORK)
You must strictly follow this analytical framework to generate your response:

1. **Gather Evidence (Observation):** - Observe the raw visual facts of the uploaded work (Medium, Line, Color, Subject).
   - Identify the specific *predecessor work* that represents the style immediately preceding this one in the same culture.
   
2. **Group Categories:** - Organize your observations into themes (e.g., "Use of Perspective," "Social Function," "Materials," "Depiction of the Individual").

3. **Identify Importance:** - Select the ONE most significant category where the shift between the two eras is most visible (e.g., "The shift from Religious Symbolism to Secular Realism").

4. **Compare (The Chart):** - Contrast the two works specifically within that category.
   - List the *Similarities* (what was kept?) and *Differences* (what was changed?).

5. **Synthesize (The Argument):** - Write a concise analytical statement using "language of comparison" (e.g., "While the predecessor focused on X, this work shifts to Y...").

### RESPONSE FORMAT (JSON ONLY)
Return valid JSON with no markdown formatting. Use this exact structure:

{
  "current_work": {
    "title": "Title",
    "artist": "Artist",
    "date": "Year",
    "style": "Art Movement/Style (e.g., Baroque, Renaissance, Modernism)",
    "description": "Brief visual summary."
  },
  "predecessor_work": {
    "title": "Title of the historical predecessor",
    "artist": "Artist or Style (e.g., 'Folk Artisan' or 'Court Painter')",
    "date": "Period/Dynasty",
    "style": "Art Movement/Style of the predecessor",
    "search_query": "Artist Title painting high resolution" 
  },
  "analysis": {
    "step_3_category": "The Primary Theme of Change",
    "step_4_comparison": {
       "similarity": "One key stylistic element that remained.",
       "difference": "The crucial innovation or departure."
    },
    "step_5_synthesis": "A 3-4 sentence comparative essay summarizing the historical shift."
  }
}

### FEW-SHOT EXAMPLE (Mental Model)
User Upload: "Playing O An Quan" by Nguyen Phan Chanh (Vietnam, 1931).
Your Output Logic:
- Predecessor: "Rat's Wedding" (Dong Ho Folk Painting).
- Category: "The shift from Symbolic Folk Narrative to Psychological Realism."
- Comparison: Dong Ho prints used flat woodblock lines and satirical animal symbols for communal morality. Nguyen Phan Chanh uses the same earthy color palette (browns/blacks) but applies Western composition and anatomical realism to depict a quiet, poetic moment of daily life.
"""


def analyze_art(image_base64: str, openai_key: str) -> dict:
    """Send the image to OpenAI (gpt-4o) and return parsed JSON result.

    The function forces the assistant to return a valid JSON object.
    Includes retry logic with exponential backoff for rate-limiting (429 errors).
    """
    if not openai_key or not openai_key.strip():
        raise ValueError("Missing or empty OpenAI API key.")

    openai_key = openai_key.strip()
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

    # Format message with image for Vision API
    image_data_url = f"data:image/png;base64,{image_base64}"
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this artwork image following the 5-step framework. Return valid JSON with the structure: {current_work: {title, artist, date, style}, predecessor_work: {title, artist, date, style}, analysis: {step_3_category, step_4_comparison, step_5_synthesis}}. Include the date/period and art movement/style for both works."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }

    # Retry logic with exponential backoff for rate-limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = resp.json()
            except:
                error_body = resp.text
            
            if resp.status_code == 429:
                # Rate limited: wait and retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential: 1s, 2s, 4s
                    st.warning(f"Rate limited by OpenAI. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ValueError(f"Rate limited after {max_retries} attempts. Please try again in a few moments.")
            elif resp.status_code == 400:
                raise ValueError(f"Bad request to OpenAI (invalid model?). Error: {json.dumps(error_body, indent=2)}")
            else:
                raise ValueError(f"OpenAI API error ({resp.status_code}): {error_body}")

    # Extract assistant text
    assistant_text = None
    try:
        assistant_text = data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback for different response shapes
        assistant_text = data.get("choices", [{}])[0].get("text") or data.get("output", [{}])[0].get("content")

    if not assistant_text:
        raise ValueError("No response text from OpenAI.")

    # Try parsing JSON directly; if it fails, try to salvage the JSON substring.
    try:
        result = json.loads(assistant_text)
    except Exception:
        # Extract JSON object by finding first { and last }
        try:
            start_idx = assistant_text.find('{')
            end_idx = assistant_text.rfind('}')
            if start_idx >= 0 and end_idx > start_idx:
                json_str = assistant_text[start_idx:end_idx + 1]
                result = json.loads(json_str)
            else:
                raise ValueError(f"Assistant did not return parseable JSON. Raw output:\n{assistant_text}")
        except Exception:
            raise ValueError(f"Assistant did not return parseable JSON. Raw output:\n{assistant_text}")
    
    # Flatten nested structure if present
    if "current_work" in result and isinstance(result["current_work"], dict):
        current = result["current_work"]
        result