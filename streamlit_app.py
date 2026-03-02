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


# ---------- configuration / constants ----------
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
"""


# ---------- helpers ----------

def load_secrets():
    """Look for API keys in Streamlit secrets or environment variables.

    Returns a tuple (openai_key, google_key, google_cx); each may be `None`.
    """
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

    return (
        openai_key if openai_key else None,
        google_key if google_key else None,
        google_cx if google_cx else None,
    )


def get_painting_info(title: str, artist: str, openai_key: str) -> str:
    """Ask OpenAI for a concise description of a painting.

    If the model cannot find the work, returns a fallback string.
    """
    if not openai_key or not openai_key.strip():
        raise ValueError("Missing or empty OpenAI API key for painting info request.")

    prompt = (
        f"Provide a very brief (1-2 sentence) description of the painting '" +
        f"{title}' by {artist if artist else 'the stated artist'}. "
        "If you do not know this work, respond with 'Information not available.'"
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a concise art encyclopedia."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 150,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return text or "Information not available."


def analyze_art(image_base64: str, openai_key: str) -> dict:
    """Send the uploaded image to OpenAI and parse the JSON response.

    Applies exponential backoff for rate limits and attempts to salvage
    malformed JSON if necessary.  Raises on missing API key or failed parse.
    """
    if not openai_key or not openai_key.strip():
        raise ValueError("Missing or empty OpenAI API key.")

    openai_key = openai_key.strip()
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

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
                        "text": (
                            "Analyze this artwork image following the 5-step "
                            "framework. Return valid JSON with the structure: "
                            "{current_work: {title, artist, date, style}, "
                            "predecessor_work: {title, artist, date, style}, "
                            "analysis: {step_3_category, step_4_comparison, "
                            "step_5_synthesis}}. Include the date/period "
                            "and art movement/style for both works."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }

    # retry loop
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError as e:
            status = getattr(resp, 'status_code', None)
            if status == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            elif status == 429:
                raise ValueError("Rate limited after retries.")
            else:
                raise

    assistant_text = None
    try:
        assistant_text = data["choices"][0]["message"]["content"]
    except Exception:
        assistant_text = data.get("choices", [{}])[0].get("text") or data.get("output", [{}])[0].get("content")

    if not assistant_text:
        raise ValueError("No response text from OpenAI.")

    # parse JSON, attempt salvage if necessary
    try:
        result = json.loads(assistant_text)
    except Exception:
        start_idx = assistant_text.find("{")
        end_idx = assistant_text.rfind("}")
        if start_idx >= 0 and end_idx > start_idx:
            result = json.loads(assistant_text[start_idx:end_idx + 1])
        else:
            raise ValueError(f"Assistant did not return parseable JSON. Raw output:\n{assistant_text}")

    # flatten nested fields for convenience later
    if "current_work" in result and isinstance(result["current_work"], dict):
        current = result["current_work"]
        result.update({
            "current_work_title": current.get("title", "Unknown"),
            "current_work_artist": current.get("artist", ""),
            "current_work_date": current.get("date", ""),
            "current_work_style": current.get("style", ""),
        })

    if "predecessor_work" in result and isinstance(result["predecessor_work"], dict):
        pred = result["predecessor_work"]
        result.update({
            "predecessor_title": pred.get("title", ""),
            "predecessor_artist": pred.get("artist", ""),
            "predecessor_date": pred.get("date", ""),
            "predecessor_style": pred.get("style", ""),
        })

    if "analysis" in result and isinstance(result["analysis"], dict):
        analysis = result["analysis"]
        parts = []
        if "step_3_category" in analysis:
            parts.append(f"**Primary Theme of Change:** {analysis['step_3_category']}")
        comp = analysis.get("step_4_comparison", {})
        if comp:
            parts.append("\n**Comparison:**")
            if "similarity" in comp:
                parts.append(f"- Similarity: {comp['similarity']}")
            if "difference" in comp:
                parts.append(f"- Difference: {comp['difference']}")
        if "step_5_synthesis" in analysis:
            parts.append(f"\n**Analysis:** {analysis['step_5_synthesis']}")
        result["analysis_text"] = "\n".join(parts)

    # ensure keys exist
    for key in [
        "current_work_title", "current_work_artist", "current_work_date", "current_work_style",
        "predecessor_title", "predecessor_artist", "predecessor_date", "predecessor_style",
        "analysis_text",
    ]:
        result.setdefault(key, "")

    return result


def get_wikimedia_image(query: str) -> str:
    """Return a thumbnail URL from Wikimedia Commons or None if not found."""
    if not query:
        return None

    url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": "Art-Ancestry-App/1.0 (Streamlit)"}
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": 10,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|thumburl",
        "iiurlwidth": 2048,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("thumburl"):
                return info["thumburl"]
            if info.get("url"):
                return info["url"]
    except Exception as e:
        st.warning(f"Error searching Wikimedia: {e}")
    st.info(f"No images found for '{query}'.")
    return None


# ---------- presentation helpers ----------

def format_identified_text(title, artist, style, date):
    parts = [title or "Unknown"]
    if artist:
        parts.append(f"— {artist}")
    if style:
        parts.append(f"({style})")
    if date:
        parts.append(f"— {date}")
    return " ".join(parts)


def format_predecessor_text(title, artist, style, date):
    parts = [title or ""]
    if artist:
        parts.append(f"— {artist}")
    if style:
        parts.append(f"({style})")
    if date:
        parts.append(f"— {date}")
    return " ".join(parts)


def render_columns(image, current_title, current_artist, current_style, current_date,
                   pred_image_url, pred_title, pred_artist, pred_style, pred_date):
    left, right = st.columns(2)
    with left:
        st.subheader("Your Upload")
        st.image(image, use_container_width=True)
        identified = format_identified_text(current_title, current_artist, current_style, current_date)
        st.markdown(f"**Identified:** {identified}")
        query = f"{current_title} {current_artist}".strip()
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
        st.link_button("🔍 double check", url)

    with right:
        st.subheader("The Predecessor")
        if pred_image_url:
            st.image(pred_image_url, use_container_width=True)
        else:
            st.info("Predecessor image not found in Wikimedia Commons.")
        pred_text = format_predecessor_text(pred_title, pred_artist, pred_style, pred_date)
        st.markdown(f"**Predecessor:** {pred_text}")
        query = f"{pred_title} {pred_artist} {pred_style}".strip()
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
        st.link_button("🔍 double check", url)


# ---------- main application ----------

def main():
    st.title("Art Ancestry — Comparative Historical Analysis")

    openai_key, _, _ = load_secrets()
    if not openai_key:
        st.error("OpenAI API key not found. Put OPENAI_API_KEY in .streamlit/secrets.toml or environment.")
        return

    st.write("Upload a JPG or PNG image of an artwork to analyze its probable predecessor and comparative context.")
    uploaded = st.file_uploader("Upload artwork image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image_bytes = uploaded.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with st.spinner("Analyzing provenance..."):
        try:
            analysis = analyze_art(b64, openai_key)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    current_title = analysis.get("current_work_title", "Unknown")
    current_artist = analysis.get("current_work_artist", "")
    current_date = analysis.get("current_work_date", "")
    current_style = analysis.get("current_work_style", "")
    pred_title = analysis.get("predecessor_title", "")
    pred_artist = analysis.get("predecessor_artist", "")
    pred_date = analysis.get("predecessor_date", "")
    pred_style = analysis.get("predecessor_style", "")
    analysis_text = analysis.get("analysis_text", "No analysis returned.")

    painting_info = ""
    if current_title and current_title != "Unknown":
        try:
            painting_info = get_painting_info(current_title, current_artist, openai_key)
        except Exception:
            painting_info = "Information not available."

    pred_image_url = None
    if pred_title and pred_artist:
        query = f"{pred_title} {pred_artist}"
        pred_image_url = get_wikimedia_image(query)

    if painting_info:
        st.subheader("About This Painting")
        st.markdown(painting_info)
        st.markdown("---")

    render_columns(image, current_title, current_artist, current_style, current_date,
                   pred_image_url, pred_title, pred_artist, pred_style, pred_date)

    st.markdown("---")
    st.subheader("Comparative Historical Analysis")
    st.markdown(analysis_text)


if __name__ == "__main__":
    main()
