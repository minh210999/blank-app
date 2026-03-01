import streamlit as st
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import os
import re
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
    return openai_key if openai_key else None, google_key if google_key else None, google_cx if google_cx else None


SYSTEM_PROMPT = system_prompt = """
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
    """Send the image to OpenAI (GPT-4o) and return parsed JSON result.

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
        result["current_work_title"] = current.get("title", "Unknown")
        result["current_work_artist"] = current.get("artist", "")
        result["current_work_date"] = current.get("date", "")
        result["current_work_style"] = current.get("style", "")
    
    if "predecessor_work" in result and isinstance(result["predecessor_work"], dict):
        pred = result["predecessor_work"]
        result["predecessor_title"] = pred.get("title", "")
        result["predecessor_artist"] = pred.get("artist", "")
        result["predecessor_date"] = pred.get("date", "")
        result["predecessor_style"] = pred.get("style", "")
    
    # Extract analysis text from nested structure
    if "analysis" in result and isinstance(result["analysis"], dict):
        analysis = result["analysis"]
        # Build analysis text from fields
        parts = []
        if "step_3_category" in analysis:
            parts.append(f"**Primary Theme of Change:** {analysis['step_3_category']}")
        if "step_4_comparison" in analysis and isinstance(analysis["step_4_comparison"], dict):
            comp = analysis["step_4_comparison"]
            parts.append(f"\n**Comparison:**")
            if "similarity" in comp:
                parts.append(f"- Similarity: {comp['similarity']}")
            if "difference" in comp:
                parts.append(f"- Difference: {comp['difference']}")
        if "step_5_synthesis" in analysis:
            parts.append(f"\n**Analysis:** {analysis['step_5_synthesis']}")
        result["analysis_text"] = "\n".join(parts)
    
    # Handle flat structure (fallback)
    if "current_work_title" not in result:
        result["current_work_title"] = result.get("current_work_title", "Unknown")
    if "current_work_artist" not in result:
        result["current_work_artist"] = result.get("current_work_artist", "")
    if "current_work_date" not in result:
        result["current_work_date"] = result.get("current_work_date", "")
    if "current_work_style" not in result:
        result["current_work_style"] = result.get("current_work_style", "")
    if "predecessor_title" not in result:
        result["predecessor_title"] = result.get("predecessor_title", "")
    if "predecessor_artist" not in result:
        result["predecessor_artist"] = result.get("predecessor_artist", "")
    if "predecessor_date" not in result:
        result["predecessor_date"] = result.get("predecessor_date", "")
    if "predecessor_style" not in result:
        result["predecessor_style"] = result.get("predecessor_style", "")
    if "analysis_text" not in result:
        result["analysis_text"] = result.get("analysis_text", "No analysis returned.")
    
    return result


def get_wikimedia_image(query: str) -> str:
    """Search Wikimedia Commons for an image and return the image URL."""
    if not query:
        return None
    
    url = "https://commons.wikimedia.org/w/api.php"
    
    # Wikimedia requires a User-Agent header
    headers = {
        "User-Agent": "Art-Ancestry-App/1.0 (Streamlit; +https://github.com)"
    }
    
    # Request a generated thumbnail up to 2048px wide (if available).
    # The MediaWiki API will include a `thumburl` when iiurlwidth is provided.
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,  # Namespace 6 is "File:" (Images)
        "gsrsearch": query,  # Simplified search query
        "gsrlimit": 10,  # Get up to 10 results
        "prop": "imageinfo",
        # Request url, size, mime and thumburl fields
        "iiprop": "url|size|mime|thumburl",
        # Ask the API to generate a thumbnail with max width 2048px
        "iiurlwidth": 2048,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id in pages:
            image_info = pages[page_id].get("imageinfo", [])
            if not image_info:
                continue
            info = image_info[0]
            # Prefer the generated thumbnail (thumburl) which will be <= 2048px wide.
            if info.get("thumburl"):
                return info["thumburl"]
            # Fall back to the original file URL if thumbnail not provided
            if info.get("url"):
                return info["url"]

        st.info(f"No images found for '{query}'. You can still view the analysis above.")
        return None
    except Exception as e:
        st.warning(f"Error searching Wikimedia: {e}")
        return None


def main():
    st.title("Art Ancestry — Comparative Historical Analysis")

    openai_key, _, _ = load_secrets()
    if not openai_key:
        st.error("OpenAI API key not found. Put OPENAI_API_KEY in .streamlit/secrets.toml or environment.")
        return

    st.write("Upload a JPG or PNG image of an artwork to analyze its probable predecessor and comparative context.")
    uploaded = st.file_uploader("Upload artwork image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        # Read image bytes and encode base64
        image_bytes = uploaded.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

        with st.spinner("Analyzing provenance..."):
            try:
                analysis = analyze_art(image_b64, openai_key)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        # Validate analysis fields
        current_title = analysis.get("current_work_title", "Unknown")
        current_artist = analysis.get("current_work_artist", "")
        current_date = analysis.get("current_work_date", "")
        current_style = analysis.get("current_work_style", "")
        pred_title = analysis.get("predecessor_title") or ""
        pred_artist = analysis.get("predecessor_artist") or ""
        pred_date = analysis.get("predecessor_date", "")
        pred_style = analysis.get("predecessor_style", "")
        analysis_text = analysis.get("analysis_text", "No analysis returned.")

        # Try to fetch predecessor image from Wikimedia
        pred_image_url = None
        if pred_title and pred_artist:
            query = f"{pred_title} {pred_artist}"
            pred_image_url = get_wikimedia_image(query)

        # Layout: two rows
        left, right = st.columns(2)

        with left:
            st.subheader("Your Upload")
            st.image(image, use_container_width=True)
            identified_text = current_title
            if current_artist:
                identified_text += f" — {current_artist}"
            if current_style:
                identified_text += f" ({current_style})"
            if current_date:
                identified_text += f" — {current_date}"
            st.markdown(f"**Identified:** {identified_text}")
            search_query_current = f"{current_title} {current_artist}".strip()
            search_url_current = f"https://www.google.com/search?q={urllib.parse.quote(search_query_current)}&tbm=isch"
            st.link_button("🔍 double check", search_url_current)

        with right:
            st.subheader("The Predecessor")
            if pred_image_url:
                st.image(pred_image_url, use_container_width=True)
            else:
                st.info("Predecessor image not found in Wikimedia Commons.")
            
            pred_text = f"{pred_title} — {pred_artist}"
            if pred_style:
                pred_text += f" ({pred_style})"
            if pred_date:
                pred_text += f" — {pred_date}"
            st.markdown(f"**Predecessor:** {pred_text}")
            search_query_pred = f"{pred_title} {pred_artist} {pred_style}".strip()
            search_url_pred = f"https://www.google.com/search?q={urllib.parse.quote(search_query_pred)}&tbm=isch"
            st.link_button("🔍 double check", search_url_pred)

        st.markdown("---")
        st.subheader("Comparative Historical Analysis")
        st.markdown(analysis_text)


if __name__ == "__main__":
    main()
