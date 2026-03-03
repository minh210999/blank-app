import streamlit as st
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import os
import time
import urllib.parse
import re
from PIL import ExifTags


st.set_page_config(page_title="Art a Look", layout="wide")


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
       "similarity": "One or Two key stylistic elements that remained.",
       "difference": "The One or Two crucial innovation or departure."
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


def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Fix image orientation based on EXIF data.
    
    Handles rotation that occurs in images from phones/cameras.
    Returns the rotated image or original if no EXIF orientation found.
    """
    try:
        exif = image._getexif()
        if exif is None:
            return image
        
        exif_dict = {ExifTags.TAGS[k]: v for k, v in exif.items() if k in ExifTags.TAGS}
        orientation = exif_dict.get('Orientation', 1)
        
        # Apply rotation based on EXIF orientation tag
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        
        return image
    except Exception:
        # If any error occurs, just return the original image
        return image


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


def get_art_movement_info(movement: str, openai_key: str) -> str:
    """Ask OpenAI for a concise description of an art movement.

    If the model cannot find the movement, returns a fallback string.
    """
    if not openai_key or not openai_key.strip():
        raise ValueError("Missing or empty OpenAI API key for art movement info request.")

    prompt = (
        f"Provide a very brief (2-3 sentence) description of the '{movement}' art movement, "
        "including its key characteristics and time period. "
        "If you do not know this movement, respond with 'Information not available.'"
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
        "max_tokens": 200,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return text or "Information not available."


def get_fun_fact(title: str, artist: str, openai_key: str) -> str:
    """Ask OpenAI for a fun fact about a painting.

    Returns a single-sentence tidbit or a placeholder if none is available.
    """
    if not openai_key or not openai_key.strip():
        raise ValueError("Missing or empty OpenAI API key for fun fact request.")

    prompt = (
        f"Provide one engaging fun fact about the painting '{title}' by {artist}. "
        "Keep it to a single sentence. If no fact is available, reply 'No fun fact available.'"
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a friendly art guide."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 60,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return text or "No fun fact available."


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


def get_wikimedia_images(query: str) -> list:
    """Return up to 10 images from Wikimedia Commons with their metadata.
    
    Returns a list of dicts with keys: url, thumburl, title, description
    """
    if not query:
        return []

    url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": "Art-Ancestry-App/1.0 (Streamlit)"}
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": 10,
        "prop": "imageinfo|pageterms",
        "iiprop": "url|size|mime|thumburl|extmetadata",
        "iiurlwidth": 2048,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        
        images = []
        for page_id, page in pages.items():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("thumburl") or info.get("url"):
                image_data = {
                    "url": info.get("url", ""),
                    "thumburl": info.get("thumburl", ""),
                    "title": page.get("title", "Unknown"),
                }
                images.append(image_data)
        return images
    except Exception as e:
        st.warning(f"Error searching Wikimedia: {e}")
        return []


def verify_images_with_ai(uploaded_image_base64: str, images: list, openai_key: str) -> tuple:
    """Verify which image from Wikimedia matches the uploaded artwork using OpenAI.
    
    Returns (verified_image, is_confident) where:
    - verified_image is the matching image dict or None
    - is_confident is True if we're confident in the match
    """
    if not images or not openai_key:
        return None, False

    # Verify up to 10 images
    verification_results = []
    
    for idx, img_data in enumerate(images[:10]):
        if not img_data.get("url"):
            continue
        
        attempts = 0
        while attempts < 2:  # Retry once if it fails
            try:
                # Use direct URL for wikimedia image (OpenAI supports direct URLs)
                wikimedia_url = img_data.get("url", "")
                
                if not wikimedia_url:
                    break
                
                # Use OpenAI to compare
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "I will show you two images. The first is a user-uploaded artwork. The second is a candidate image from Wikimedia Commons. Are they the same artwork or showing the same painting? Answer with ONLY 'YES', 'NO', or 'MAYBE' followed by a confidence score 0-100."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{uploaded_image_base64}"}
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": wikimedia_url}
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50,
                }
                
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                resp_data = resp.json()
                response_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # Parse response
                confidence = 0
                verdict = "NO"
                if "YES" in response_text.upper():
                    verdict = "YES"
                    # Extract confidence score
                    try:
                        match = re.search(r'(\d+)', response_text)
                        if match:
                            confidence = int(match.group(1))
                    except:
                        confidence = 85
                elif "MAYBE" in response_text.upper():
                    verdict = "MAYBE"
                    try:
                        match = re.search(r'(\d+)', response_text)
                        if match:
                            confidence = int(match.group(1))
                    except:
                        confidence = 50
                
                verification_results.append({
                    "image": img_data,
                    "verdict": verdict,
                    "confidence": confidence,
                    "response": response_text
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                attempts += 1
                if attempts >= 2:
                    continue  # Move to next image
                time.sleep(1)  # Wait before retry
    
    # Find the best match
    if verification_results:
        # Sort by verdict priority and confidence
        def score_result(result):
            verdict = result["verdict"]
            confidence = result["confidence"]
            if verdict == "YES":
                return (2, confidence)
            elif verdict == "MAYBE":
                return (1, confidence)
            else:
                return (0, confidence)
        
        verification_results.sort(key=score_result, reverse=True)
        best = verification_results[0]
        
        # Consider it verified if high confidence YES or high confidence MAYBE
        is_confident = (best["verdict"] == "YES" and best["confidence"] >= 70) or \
                       (best["verdict"] == "MAYBE" and best["confidence"] >= 85)
        
        if is_confident:
            return best["image"], True
        else:
            # If not confident, return list of top 5 for slideshow
            top_5 = verification_results[:5]
            return [r["image"] for r in top_5], False
    
    return None, False


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
        st.link_button("🔍 double check", url, type="primary")

    with right:
        st.subheader("The Predecessor")
        if isinstance(pred_image_url, str) and pred_image_url:
            st.image(pred_image_url, use_container_width=True)
        elif isinstance(pred_image_url, list) and pred_image_url:
            st.info("⚠️ **Note:** Images might be incorrect. Please double-check!")
            # Slideshow of possible matches
            st.write("**Possible matches from Wikimedia (scroll to view all):**")
            cols = st.columns(min(len(pred_image_url), 3))
            for idx, img in enumerate(pred_image_url):
                with cols[idx % len(cols)]:
                    if img.get("thumburl"):
                        st.image(img["thumburl"], use_container_width=True, caption=f"Option {idx+1}")
                    elif img.get("url"):
                        st.image(img["url"], use_container_width=True, caption=f"Option {idx+1}")
                    st.caption(img.get("title", "Unknown"))
        else:
            st.info("Predecessor image not found in Wikimedia Commons.")
        pred_text = format_predecessor_text(pred_title, pred_artist, pred_style, pred_date)
        st.markdown(f"**Predecessor:** {pred_text}")
        query = f"{pred_title} {pred_artist} {pred_style}".strip()
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
        st.link_button("🔍 double check", url, type="primary")


# ---------- main application ----------

def main():
    st.title("Art a Look: Understanding Artworks")

    openai_key, _, _ = load_secrets()
    if not openai_key:
        st.error("OpenAI API key not found. Put OPENAI_API_KEY in .streamlit/secrets.toml or environment.")
        return

    st.write("Google Lens for art lover! Upload a JPG or PNG image of an artwork to analyze it and put it in context with artwork around the same period.")
    uploaded = st.file_uploader("Upload artwork image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image_bytes = uploaded.read()
    image = Image.open(BytesIO(image_bytes))
    # Fix orientation based on EXIF data before converting to RGB
    image = fix_image_orientation(image)
    image = image.convert("RGB")
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

    movement_info = ""
    if current_style:
        try:
            movement_info = get_art_movement_info(current_style, openai_key)
        except Exception:
            movement_info = ""

    fun_fact = ""
    if current_title and current_artist:
        try:
            fun_fact = get_fun_fact(current_title, current_artist, openai_key)
        except Exception:
            fun_fact = ""  # silent failure

    pred_image_url = None
    is_image_verified = False
    
    if pred_title and pred_artist:
        with st.spinner("Verifying predecessor image from Wikimedia..."):
            # Search with both title and artist for more specific results
            # Try primary search with both title and artist
            query = f'"{pred_title}" {pred_artist}'
            images = get_wikimedia_images(query)
            
            # If no results, try without quotes
            if not images:
                query = f"{pred_title} {pred_artist}"
                images = get_wikimedia_images(query)
            
            # If still no results, try just the title
            if not images and pred_title:
                images = get_wikimedia_images(pred_title)
            
            if images:
                result, is_verified = verify_images_with_ai(b64, images, openai_key)
                is_image_verified = is_verified
                
                if is_verified and isinstance(result, dict):
                    # Single verified image
                    pred_image_url = result.get("thumburl") or result.get("url")
                elif result and isinstance(result, list) and len(result) > 0:
                    # List of possible matches
                    pred_image_url = result
                else:
                    # Fallback: just get first image
                    first_img = images[0]
                    pred_image_url = first_img.get("thumburl") or first_img.get("url")

    if painting_info:
        st.subheader("About This Painting")
        left, right = st.columns(2, gap="large")
        with left:
            st.image(image, use_container_width=True)
        with right:
            st.markdown(painting_info)
            st.markdown(f"**Name:** {current_title}")
            st.markdown(f"**Artist:** {current_artist}")
            st.markdown(f"**Date:** {current_date}")
            if movement_info:
                with st.expander(f"**Movement:** {current_style}"):
                    st.markdown(movement_info)
            if fun_fact:
                with st.expander(f"**Fun Fact**"):
                    st.markdown(fun_fact)

    st.subheader("Compare to painting around the same time?")
    with st.expander("**Compare**"):
        if is_image_verified:
            st.success("✓ Predecessor image verified!")
        
        render_columns(image, current_title, current_artist, current_style, current_date,
                       pred_image_url, pred_title, pred_artist, pred_style, pred_date)

        st.markdown("---")
        st.subheader("Understanding the Shift")
        st.markdown(analysis_text)


if __name__ == "__main__":
    main()
