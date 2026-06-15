"""
Ghibli-Vibe Travel Mapper
=========================
사용자가 원하는 여행 분위기를 일상어로 입력하면,
HuggingFace Inference API의 LLM을 통해 지브리 애니메이션 감성과 연상되는
전 세계 실제 여행지와 반나절 여행 동선을 추천해 주는 서비스입니다.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.parse
from typing import Any

import gradio as gr
from gradio_client import utils as _gc_utils  # noqa: E402

# --- workaround: gradio_client의 JSON Schema walker가 bool 스키마를 만나면
# 터지는 버그(#10178) 우회. Label/JSON 컴포넌트가 생성하는
# additionalProperties: true 스키마에서 발생한다.
_orig_get_type = _gc_utils.get_type
def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)
_gc_utils.get_type = _safe_get_type

_orig_j2p = _gc_utils._json_schema_to_python_type
def _safe_j2p(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2p(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_j2p

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from model_config import LLM_MODEL, get_token

load_dotenv()

SYSTEM_PROMPT = (
    "너는 감성적인 여행 큐레이터이자 스튜디오 지브리(Studio Ghibli) 애니메이션 전문가인 '지브리 감성 여행 큐레이터'다.\n"
    "사용자가 원하는 여행 분위기나 감성, 테마를 입력하면, 다음 기준에 따라 실제 여행지와 코스를 추천하라:\n"
    "1. 사용자의 입력에서 추상적인 감성(예: 조용함, 아기자기함, 레트로, 청량함, 자연, 신비로움)을 분석한다.\n"
    "2. 이 분위기와 가장 잘 매치되는 스튜디오 지브리 애니메이션 작품을 하나 선정한다. 반드시 실제 스튜디오 지브리 대표작(예: 이웃집 토토로, 센과 치히로의 행방불명, 마녀 배달부 키키, 하울의 움직이는 성, 천공의 성 라퓨타, 모노노케 히메, 벼랑 위의 포뇨, 귀를 기울이면, 바람계곡의 나우시카, 마루 밑 아리에티 등) 중에서만 선택해야 하며, 신카이 마코토 작품이나 다른 제작사 애니메이션(예: 너의 이름은, 목소리의 형태 등)은 절대 제외하라.\n"
    "3. 그 지브리 작품의 감성을 고스란히 느낄 수 있는 '실제 전 세계 여행지(도시, 마을, 혹은 특정 장소)' 1곳을 추천한다.\n"
    "   ※ destination과 half_day_course의 place 필드에는 반드시 영문 지명을 괄호 () 안에 함께 표기해야 한다. (예: 일본 교토 아라시야마 (Arashiyama, Kyoto))\n"
    "4. 이 장소가 선정된 이유와 지브리 작품의 어떤 장면/감성과 연결되는지 따뜻하고 감성적인 어조로 설명하는 '감성 코멘트'를 작성한다.\n"
    "5. 추천된 장소를 중심으로 도보 또는 가볍게 이동 가능한 3단계 반나절 여행 동선(코스)을 기획한다. 각 단계는 장소 이름과 그곳에서 느낄 수 있는 감성적 활동을 포함해야 한다.\n\n"
    "반드시 아래 JSON 스키마 형식으로만 응답해야 하며, 다른 여담이나 설명, markdown 코드 블록 기호(```json 등)는 절대 포함하지 마라. JSON 객체로만 출력하라.\n"
    "{\n"
    '  "destination": "추천 목적지 한국어명과 영문명. 반드시 영문명을 괄호 () 안에 포함할 것 (예: 일본 가마쿠라 (Kamakura, Japan))",\n'
    '  "matching_ghibli_work": "매칭되는 지브리 애니메이션 제목 (예: 이웃집 토토로)",\n'
    '  "vibe_comment": "따뜻하고 서정적인 어조의 감성 코멘트 (2-3문장)",\n'
    '  "half_day_course": [\n'
    "    {\n"
    '      "step": 1,\n'
    '      "place": "코스 1의 장소명 (영문명 포함, 예: 고쿠라쿠지 (Gokurakuji))",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    },\n"
    "    {\n"
    '      "step": 2,\n'
    '      "place": "코스 2의 장소명 (영문명 포함, 예: 에노시마 (Enoshima))",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    },\n"
    "    {\n"
    '      "step": 3,\n'
    '      "place": "코스 3의 장소명 (영문명 포함, 예: 가마쿠라 대불 (Kamakura Daibutsu))",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    }\n"
    "  ]\n"
    "}"
)

# 지브리 작품별 감성 Fallback 이미지 (Wikipedia 검색 실패 시 분위기에 맞는 사진 제공)
GHIBLI_FALLBACK_IMAGES = {
    "토토로":        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=800",
    "센과 치히로":   "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=800",
    "모노노케":      "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    "마녀 배달부":   "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=800",
    "하울":          "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=800",
    "라퓨타":        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    "나우시카":      "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    "포뇨":          "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=800",
    "귀를 기울이면": "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=800",
    "아리에티":      "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    "키키":          "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=800",
}
_DEFAULT_FALLBACK_IMAGE = "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800"


FALLBACK_RECOMMENDATION = {
    "destination": "일본 유후인 온천마을 (Yufuin)",
    "matching_ghibli_work": "이웃집 토토로 (My Neighbor Totoro)",
    "vibe_comment": "푸르른 긴린코 호수와 아기자기한 상점들이 늘어선 골목길을 걷다 보면, 금방이라도 토토로가 살고 있는 신비한 고목나무 숲이 나타날 것 같은 아늑한 감성을 줍니다.",
    "half_day_course": [
        {
            "step": 1,
            "place": "유후인 거리 산책 (Yufuin Floral Village)",
            "description": "동화 속 마을처럼 꾸며진 아기자기한 골목 상점가에서 지브리 캐릭터 샵을 구경하며 따뜻한 감성에 젖어듭니다."
        },
        {
            "step": 2,
            "place": "긴린코 호수 (Lake Kinrinko)",
            "description": "아침 안개가 피어오르는 맑고 조용한 호숫가를 따라 걸으며, 자연 속 평화로움을 느낍니다."
        },
        {
            "step": 3,
            "place": "숲속 전통 찻집 (Classic Tea House)",
            "description": "호수 인근의 오래된 목조 건물 찻집에서 따뜻한 녹차를 마시며 고즈넉한 시간을 보냅니다."
        }
    ]
}

def parse_llm_output_safely(raw_output: Any) -> dict[str, Any]:
    """
    LLM 출력을 안전하게 JSON으로 파싱합니다.
    마크다운 코드블록, 부가 텍스트, trailing comma 등 다양한 형식을 처리합니다.
    """
    if isinstance(raw_output, dict):
        return raw_output

    if not isinstance(raw_output, str):
        raise ValueError("Invalid JSON output")

    text = raw_output.strip()

    # 단계 1: 그대로 파싱 시도
    try:
        return json.loads(text)
    except Exception:
        pass

    # 단계 2: 마크다운 코드블록 제거 후 파싱 (```json ... ``` 또는 ``` ... ```)
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    stripped = re.sub(r"```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # 단계 3: 첫 번째 { 부터 마지막 } 까지 추출해서 파싱
    start = text.find("{")
    end = text.rfind("}")
    candidate = ""
    if start != -1 and end != -1 and end > start:
        candidate = text[start: end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 단계 4: 마크다운 제거 후 { ~ } 재시도
    start2 = stripped.find("{")
    end2 = stripped.rfind("}")
    candidate2 = ""
    if start2 != -1 and end2 != -1 and end2 > start2:
        candidate2 = stripped[start2: end2 + 1]
        try:
            return json.loads(candidate2)
        except Exception:
            pass

    # 단계 5: trailing comma 등 흔한 JSON 오류 자동 수정 후 파싱
    base = candidate2 or candidate
    if base:
        try:
            fixed = re.sub(r",\s*([}\]])", r"\1", base)
            return json.loads(fixed)
        except Exception:
            pass

    print(f"[parse_llm_output_safely] 파싱 실패. 원본 일부:\n{text[:400]}")
    raise ValueError("Invalid JSON output")


def _validate_image_url(url: str) -> bool:
    """URL이 실제로 접근 가능한지 HEAD 요청으로 확인한다."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        req.get_method = lambda: 'HEAD'
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _search_wiki_image(lang: str, q: str) -> str | None:
    """Wikipedia API로 특정 언어/쿼리에 맞는 이미지 URL을 반환. 없으면 None."""
    try:
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(q)}&format=json"
        )
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode('utf-8'))
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return None
            page_title = search_results[0]["title"]

        img_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(page_title)}"
            f"&prop=pageimages&format=json&pithumbsize=640"
        )
        req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for _, page_data in pages.items():
                source = page_data.get("thumbnail", {}).get("source")
                # URL이 실제로 접근 가능한지 검증 후 반환
                if source and _validate_image_url(source):
                    return source
    except Exception as e:
        print(f"Wikipedia image search failed ({lang}, '{q}'): {e}")
    return None


def get_real_image_wiki(query: str) -> str | None:
    """목적지 쿼리로 Wikipedia 이미지를 검색한다.
    괄호 안 영문명을 우선 사용하고, 국가명 접두사를 제거한 한국어명도 시도한다.
    """
    # 괄호 안 영문명 추출 (예: "유후인 온천마을 (Yufuin)" → "Yufuin")
    english_match = re.search(r'\(([A-Za-z][\w\s,\-\']+)\)', query)
    english_name = english_match.group(1).strip() if english_match else None

    # 국가명/불필요 단어 제거한 한국어 쿼리
    COUNTRY_PREFIXES = [
        "일본 ", "한국 ", "프랑스 ", "이탈리아 ", "스코틀랜드 ", "영국 ",
        "독일 ", "오스트리아 ", "스위스 ", "네덜란드 ", "중국 ", "대만 ",
        "포르투갈 ", "스페인 ", "노르웨이 ", "스웨덴 ", "아이슬란드 ",
        "뉴질랜드 ", "캐나다 ", "미국 ", "체코 ", "헝가리 ", "크로아티아 ",
    ]
    korean_query = re.sub(r'\(.*?\)', '', query).strip()
    for prefix in COUNTRY_PREFIXES:
        if korean_query.startswith(prefix):
            korean_query = korean_query[len(prefix):].strip()
            break
    # 불필요한 단어 추가 제거
    for word in ["원시림", "온천마을", "마을", "역", "항구", "구시가지"]:
        korean_query = korean_query.replace(word, "").strip()

    # 검색 우선순위: 영문명(EN) → 한글명(KO) → 한글명(EN)
    search_list: list[tuple[str, str]] = []
    if english_name:
        search_list.append(("en", english_name))
    if korean_query:
        search_list.append(("ko", korean_query))
    if korean_query and not english_name:
        search_list.append(("en", korean_query))

    for lang, q in search_list:
        result = _search_wiki_image(lang, q)
        if result:
            return result
    return None

def recommend(user_vibe: str):
    if not user_vibe or not user_vibe.strip():
        return (
            "⚠️ 입력 필요",
            "⚠️ 입력 필요",
            "원하는 여행 분위기를 텍스트 창에 입력해 주세요.",
            "<p style='color: #888;'>분위기 추천을 입력하시면 여기에 코스가 표시됩니다.</p>",
            None
        )
    
    try:
        client = InferenceClient(token=get_token(), timeout=15)
        response = client.chat_completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"내가 원하는 여행 분위기: {user_vibe}"}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        raw_result = response.choices[0].message.content
        result = parse_llm_output_safely(raw_result)
    except Exception as e:
        import traceback
        err_msg = f"Error during LLM call: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        # fallback
        
        # 에러 원인 분석 (토큰 누락 여부)
        env_keys = [k for k in os.environ.keys() if "HF" in k or "TOKEN" in k]
        is_token_missing = "HF_TOKEN" in str(e) or "환경변수가 비어 있습니다" in str(e)
        if is_token_missing:
            debug_info = (
                f"⚠️ [오류 안내: Hugging Face Space의 Settings에 API 토큰(HF_TOKEN)이 등록되지 않아 기본 추천지가 제공됩니다]\n"
                f"(현재 컨테이너 내부의 토큰 관련 환경변수 리스트: {env_keys})\n\n"
                "실시간 AI 추천 기능을 활성화하려면 다음 단계를 진행해 주세요:\n"
                "1. 본인의 Hugging Face Space 페이지 우상단의 'Settings' 메뉴 클릭\n"
                "2. 'Variables and secrets' 섹션으로 이동\n"
                "3. 'New secret' 버튼 클릭 → Name: 'HF_TOKEN', Value: 본인의 Hugging Face Write 토큰 값 입력\n\n"
            )
        else:
            debug_info = f"⚠️ [오류 안내: API 호출 중 오류가 발생하여 기본 추천지가 제공됩니다 (오류내용: {type(e).__name__} - {str(e)})]\n\n"
            
        result = {
            "destination": FALLBACK_RECOMMENDATION["destination"],
            "matching_ghibli_work": FALLBACK_RECOMMENDATION["matching_ghibli_work"],
            "vibe_comment": f"{debug_info}{FALLBACK_RECOMMENDATION['vibe_comment']}",
            "half_day_course": FALLBACK_RECOMMENDATION["half_day_course"]
        }
        
    destination = result.get("destination", "알 수 없는 목적지")
    ghibli_work = result.get("matching_ghibli_work", "지브리 작품")
    vibe_comment = result.get("vibe_comment", "")
    
    course_list = result.get("half_day_course", [])
    course_html = ""
    for item in course_list:
        step = item.get("step", 1)
        place = item.get("place", "")
        desc = item.get("description", "")
        
        course_html += f"""
        <div class="course-step">
            <div class="step-num">Step {step}</div>
            <div class="step-content">
                <h4 class="step-place">{place}</h4>
                <p class="step-desc">{desc}</p>
            </div>
        </div>
        """
    
    # 📸 목적지에 맞는 실제 이미지 가져오기 (Wikipedia → 코스 장소 → 지브리별 Fallback 순)
    image_result = None
    if destination and destination not in ("알 수 없는 목적지", "⚠️ 입력 필요"):
        try:
            # 1차: 목적지명으로 Wikipedia 검색
            image_result = get_real_image_wiki(destination)
        except Exception as img_err:
            print(f"Error fetching destination image: {img_err}")

    if not image_result and course_list:
        # 2차: 코스 장소들을 순서대로 시도 (영문명 포함 가능성 높음)
        for course_item in course_list:
            try:
                place_name = course_item.get("place", "")
                if place_name:
                    image_result = get_real_image_wiki(place_name)
                    if image_result:
                        break
            except Exception as img_err2:
                print(f"Error fetching course place image: {img_err2}")

    if not image_result:
        # 3차: 지브리 작품별 감성에 맞는 Fallback 이미지 선택
        image_result = _DEFAULT_FALLBACK_IMAGE
        for work_key, url in GHIBLI_FALLBACK_IMAGES.items():
            if work_key in ghibli_work:
                image_result = url
                break
            
    return destination, ghibli_work, vibe_comment, course_html, image_result

def build_ui() -> gr.Blocks:
    # 1. 지브리 테마 스타일 정의 (라이트 모드 강제 적용)
    theme = gr.themes.Base().set(
        body_background_fill="transparent",
        body_background_fill_dark="transparent",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        input_background_fill="transparent",
        input_background_fill_dark="transparent",
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        block_label_text_color="rgba(255,255,255,0.6)",
        block_label_text_color_dark="rgba(255,255,255,0.6)",
        button_primary_background_fill="linear-gradient(135deg, #56ab2f 0%, #2e7d32 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #56ab2f 0%, #2e7d32 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        border_color_primary="rgba(255,255,255,0.15)",
        border_color_primary_dark="rgba(255,255,255,0.15)",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;600;700&family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    /* ─── 전체 배경 ─── */
    .gradio-container {
        background: linear-gradient(160deg, #0d2b1a 0%, #1a4a2e 40%, #0f3624 100%) !important;
        min-height: 100vh !important;
        font-family: 'Outfit', 'Noto Sans KR', sans-serif !important;
    }
    body, .gradio-container * {
        box-sizing: border-box;
    }

    /* ─── 히어로 헤더 ─── */
    .title-section {
        text-align: center !important;
        padding: 3rem 2rem 2.5rem !important;
        background: linear-gradient(135deg,
            rgba(255,255,255,0.07) 0%,
            rgba(255,255,255,0.03) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1) !important;
        margin-bottom: 1.5rem !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .title-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 60% 40%, rgba(76,175,80,0.12) 0%, transparent 60%),
                    radial-gradient(ellipse at 20% 80%, rgba(212,167,87,0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .title-section h1, .title-section .prose h1, .title-section h1 * {
        color: #ffffff !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.6rem !important;
        text-shadow: 0 2px 20px rgba(76,175,80,0.4) !important;
        text-align: center !important;
    }
    .title-section p, .title-section .prose p, .title-section p * {
        color: rgba(255,255,255,0.75) !important;
        font-size: 1.05rem !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 0.3px !important;
        margin: 0 !important;
        text-align: center !important;
    }

    /* ─── 유리 카드 (glassmorphism) ─── */
    .main-box {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.08) !important;
        padding: 1.8rem !important;
        transition: box-shadow 0.3s ease !important;
    }
    .main-box:hover {
        box-shadow:
            0 12px 40px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1) !important;
    }

    /* ─── 텍스트 박스 ─── */
    textarea, input[type="text"] {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    textarea:focus, input[type="text"]:focus {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(76,175,80,0.6) !important;
        box-shadow: 0 0 0 3px rgba(76,175,80,0.15) !important;
        outline: none !important;
    }
    textarea::placeholder, input[type="text"]::placeholder {
        color: rgba(255,255,255,0.35) !important;
    }
    textarea[readonly], input[readonly] {
        background: rgba(255,255,255,0.05) !important;
        color: rgba(255,255,255,0.85) !important;
        border-color: rgba(255,255,255,0.1) !important;
        cursor: default !important;
    }

    /* ─── 라벨 텍스트 ─── */
    label, .label-wrap span, .block label span {
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }

    /* ─── 마크다운 헤딩 (결과 영역) ─── */
    .main-box .prose h3, .main-box h3 {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* ─── 버튼 ─── */
    .submit-btn {
        background: linear-gradient(135deg, #56ab2f 0%, #2e7d32 50%, #1b5e20 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.5px !important;
        padding: 0.9rem 2rem !important;
        box-shadow:
            0 4px 20px rgba(46,125,50,0.4),
            inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .submit-btn::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    .submit-btn:hover::before { left: 100%; }
    .submit-btn:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow:
            0 8px 30px rgba(46,125,50,0.5),
            inset 0 1px 0 rgba(255,255,255,0.25) !important;
    }
    .submit-btn:active {
        transform: translateY(-1px) !important;
    }

    /* ─── 어코디언 (가이드) ─── */
    .guide-accordion {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        margin-top: 0.8rem !important;
        margin-bottom: 1.2rem !important;
        backdrop-filter: blur(10px) !important;
    }
    .guide-accordion button {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.9rem !important;
    }
    .guide-accordion .prose p, .guide-accordion p {
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.9rem !important;
    }
    .guide-accordion .prose strong, .guide-accordion strong {
        color: rgba(255,255,255,0.85) !important;
    }

    /* ─── Examples 버튼 ─── */
    .examples-holder table td button,
    .gr-samples-table td button {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.75) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        font-size: 0.85rem !important;
    }
    .examples-holder table td button:hover,
    .gr-samples-table td button:hover {
        background: rgba(76,175,80,0.2) !important;
        border-color: rgba(76,175,80,0.4) !important;
        color: #ffffff !important;
    }

    /* ─── 감성 코스 타임라인 ─── */
    .course-step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.4rem;
        position: relative;
        animation: fadeInUp 0.4s ease both;
    }
    .course-step:nth-child(2) { animation-delay: 0.1s; }
    .course-step:nth-child(3) { animation-delay: 0.2s; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .course-step:not(:last-child)::after {
        content: '';
        position: absolute;
        left: 22px;
        top: 46px;
        bottom: -20px;
        width: 2px;
        background: linear-gradient(to bottom, rgba(76,175,80,0.6), rgba(76,175,80,0.1));
    }
    .step-num {
        background: linear-gradient(135deg, #56ab2f, #2e7d32) !important;
        color: white !important;
        font-weight: 700 !important;
        width: 44px !important;
        height: 44px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.85rem !important;
        flex-shrink: 0 !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.4), 0 0 0 3px rgba(76,175,80,0.15) !important;
        z-index: 2 !important;
        position: relative !important;
    }
    .step-content {
        margin-left: 1.2rem !important;
        background: rgba(255,255,255,0.06) !important;
        padding: 1rem 1.4rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        flex-grow: 1 !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
    }
    .step-content:hover {
        transform: translateX(6px) !important;
        background: rgba(255,255,255,0.09) !important;
        border-color: rgba(76,175,80,0.35) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
    }
    .step-place {
        color: #7dde8b !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.3rem 0 !important;
        font-family: 'Noto Serif KR', serif !important;
    }
    .step-desc {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
        line-height: 1.6 !important;
    }

    /* ─── 섹션 헤딩 마크다운 ─── */
    .prose h3 { color: rgba(255,255,255,0.88) !important; }

    /* ─── 이미지 영역 ─── */
    .main-box img {
        border-radius: 14px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4) !important;
    }

    /* ─── 스크롤바 ─── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
    ::-webkit-scrollbar-thumb { background: rgba(76,175,80,0.4); border-radius: 3px; }

    /* ─── 숨김 처리 ─── */
    .meta-text, [class*="meta-text"], .timer, .duration,
    .eta-bar, [class*="eta-bar"], [class*="timer"] {
        display: none !important;
    }
    """
    
    with gr.Blocks(css=css, theme=theme, title="지브리 감성 여행지 추천 서비스") as demo:
        with gr.Group(elem_classes=["title-section"]):
            gr.Markdown("# 🌿 Ghibli-Vibe Travel Mapper")
            gr.Markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 1.05rem; font-weight: 300; letter-spacing: 0.3px;'>원하는 여행 분위기를 일상어로 쓰시면, 지브리 감성이 가득한 실제 여행지와 반나절 코스를 추천해 드립니다.</p>")
            
        with gr.Accordion("💡 Ghibli Travel Mapper 사용 가이드 (클릭하여 열기)", open=False, elem_classes=["guide-accordion"]):
            gr.Markdown("""
**✨ 이렇게 사용하세요:**
1. 원하는 **여행 분위기나 테마**를 자유롭게 입력하세요
2. **🌿 감성 여행지 찾기** 버튼을 누르세요
3. AI가 추천하는 **지브리 감성 여행지 + 반나절 코스**를 확인하세요

**💬 예시 키워드:** 조용한 숲속 오두막, 바다가 보이는 레트로 골목길, 안개 낀 유럽 산골 마을...
            """)
            
        # 2. 상단 입력 영역 (가운데 정렬된 아담한 폭의 카드)
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):  # 모바일에서는 꽉 차고, PC에서는 가로폭 50%를 차지하는 카드
                with gr.Group(elem_classes=["main-box"]):
                    user_input = gr.Textbox(
                        label="원하는 분위기 또는 테마 입력",
                        placeholder="예: 바다가 보이는 조용한 레트로 골목길 / 초록빛 숲속과 신비로운 분위기가 가득한 조용한 오두막",
                        lines=3
                    )
                    submit_btn = gr.Button("🌿 감성 여행지 찾기", elem_classes=["submit-btn"])
                    gr.Examples(
                        examples=[
                            ["바다가 보이는 조용한 레트로 골목길"],
                            ["초록빛 숲속과 신비로운 분위기가 가득한 조용한 오두막"],
                            ["빨간 지붕이 있고 구름이 흐르는 청량한 유럽풍 하늘 아래 마을"],
                            ["복잡한 도심 속 숨겨진 신비로운 전통 정원"]
                        ],
                        inputs=user_input,
                        label="💡 추천 분위기 예시 (클릭하면 자동 입력됩니다)"
                    )
            with gr.Column(scale=1):
                pass

        # 3. 하단 결과 영역 (가로폭을 넓게 쓰는 시원한 레이아웃 - 좌측 이미지, 우측 텍스트)
        with gr.Row():
            with gr.Column(scale=5):  # 좌측: 실제 감성 풍경화
                with gr.Group(elem_classes=["main-box"]):
                    gr.Markdown("### 🎨 감성 풍경화")
                    image_out = gr.Image(label="감성 풍경화", interactive=False, show_label=False)
                    
            with gr.Column(scale=7):  # 우측: AI 추천 여행지 큐레이션 및 타임라인
                with gr.Group(elem_classes=["main-box"]):
                    gr.Markdown("### 📍 추천 결과")
                    with gr.Row():
                        dest_out = gr.Textbox(label="추천 목적지", interactive=False)
                        work_out = gr.Textbox(label="매칭된 지브리 작품", interactive=False)
                    vibe_out = gr.Textbox(label="큐레이터의 감성 코멘트", lines=3, interactive=False)
                    
                    gr.Markdown("### 🗺️ 반나절 감성 동선 (Course)")
                    course_out = gr.HTML(label="추천 코스")
                    
        submit_btn.click(
            fn=recommend,
            inputs=user_input,
            outputs=[dest_out, work_out, vibe_out, course_out, image_out]
        )
        
    return demo

demo = build_ui()

if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        show_error=True,
    )
