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
    "4. 이 분위기에 어울리는 낭만적이고 직관적인 영문 감성 라벨(vibe_name)을 정한다. (예: Ocean Waves Vibe, Forest Whispers Vibe, Emerald Valley Vibe, Sunset Breeze Vibe 등)\n"
    "5. 이 분위기와 추천 장소와의 매칭률(match_percentage)을 90에서 98 사이의 정수형 숫자로 나타낸다.\n"
    "6. 이 장소가 선정된 이유와 지브리 작품의 어떤 장면/감성과 연결되는지 따뜻하고 감성적인 어조로 설명하는 '감성 코멘트'를 작성한다.\n"
    "7. 추천된 장소를 중심으로 도보 또는 가볍게 이동 가능한 3단계 반나절 여행 동선(코스)을 기획한다. 각 단계는 장소 이름과 그곳에서 느낄 수 있는 감성적 활동을 포함해야 한다.\n\n"
    "반드시 아래 JSON 스키마 형식으로만 응답해야 하며, 다른 여담이나 설명, markdown 코드 블록 기호(```json 등)는 절대 포함하지 마라. JSON 객체로만 출력하라.\n"
    "{\n"
    '  "destination": "추천 목적지 한국어명과 영문명. 반드시 영문명을 괄호 () 안에 포함할 것 (예: 일본 가마쿠라 (Kamakura, Japan))",\n'
    '  "vibe_name": "이 분위기에 어울리는 낭만적인 감성 라벨 영문명 (예: Ocean Waves Vibe)",\n'
    '  "match_percentage": 95,\n'
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
    "vibe_name": "Forest Whispers Vibe",
    "match_percentage": 95,
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


# ── 이미지 파일명 기반 풍경 사진 판별 ──
_EXCLUDE_IMG_KW = [
    'food', 'cuisine', 'dish', 'meal', 'restaurant', 'cooking', 'recipe',
    'sushi', 'ramen', 'noodle', 'rice', 'bread', 'cake', 'dessert', 'drink',
    'portrait', 'people', 'face', 'person', 'headshot', 'selfie',
    'flag', 'map', 'coat', 'arms', 'logo', 'seal', 'icon', 'symbol',
    'emblem', 'stamp', 'chart', 'graph', 'sign', 'label', 'menu',
    'interior', 'room', 'inside', 'ceiling', 'floor',
    'wikipedia', 'commons-logo', 'wikidata', 'openstreetmap',
]
_PREFER_IMG_KW = [
    'panorama', 'panoramic', 'aerial', 'view', 'scenic', 'scenery', 'landscape',
    'skyline', 'overview', 'vista', 'horizon', 'sunset', 'sunrise', 'twilight',
    'castle', 'temple', 'shrine', 'palace', 'ruins', 'monument', 'tower', 'arch',
    'forest', 'mountain', 'valley', 'river', 'lake', 'sea', 'coast', 'beach',
    'ocean', 'waterfall', 'canyon', 'field', 'meadow', 'snow', 'glacier',
    'town', 'village', 'street', 'alley', 'bridge', 'district', 'quarter',
    'garden', 'park', 'trail', 'path', 'road',
]

def _is_good_landscape(fname: str) -> tuple[bool, bool]:
    """(허용 여부, 선호 여부) 반환"""
    lower = fname.lower()
    if not any(lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
        return False, False
    for kw in _EXCLUDE_IMG_KW:
        if kw in lower:
            return False, False
    for kw in _PREFER_IMG_KW:
        if kw in lower:
            return True, True
    return True, False


# 위키백과 API 재단 정책(User-Agent Policy)에 맞춘 식별 문자열 선언 (Mozilla/5.0 등 범용 사용 시 429 차단 방지)
WIKI_USER_AGENT = "GhibliVibeTravelMapper/1.0 (yuhj812@naver.com; course project)"


def _get_landscape_from_article(lang: str, page_title: str, width: int = 800) -> str | None:
    """Wikipedia 문서에서 풍경 사진 URL을 찾는다."""
    try:
        # 1. 문서 내 모든 이미지 파일명 조회
        images_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(page_title)}"
            f"&prop=images&imlimit=30&format=json"
        )
        req = urllib.request.Request(images_url, headers={'User-Agent': WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            preferred, general = [], []
            for _, page_data in pages.items():
                for img in page_data.get("images", []):
                    fname = img.get("title", "")
                    allowed, preferred_flag = _is_good_landscape(fname)
                    if allowed:
                        (preferred if preferred_flag else general).append(fname)

        # 선호 → 일반 순, 최대 4개만 시도
        candidates = (preferred + general)[:4]
        if not candidates:
            return None

        # 2. 파일 URL 조회 (imageinfo API)
        titles_param = "|".join(urllib.parse.quote(f) for f in candidates)
        info_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&titles={titles_param}"
            f"&prop=imageinfo&iiprop=url&iiurlwidth={width}&format=json"
        )
        req = urllib.request.Request(info_url, headers={'User-Agent': WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for _, page_data in pages.items():
                for info in page_data.get("imageinfo", []):
                    url = info.get("thumburl") or info.get("url")
                    if url and url.startswith("http"):
                        return url
    except Exception as e:
        print(f"Landscape image search failed ({lang}, '{page_title}'): {e}")
    return None


def _search_wiki_image(lang: str, q: str) -> str | None:
    """Wikipedia 검색 → 풍경 사진 URL 반환. 없으면 None."""
    try:
        # 1. Wikipedia 검색으로 문서 제목 가져오기
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(q)}&format=json"
        )
        req = urllib.request.Request(search_url, headers={'User-Agent': WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get("query", {}).get("search", [])
            if not results:
                return None
            page_title = results[0]["title"]

        # 2. 풍경 사진 우선 검색 (이미지 필터링 적용)
        landscape_url = _get_landscape_from_article(lang, page_title)
        if landscape_url:
            return landscape_url

        # 3. 풍경 사진 없으면 pageimage(대표 이미지) fallback
        img_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(page_title)}"
            f"&prop=pageimages&format=json&pithumbsize=800"
        )
        req = urllib.request.Request(img_url, headers={'User-Agent': WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for _, page_data in pages.items():
                source = page_data.get("thumbnail", {}).get("source")
                if source:
                    return source  # validation 없이 바로 반환 (Wikimedia CDN 특성상 HEAD 요청 불가)
    except Exception as e:
        print(f"Wikipedia image search failed ({lang}, '{q}'): {e}")
    return None


def get_real_image_wiki(query: str) -> str | None:
    """목적지 쿼리로 Wikipedia 풍경 이미지를 검색한다.
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


def get_watercolor_fallback(place_name: str) -> str:
    name_lower = place_name.lower()
    if any(x in name_lower for x in ["temple", "shrine", "gate", "pagoda", "wat", "ji", "절", "사원", "신사", "사찰", "대불", "성당", "교회"]):
        return "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=800"
    if any(x in name_lower for x in ["beach", "sea", "ocean", "coast", "shore", "water", "bay", "port", "harbor", "바다", "해변", "해수욕장", "연안", "포구", "항구"]):
        return "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=800"
    if any(x in name_lower for x in ["forest", "park", "mountain", "wood", "garden", "hill", "valley", "lake", "pond", "숲", "공원", "산", "정원", "동산", "계곡", "호수", "연못"]):
        return "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800"
    if any(x in name_lower for x in ["street", "market", "alley", "road", "dori", "town", "village", "shop", "cafe", "station", "거리", "시장", "골목", "마을", "상점", "카페", "역"]):
        return "https://images.unsplash.com/photo-1528164344705-47542687000d?w=800"
    return "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=800"


def get_empty_state_html() -> str:
    return """
    <div class="empty-state-container">
        <div class="empty-state-box">
            <div class="empty-icon">🍃</div>
            <h3>어떤 분위기의 여행을 떠나고 싶으신가요?</h3>
            <p>위의 검색창에 조용한 숲속 마을, 바다가 보이는 골목길 등 원하시는 분위기를 자유롭게 적어 보세요.</p>
        </div>
    </div>
    """


def recommend(user_vibe: str) -> str:
    if not user_vibe or not user_vibe.strip():
        return get_empty_state_html()
    
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
            "vibe_name": FALLBACK_RECOMMENDATION["vibe_name"],
            "match_percentage": FALLBACK_RECOMMENDATION["match_percentage"],
            "matching_ghibli_work": FALLBACK_RECOMMENDATION["matching_ghibli_work"],
            "vibe_comment": f"{debug_info}{FALLBACK_RECOMMENDATION['vibe_comment']}",
            "half_day_course": FALLBACK_RECOMMENDATION["half_day_course"]
        }
        
    destination = result.get("destination", "알 수 없는 목적지")
    vibe_name = result.get("vibe_name", "Ghibli Vibe")
    match_percentage = result.get("match_percentage", 95)
    ghibli_work = result.get("matching_ghibli_work", "지브리 작품")
    vibe_comment = result.get("vibe_comment", "")
    course_list = result.get("half_day_course", [])
    
    # 📸 목적지에 맞는 실제 이미지 가져오기
    main_image = None
    if destination and destination not in ("알 수 없는 목적지", "⚠️ 입력 필요"):
        try:
            main_image = get_real_image_wiki(destination)
        except Exception as img_err:
            print(f"Error fetching destination image: {img_err}")

    if not main_image and course_list:
        for course_item in course_list:
            try:
                place_name = course_item.get("place", "")
                if place_name:
                    main_image = get_real_image_wiki(place_name)
                    if main_image:
                        break
            except Exception as img_err2:
                print(f"Error fetching course place image: {img_err2}")

    if not main_image:
        main_image = _DEFAULT_FALLBACK_IMAGE
        for work_key, url in GHIBLI_FALLBACK_IMAGES.items():
            if work_key in ghibli_work:
                main_image = url
                break
                
    # 📸 각 코스 단계의 이미지 가져오기
    step_images = []
    for step_item in course_list:
        place_name = step_item.get("place", "")
        img_url = None
        if place_name:
            try:
                img_url = get_real_image_wiki(place_name)
            except Exception as e:
                print(f"Error fetching image for step place '{place_name}': {e}")
        if not img_url:
            img_url = get_watercolor_fallback(place_name)
        step_images.append(img_url)

    # 코스 스텝 HTML 렌더링
    steps_html = ""
    for idx, item in enumerate(course_list):
        step = item.get("step", idx + 1)
        place = item.get("place", "")
        desc = item.get("description", "")
        img = step_images[idx] if idx < len(step_images) else get_watercolor_fallback(place)
        
        display_place = re.sub(r'\(.*?\)', '', place).strip()
        
        arrow_html = ""
        if idx < len(course_list) - 1:
            arrow_html = """
            <div class="step-arrow">
                <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="#a4b494" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
            </div>
            """
            
        steps_html += f"""
        <div class="step-col animate-step" style="animation-delay: {idx * 0.15}s;">
            <div class="step-img-wrapper">
                <img src="{img}" alt="{display_place}" class="step-circle-img" />
            </div>
            <h3 class="step-title">Step {step}: {display_place}</h3>
            <p class="step-desc">{desc}</p>
        </div>
        {arrow_html}
        """

    result_html = f"""
    <div class="result-container animate-fade-in">
        <!-- Recommendation Card -->
        <div class="recommendation-card">
            <div class="rec-left">
                <div class="circle-image-wrapper">
                    <img src="{main_image}" alt="{destination}" class="circle-img" />
                </div>
                <div class="vibe-label">{vibe_name}</div>
                <div class="match-badge">Match: {match_percentage}%</div>
            </div>
            <div class="rec-right">
                <div class="rec-badge-wrap">
                    <span class="rec-badge">Recommendation</span>
                </div>
                <h2 class="rec-title">{destination}</h2>
                <p class="rec-work-info">🎬 매칭 지브리 작품: <strong>{ghibli_work}</strong></p>
                <p class="rec-desc">{vibe_comment}</p>
            </div>
        </div>

        <!-- Timeline Section -->
        <div class="timeline-section">
            <div class="timeline-badge-wrap">
                <span class="timeline-badge">3-step walking timeline course</span>
            </div>
            <div class="timeline-card">
                <!-- Leafy corner illustrations -->
                <div class="leaf-corner tl">
                    <svg viewBox="0 0 100 100" width="80" height="80">
                        <path d="M 10 10 C 25 15, 35 25, 40 45 C 38 48, 30 40, 25 32 C 15 22, 10 15, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 15 25, 25 35, 45 40 C 48 38, 40 30, 32 25 C 22 15, 15 10, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 20 20, 30 30, 35 35" stroke="#3d2b1f" stroke-width="2" fill="none"/>
                        <circle cx="28" cy="18" r="3" fill="#a4b494"/>
                        <circle cx="18" cy="28" r="3" fill="#a4b494"/>
                    </svg>
                </div>
                <div class="leaf-corner tr">
                    <svg viewBox="0 0 100 100" width="80" height="80" style="transform: scaleX(-1);">
                        <path d="M 10 10 C 25 15, 35 25, 40 45 C 38 48, 30 40, 25 32 C 15 22, 10 15, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 15 25, 25 35, 45 40 C 48 38, 40 30, 32 25 C 22 15, 15 10, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 20 20, 30 30, 35 35" stroke="#3d2b1f" stroke-width="2" fill="none"/>
                        <circle cx="28" cy="18" r="3" fill="#a4b494"/>
                        <circle cx="18" cy="28" r="3" fill="#a4b494"/>
                    </svg>
                </div>
                <div class="leaf-corner bl">
                    <svg viewBox="0 0 100 100" width="80" height="80" style="transform: scaleY(-1);">
                        <path d="M 10 10 C 25 15, 35 25, 40 45 C 38 48, 30 40, 25 32 C 15 22, 10 15, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 15 25, 25 35, 45 40 C 48 38, 40 30, 32 25 C 22 15, 15 10, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 20 20, 30 30, 35 35" stroke="#3d2b1f" stroke-width="2" fill="none"/>
                        <circle cx="28" cy="18" r="3" fill="#a4b494"/>
                        <circle cx="18" cy="28" r="3" fill="#a4b494"/>
                    </svg>
                </div>
                <div class="leaf-corner br">
                    <svg viewBox="0 0 100 100" width="80" height="80" style="transform: scale(-1);">
                        <path d="M 10 10 C 25 15, 35 25, 40 45 C 38 48, 30 40, 25 32 C 15 22, 10 15, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 15 25, 25 35, 45 40 C 48 38, 40 30, 32 25 C 22 15, 15 10, 10 10 Z" fill="#849b73" opacity="0.85"/>
                        <path d="M 10 10 C 20 20, 30 30, 35 35" stroke="#3d2b1f" stroke-width="2" fill="none"/>
                        <circle cx="28" cy="18" r="3" fill="#a4b494"/>
                        <circle cx="18" cy="28" r="3" fill="#a4b494"/>
                    </svg>
                </div>
                
                <!-- Totoro spirits -->
                <div class="totoro-spirit top-right">
                    <svg viewBox="0 0 50 50" width="45" height="45">
                        <path d="M 25 10 C 22 10, 21 13, 21 16 C 18 17, 15 21, 15 26 C 15 32, 19 36, 25 36 C 31 36, 35 32, 35 26 C 35 21, 32 17, 29 16 C 29 13, 28 10, 25 10 Z" fill="#788b6c"/>
                        <path d="M 21 16 L 22 10 L 23 15 M 29 16 L 28 10 L 27 15" stroke="#788b6c" stroke-width="2" stroke-linecap="round"/>
                        <circle cx="22" cy="22" r="1.5" fill="#ffffff"/>
                        <circle cx="28" cy="22" r="1.5" fill="#ffffff"/>
                        <circle cx="22" cy="22" r="0.5" fill="#000000"/>
                        <circle cx="28" cy="22" r="0.5" fill="#000000"/>
                    </svg>
                </div>
                <div class="totoro-spirit bottom-center">
                    <svg viewBox="0 0 100 50" width="90" height="45">
                        <g transform="translate(10, 5)">
                            <path d="M 25 10 C 22 10, 21 13, 21 16 C 18 17, 15 21, 15 26 C 15 32, 19 36, 25 36 C 31 36, 35 32, 35 26 C 35 21, 32 17, 29 16 C 29 13, 28 10, 25 10 Z" fill="#889c7c"/>
                            <path d="M 21 16 L 22 10 L 23 15 M 29 16 L 28 10 L 27 15" stroke="#889c7c" stroke-width="2" stroke-linecap="round"/>
                            <circle cx="22" cy="22" r="1.5" fill="#ffffff"/>
                            <circle cx="28" cy="22" r="1.5" fill="#ffffff"/>
                        </g>
                        <g transform="translate(45, 12) scale(0.8)">
                            <path d="M 25 10 C 22 10, 21 13, 21 16 C 18 17, 15 21, 15 26 C 15 32, 19 36, 25 36 C 31 36, 35 32, 35 26 C 35 21, 32 17, 29 16 C 29 13, 28 10, 25 10 Z" fill="#98ac8c"/>
                            <path d="M 21 16 L 22 10 L 23 15 M 29 16 L 28 10 L 27 15" stroke="#98ac8c" stroke-width="2" stroke-linecap="round"/>
                            <circle cx="22" cy="22" r="1.5" fill="#ffffff"/>
                            <circle cx="28" cy="22" r="1.5" fill="#ffffff"/>
                        </g>
                    </svg>
                </div>

                <div class="steps-container">
                    {steps_html}
                </div>
            </div>
        </div>
    </div>
    """
    return result_html


def build_ui() -> gr.Blocks:
    # 1. 지브리 테마 스타일 정의 (라이트 모드 강제 적용)
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="yellow",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Noto Serif KR"), gr.themes.GoogleFont("Noto Sans KR"), "sans-serif"],
    ).set(
        body_background_fill="#fcfaf2",
        body_background_fill_dark="#fcfaf2",
        block_background_fill="#fffdf5",
        block_background_fill_dark="#fffdf5",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        body_text_color="#3d2b1f",
        body_text_color_dark="#3d2b1f",
        block_label_text_color="#5a7a4a",
        block_label_text_color_dark="#5a7a4a",
        button_primary_background_fill="linear-gradient(135deg, #ebdcb9 0%, #e2d4b7 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #ebdcb9 0%, #e2d4b7 100%)",
        button_primary_text_color="#3d2b1f",
        button_primary_text_color_dark="#3d2b1f",
        border_color_primary="#a3b899",
        border_color_primary_dark="#a3b899",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;500;600;700&family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');

    /* --- Base reset & Background --- */
    .gradio-container {
        background-color: #fcfaf2 !important;
        background-image:
            radial-gradient(ellipse at 15% 20%, rgba(168,197,160,0.15) 0%, transparent 45%),
            radial-gradient(ellipse at 85% 75%, rgba(200,168,75,0.08) 0%, transparent 40%),
            radial-gradient(ellipse at 50% 50%, rgba(252,250,242,0.8) 0%, transparent 100%) !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        min-height: 100vh !important;
        padding: 0 1rem 3rem !important;
    }
    
    /* Hide default Gradio headers or elements that clutter */
    .meta-text, [class*="meta-text"], .timer, .duration,
    .eta-bar, [class*="eta-bar"], [class*="timer"] {
        display: none !important;
    }
    
    /* Remove Gradio card wrappers default borders/shadows to let custom CSS shine */
    .main-box, .gradio-container .block {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* --- Custom Ghibli Header --- */
    .ghibli-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid rgba(164, 180, 148, 0.2);
        margin-bottom: 2.5rem;
    }
    .ghibli-logo {
        font-family: 'Playfair Display', 'Noto Serif KR', serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: #3d2b1f;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .home-icon {
        font-size: 1.5rem;
    }
    .ghibli-nav {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    .nav-item {
        color: #5a4c40;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 6px 12px;
    }
    .nav-item:hover {
        color: #3d5a2d;
    }
    .nav-item.active {
        background-color: #d5e5cf;
        color: #3b5e2f;
        border-radius: 20px;
        font-weight: 600;
    }

    /* --- Custom Search Bar Row --- */
    .search-row {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 15px !important;
        max-width: 800px;
        margin: 0 auto 3rem !important;
        background: transparent !important;
        width: 100% !important;
    }
    .search-input {
        background: transparent !important;
    }
    .search-input textarea, .search-input input[type="text"] {
        background: #ffffff !important;
        color: #3d2b1f !important;
        border: 2px solid #a3b899 !important;
        border-radius: 20px !important;
        padding: 12px 24px !important;
        font-size: 1.1rem !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02), 0 4px 12px rgba(163, 184, 153, 0.1) !important;
        transition: all 0.3s ease !important;
        height: 60px !important;
        line-height: 36px !important;
        min-height: 60px !important;
    }
    .search-input textarea:focus, .search-input input[type="text"]:focus {
        border-color: #6aaa4a !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02), 0 6px 20px rgba(106, 170, 74, 0.2) !important;
    }
    
    .search-btn {
        width: 90px !important;
        height: 90px !important;
        min-width: 90px !important;
        border-radius: 50% !important;
        background-color: #ebdcb9 !important;
        border: 2px solid #b8a68a !important;
        color: #3d2b1f !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80' width='80' height='80'><path d='M36 24 C33 21, 37 19, 40 21 C43 19, 47 21, 44 24 C41 27, 38 27, 36 24 Z' fill='%23849b73' stroke='%233d2b1f' stroke-width='1.5'/><path d='M46 22 C44 20, 47 18, 49 19 C51 18, 54 20, 52 22 C50 24, 48 24, 46 22 Z' fill='%23a4b494' stroke='%233d2b1f' stroke-width='1.2'/><path d='M 25 55 Q 40 68 55 55 M 51 58 L 55 55 L 53 51' stroke='%233d2b1f' stroke-width='2' fill='none' stroke-linecap='round'/></svg>") !important;
        background-repeat: no-repeat !important;
        background-position: center bottom 10px !important;
        background-size: 50px !important;
        padding-top: 10px !important;
        padding-bottom: 35px !important;
    }
    .search-btn:hover {
        transform: scale(1.05) rotate(3deg) !important;
        background-color: #e2d4b7 !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
    }
    .search-btn:active {
        transform: scale(0.95) !important;
    }
    
    /* --- Examples Area --- */
    .examples-row {
        margin: -1.5rem auto 3rem !important;
        max-width: 800px !important;
        text-align: center !important;
    }
    .examples-row table td button {
        background: #f3edd8 !important;
        color: #5a4c40 !important;
        border: 1px solid #dcd3be !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        padding: 6px 14px !important;
        transition: all 0.2s ease !important;
    }
    .examples-row table td button:hover {
        background: #e2d4b7 !important;
        border-color: #b8a68a !important;
        transform: translateY(-2px) !important;
        color: #3d2b1f !important;
    }

    /* --- Empty State --- */
    .empty-state-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 5rem 2rem;
        background: #fffdf9;
        border: 2px dashed #d5e5cf;
        border-radius: 24px;
        max-width: 900px;
        margin: 0 auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    }
    .empty-state-box {
        text-align: center;
    }
    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: floatLeaf 3s ease-in-out infinite;
    }
    .empty-state-box h3 {
        font-family: 'Noto Serif KR', serif;
        color: #3d2b1f;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    .empty-state-box p {
        color: #8a7c70;
        font-size: 0.95rem;
    }
    
    @keyframes floatLeaf {
        0%, 100% { transform: translateY(0) rotate(0); }
        50% { transform: translateY(-10px) rotate(10deg); }
    }

    /* --- Results Container --- */
    .result-container {
        max-width: 900px;
        margin: 0 auto;
    }
    .animate-fade-in {
        animation: fadeIn 0.6s ease both;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- Recommendation Card --- */
    .recommendation-card {
        display: flex;
        gap: 3rem;
        background: transparent;
        margin-bottom: 3.5rem;
        align-items: center;
    }
    .rec-left {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 240px;
    }
    .circle-image-wrapper {
        width: 210px;
        height: 210px;
        border-radius: 50%;
        overflow: hidden;
        border: 3.5px solid #a4b494;
        box-shadow: 0 8px 25px rgba(164, 180, 148, 0.3), 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        position: relative;
    }
    .circle-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    .circle-image-wrapper:hover .circle-img {
        transform: scale(1.08);
    }
    .vibe-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #3d2b1f;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .match-badge {
        font-family: 'Playfair Display', 'Noto Serif KR', serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #7b6a5a;
        text-align: center;
    }
    .rec-right {
        flex-grow: 1;
    }
    .rec-badge-wrap {
        margin-bottom: 0.6rem;
    }
    .rec-badge {
        background-color: #fde8db;
        color: #c85a25;
        font-family: 'Playfair Display', 'Noto Serif KR', serif;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .rec-title {
        font-family: 'Noto Serif KR', serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #2b261f;
        margin: 0.4rem 0 0.8rem;
    }
    .rec-work-info {
        font-size: 0.92rem;
        color: #5a4c40;
        margin-bottom: 0.8rem;
    }
    .rec-work-info strong {
        color: #3b5e2f;
    }
    .rec-desc {
        font-family: 'Noto Serif KR', serif;
        color: #4c3f35;
        font-size: 1.05rem;
        line-height: 1.75;
        margin: 0;
    }

    /* --- Timeline Section --- */
    .timeline-section {
        margin-top: 2rem;
        position: relative;
    }
    .timeline-badge-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: -15px;
        position: relative;
        z-index: 10;
    }
    .timeline-badge {
        background-color: #d5e5cf;
        color: #3b5e2f;
        font-family: 'Playfair Display', 'Noto Serif KR', serif;
        font-weight: 700;
        font-size: 1rem;
        padding: 8px 20px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(106, 170, 74, 0.1);
    }
    .timeline-card {
        background-color: #fffefb;
        border: 2px solid #a4b494;
        border-radius: 30px;
        padding: 3.5rem 2.5rem 3rem;
        position: relative;
        box-shadow: 0 8px 30px rgba(164, 180, 148, 0.15);
    }
    
    /* Vine Corner decorations */
    .leaf-corner {
        position: absolute;
        pointer-events: none;
    }
    .leaf-corner.tl { top: -8px; left: -8px; }
    .leaf-corner.tr { top: -8px; right: -8px; }
    .leaf-corner.bl { bottom: -8px; left: -8px; }
    .leaf-corner.br { bottom: -8px; right: -8px; }
    
    /* Totoro Spirit Decorations */
    .totoro-spirit {
        position: absolute;
        pointer-events: none;
    }
    .totoro-spirit.top-right {
        top: -15px;
        right: 40px;
    }
    .totoro-spirit.bottom-center {
        bottom: -15px;
        left: 50%;
        transform: translateX(-50%);
    }

    /* Steps Layout (Horizontal) */
    .steps-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .step-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        max-width: 240px;
    }
    .step-img-wrapper {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        overflow: hidden;
        border: 3.5px solid #d5e5cf;
        box-shadow: 0 6px 16px rgba(164, 180, 148, 0.15);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    .step-col:hover .step-img-wrapper {
        border-color: #a4b494;
        transform: scale(1.05);
    }
    .step-circle-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .step-title {
        font-family: 'Playfair Display', 'Noto Serif KR', serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #2b261f;
        margin: 0 0 0.5rem 0;
        line-height: 1.4;
    }
    .step-desc {
        font-size: 0.88rem;
        color: #6a5a4c;
        line-height: 1.6;
        margin: 0;
        font-family: 'Noto Sans KR', sans-serif;
    }
    .step-arrow {
        align-self: center;
        margin-top: -40px;
        opacity: 0.7;
    }

    /* Animating Step entry */
    .animate-step {
        animation: stepRise 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }
    @keyframes stepRise {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* --- Responsive design for mobile --- */
    @media (max-width: 768px) {
        .ghibli-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        .search-row {
            flex-direction: column !important;
            gap: 12px !important;
        }
        .search-input {
            width: 100% !important;
        }
        .search-btn {
            width: 80px !important;
            height: 80px !important;
            min-width: 80px !important;
            padding-bottom: 28px !important;
            background-size: 42px;
        }
        .recommendation-card {
            flex-direction: column;
            gap: 1.5rem;
            text-align: center;
        }
        .rec-left {
            width: 100%;
        }
        .steps-container {
            flex-direction: column;
            align-items: center;
            gap: 2.5rem;
        }
        .step-col {
            max-width: 100%;
        }
        .step-arrow {
            transform: rotate(90deg);
            margin: -10px 0;
        }
    }
    """
    
    with gr.Blocks(css=css, theme=theme, title="지브리 감성 여행지 추천 서비스") as demo:
        # Header
        with gr.Row(elem_classes=["header-row"]):
            gr.HTML("""
            <div class="ghibli-header">
                <div class="ghibli-logo">
                    <span class="home-icon">🏡</span>
                    Ghibli-Vibe Travel Mapper
                </div>
                <div class="ghibli-nav">
                    <a href="#" class="nav-item active">Home</a>
                    <a href="#" class="nav-item">Explore Vibes</a>
                    <a href="#" class="nav-item">My Saved Trips</a>
                    <a href="#" class="nav-item">About</a>
                </div>
            </div>
            """)
            
        # Search Box Input Row
        with gr.Row(elem_classes=["search-row"]):
            user_input = gr.Textbox(
                show_label=False,
                placeholder="어떤 분위기의 여행을 떠나고 싶으신가요? (예: 바다가 보이는 조용한 시골 마을)",
                elem_classes=["search-input"],
                scale=8
            )
            submit_btn = gr.Button(
                "Search",
                elem_classes=["search-btn"],
                scale=1
            )
            
        # Examples
        with gr.Row(elem_classes=["examples-row"]):
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

        # Output Results Container
        with gr.Row(elem_classes=["results-wrapper"]):
            with gr.Column():
                result_out = gr.HTML(value=get_empty_state_html())
                    
        submit_btn.click(
            fn=recommend,
            inputs=user_input,
            outputs=result_out
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
