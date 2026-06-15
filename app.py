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
    "4. 이 장소가 선정된 이유와 지브리 작품의 어떤 장면/감성과 연결되는지 따뜻하고 감성적인 어조로 설명하는 '감성 코멘트'를 작성한다.\n"
    "5. 추천된 장소를 중심으로 도보 또는 가볍게 이동 가능한 3단계 반나절 여행 동선(코스)을 기획한다. 각 단계는 장소 이름과 그곳에서 느낄 수 있는 감성적 활동을 포함해야 한다.\n\n"
    "반드시 아래 JSON 스키마 형식으로만 응답해야 하며, 다른 여담이나 설명, markdown 코드 블록 기호(```json 등)는 절대 포함하지 마라. JSON 객체로만 출력하라.\n"
    "{\n"
    '  "destination": "추천 목적지 이름 (예: 일본 가마쿠라 고쿠라쿠지 역)",\n'
    '  "matching_ghibli_work": "매칭되는 지브리 애니메이션 제목 (예: 이웃집 토토로)",\n'
    '  "vibe_comment": "따뜻하고 서정적인 어조의 감성 코멘트 (2-3문장)",\n'
    '  "half_day_course": [\n'
    "    {\n"
    '      "step": 1,\n'
    '      "place": "코스 1의 장소 이름",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    },\n"
    "    {\n"
    '      "step": 2,\n'
    '      "place": "코스 2의 장소 이름",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    },\n"
    "    {\n"
    '      "step": 3,\n'
    '      "place": "코스 3의 장소 이름",\n'
    '      "description": "그곳에서 할 감성적인 활동이나 느낄 수 있는 분위기 설명"\n'
    "    }\n"
    "  ]\n"
    "}"
)

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
    JsonOutputParser가 가끔 실패하거나 문자열 그대로 넘어오는 경우를 대비하여
    정규표현식이나 추가 처리를 통해 안전하게 JSON 데이터를 파싱합니다.
    """
    if isinstance(raw_output, dict):
        return raw_output
    
    if isinstance(raw_output, str):
        cleaned = raw_output.strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    raise ValueError("Invalid JSON output")

def recommend(user_vibe: str):
    if not user_vibe or not user_vibe.strip():
        return (
            "⚠️ 입력 필요",
            "⚠️ 입력 필요",
            "원하는 여행 분위기를 텍스트 창에 입력해 주세요.",
            "<p style='color: #888;'>분위기 추천을 입력하시면 여기에 코스가 표시됩니다.</p>"
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
    
    return destination, ghibli_work, vibe_comment, course_html

def build_ui() -> gr.Blocks:
    # 1. 지브리 테마 스타일 정의 (라이트 모드 강제 적용)
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="stone",
        neutral_hue="stone",
    ).set(
        # 전체 배경 및 블록 배경을 따뜻한 미색과 흰색으로 고정 (다크모드에서도 강제 적용)
        body_background_fill="#fcfaf2",
        body_background_fill_dark="#fcfaf2",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        
        # 텍스트 입출력 상자 배경색 고정
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        
        # 텍스트 색상 및 어두운 모드 대비 텍스트 색상 고정
        body_text_color="#4e342e",
        body_text_color_dark="#4e342e",
        block_label_text_color="#2e7d32",
        block_label_text_color_dark="#2e7d32",
        
        # 버튼 스타일링
        button_primary_background_fill="linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        
        # 테두리 선
        border_color_primary="#e0dcd3",
        border_color_primary_dark="#e0dcd3",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');

    .gradio-container {
        background-color: #fcfaf2 !important;
        font-family: 'Outfit', 'Noto Sans KR', -apple-system, sans-serif !important;
    }
    .title-section {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #a5d6a7;
    }
    /* 타이틀 및 마크다운 텍스트 색상 및 정렬 강제 */
    .title-section h1, .title-section .prose h1, .title-section h1 * {
        color: #2e7d32 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
        text-align: center !important;
    }
    .title-section p, .title-section .prose p, .title-section p * {
        color: #4e342e !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
        text-align: center !important;
    }
    .main-box {
        background: white !important;
        border-radius: 16px !important;
        border: 1px solid #e0dcd3 !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.02) !important;
        padding: 1.5rem !important;
    }
    /* 다크모드 대응: 입력창 텍스트 색상을 어두운 갈색으로 강제 고정 */
    textarea, input[type="text"] {
        color: #4e342e !important;
        background-color: #ffffff !important;
    }
    /* 아웃풋 텍스트박스 비활성화 상태 스타일 재정의 */
    textarea[readonly], input[readonly] {
        background-color: #fdfcf7 !important;
        color: #4e342e !important;
        border-color: #d0e8d2 !important;
        cursor: default !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .submit-btn {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .submit-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(46,125,50,0.3) !important;
        background: linear-gradient(135deg, #66bb6a 0%, #388e3c 100%) !important;
    }
    .course-step {
        display: flex;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .course-step:not(:last-child)::after {
        content: '';
        position: absolute;
        left: 24px;
        top: 48px;
        bottom: -24px;
        width: 2px;
        background: #c8e6c9;
    }
    .step-num {
        background: #4caf50;
        color: white;
        font-weight: 700;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        z-index: 2;
        flex-shrink: 0;
        box-shadow: 0 3px 6px rgba(76,175,80,0.3);
    }
    .step-content {
        margin-left: 1.5rem;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e8f5e9;
        flex-grow: 1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .step-content:hover {
        transform: translateX(6px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border-color: #a5d6a7;
    }
    .step-place {
        color: #2e7d32 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.25rem 0 !important;
    }
    .step-desc {
        color: #5d4037 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    .guide-accordion {
        background: #f1f8e9 !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 12px !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }
    """
    
    with gr.Blocks(css=css, theme=theme, title="지브리 감성 여행지 추천 서비스") as demo:
        with gr.Group(elem_classes=["title-section"]):
            gr.Markdown("# 🌿 Ghibli-Vibe Travel Mapper")
            gr.Markdown("<p style='text-align: center; color: #4e342e; font-size: 1.1rem;'>원하는 여행 분위기를 일상어로 쓰시면, 지브리 감성이 가득한 실제 여행지와 반나절 코스를 추천해 드립니다.</p>")
            
        with gr.Accordion("💡 Ghibli Travel Mapper 데모 프리뷰 및 안내 가이드 (클릭하여 열기)", open=False, elem_classes=["guide-accordion"]):
            gr.Image("https://raw.githubusercontent.com/aaa0034213/aiweb2026/main/screenshot.png", show_label=False, interactive=False)
            
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

        # 3. 하단 결과 영역 (가로폭을 넓게 쓰는 시원한 레이아웃)
        with gr.Row():
            with gr.Column(scale=12):
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
            outputs=[dest_out, work_out, vibe_out, course_out]
        )
        
    return demo

demo = build_ui()

if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        #ssr_mode=False,
    )
