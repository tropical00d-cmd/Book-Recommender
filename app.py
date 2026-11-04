from pathlib import Path

import streamlit as st

from services.ai_service import BookRecommenderAI
from services.book_service import (
    load_books,
    split_books,
    books_to_records
)


# -----------------------------
# 기본 설정
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
BOOKS_PATH = BASE_DIR / "data" / "books.csv"

st.set_page_config(
    page_title="책이랑",
    page_icon="📖",
    layout="centered",
)

# -----------------------------
# 데이터 및 AI 객체
# -----------------------------


@st.cache_data
def get_books():
    """
    csv는 매번 다시 읽을 필요가 없으므로 캐시함
    사용자 고민은 캐시하지 않음
    """

    return load_books(BOOKS_PATH)


@st.cache_resource
def get_ai_service():
    """
    Gemini 클라이언트를 세션마다 반복 생성하지 않는다
    """

    return BookRecommenderAI(
        api_key=st.secrets["GEMINI_API_KEY"],
        model_name=st.secrets["GEMINI_MODEL"],
    )


# -----------------------------
# AI 평가 결과 검증 함수
# -----------------------------

def validate_batch_results(
    batch_result,
    source_books
):
    """
    AI가 반환한 book_id가 실제 해당 그룹에 있는지 확인하고, 
    최종 점수를 Python에서 다시 계산함
    """

    valid_ids = set(
        source_books["book_id"].tolist()
    )

    validated = {}

    for evaluation in batch_result.results:
        if evaluation.book_id not in valid_ids:
            continue

        total_score = (
            evaluation.problem_relevance
            + evaluation.emotional_fit
            + evaluation.underlying_need_fit
            + evaluation.helpfulness
            - evaluation.risk
        )

        item = evaluation.model_dump()
        item["total_score"] = total_score

        existing = validated.get(
            evaluation.book_id
        )

        if (
            existing is None
            or total_score
            > existing["total_score"]
        ):
            validated[evaluation.book_id] = item

    sorted_results = sorted(
        validated.values(),
        key=lambda item: item["total_score"],
        reverse=True,
    )

    return sorted_results[:5]


def build_candidate_records(
    books,
    evaluations
):
    """
    AI의 평가 결과와 원래 CSV의 도서 정보를 결합하기
    """

    book_lookup = (
        books
        .set_index("book_id")
        .to_dict(orient="index")
    )       

    candidates = []

    for evaluation in evaluations:
        book_id = evaluation["book_id"]

        if book_id not in book_lookup:
            continue
        
        book = book_lookup[book_id]

        candidates.append({
            "book_id": book_id,
            "title": book["title"],
            "author": book["author"],
            "publisher": book["publisher"],
            "description": book["description"][:1000],
            "first_stage_score": (
                evaluation["total_score"]
            ),
            "matched_need": (
                evaluation["matched_need"]
            ),
            "first_stage_reason": (
                evaluation["reason"]
            ),
        })

    return candidates


def validate_final_results(
    final_result,
    candidates
):
    """
    최종 AI 응답이 실제 후보를 사용했는지 확인
    """

    candidate_ids = {
        candidate["book_id"]
        for candidate in candidates
    }

    validated = []
    used_ids = set()

    for recommendation in (
        final_result.recommendations
    ):
        book_id = recommendation.book_id

        if book_id not in candidate_ids:
            continue

        if book_id in used_ids:
            continue
        
        validated.append(
            recommendation.model_dump()
        )
        used_ids.add(book_id)

    if len(validated) != 3:
        raise ValueError(
            "AI가 유효한 도서 3권을 반환하지 못하였습니다."
        )


    return validated


# -----------------------------
# 추천 실행 함수
# -----------------------------

def run_recommendation(
    concern,
    books,
    ai_service
):
    """
    고민 분석 → 그룹별 평가 → 최종 평가를 실행함
    """

    analysis = ai_service.analyze_concern(
        concern
    )

    book_groups = split_books(books)

    all_evaluations = []

    for group_number, book_group in enumerate(
        book_groups,
        start=1
    ):
        group_records = books_to_records(
            book_group
        )

        batch_result = (
            ai_service.evaluate_book_group(
                analysis=analysis,
                book_records=group_records,
            )
        )

        validated_batch = (
            validate_batch_results(
                batch_result=batch_result,
                source_books=book_group,
            )
        )
        if len(validated_batch) < 3:
            raise ValueError(
                f"{group_number}번 도서 그룹의 "
                "유효한 평가 결과가 부족합니다."
            )

        all_evaluations.extend(
            validated_batch
        )

    candidates = build_candidate_records(
        books=books,
        evaluations=all_evaluations,
    )

    #정렬
    candidates.sort(key=lambda x: x.get("first_stage_score", 0), reverse=True)

    strong_candidates = [
        candidate
        for candidate in candidates
        if candidate["first_stage_score"] >= 60
    ]

    if len(strong_candidates) < 1:
        raise ValueError(
            "현재 등록된 도서 중 고민에 충분히 "
            "적합한 도서를 찾지 못했습니다."
        )

    final_result = (
        ai_service.select_final_books(
            analysis=analysis,
            candidates=strong_candidates[:3],
        )
    )

    recommendations = (
        validate_final_results(
            final_result=final_result,
            candidates=candidates,
        )
    )

    book_lookup = (
        books
        .set_index("book_id")
        .to_dict(orient="index")
    )

    complete_recommendations = []

    for recommendation in recommendations:
        book_id = recommendation["book_id"]

        complete_recommendations.append({
            **recommendation,
            "book": book_lookup[book_id],
        })

    return {
        "analysis": analysis.model_dump(),
        "recommendations": (
            complete_recommendations
        ),
    }


# -----------------------------
# 결과 출력 함수
# -----------------------------

def display_result(result):
    analysis = result["analysis"]

    st.subheader("AI가 이해한 고민")

    st.write(analysis["summary"])

    with st.expander(
        "고민 분석 내용 자세히 보기"
    ):
        st.markdown(
            f"**표면적으로 드러난 고민** \n"
            f"{analysis['surface_problem']}"
        )

        st.markdown(
            "**현재 드러나는 감정**"
        )

        for emotion in analysis["emotions"]:
            st.write(f"- {emotion}")

        st.markdown(
            "**필요한 독서 방향**"
        )

        for need in analysis["desired_help"]:
            st.write(f"- {need}")

        st.markdown(
            "**피하는 것이 좋은 도서 특성**"
        )

        for item in analysis["avoid"]:
            st.write(f"- {item}")

    st.divider()
    st.subheader("추천 도서")

    for rank, recommendation in enumerate(
        result["recommendations"],
        start=1
    ):
        book = recommendation["book"]

        with st.container(border=True):
            col_image, col_info = st.columns(
                [1, 3]
            )

            with col_image:
                if book.get("cover_url"):
                    st.image(
                        book["cover_url"],
                        width=120
                    )
                else:
                    st.markdown("### 📕")
                
            with col_info:
                st.markdown(
                    f"### {rank}. {book['title']}"
                )

                author_text = book["author"]

                if book.get("publisher"):
                    author_text += (
                        f" · {book['publisher']}"
                    )

                st.write(author_text)

                st.caption(
                    recommendation[
                        "recommendation_role"
                    ]
                )

            st.markdown(
                "**고민과 연결되는 부분**"
            )
            st.write(
                recommendation["matched_concern"]
            )

            st.markdown("**추천 이유**")
            st.write(recommendation["reason"])

            st.markdown(
                "**기대할 수 있는 도움**"
            )
            st.write(
                recommendation["expected_help"]
            )

            st.markdown(
                "**읽으며 생각해 볼 질문**"
            )
            st.info(
                recommendation["reading_question"]
            )

            st.markdown("**도서 이용 정보**")
            st.write(
                f"- 청구기호: "
                f"{book['call_number']}"
            )
            st.write(
                f"- 소장 위치: "
                f"{book['location']}"
            )

            if book.get("catalog_url"):
                st.link_button(
                    "도서 상세정보 확인",
                    book["catalog_url"]
                )


# -----------------------------
# 화면 구성
# -----------------------------

st.title("📚 책이랑")

st.caption(
    "고민을 분석하여 자관 도서관이 보유중인 도서 중 "
    "현재 상황에 적합한 책을 추천합니다."
)

try:
    books = get_books()
    ai_service = get_ai_service()

except Exception as error:
    st.error(
        f"서비스 초기화 중 오류가 발생했습니다: "
        f"{error}"
    )
    st.stop()


with st.sidebar:
    st.header("서비스 정보")

    st.metric(
        "추천 대상 도서",
        f"{len(books)}권"
    )

    st.write(
        "본 프로토타입은 학술제를 위해 선정한 "
        "자관 도서관 보유도서 중 일부만을 대상으로 "
        "추천 기능을 제공합니다."
    )

    st.write(
        "추천 결과는 전문적인 심리 상담이나 "
        "의학적 진단을 대신하지 않습니다."
    )


concern = st.text_area(
    "현재 어떤 고민을 가지고 있나요?",
    placeholder=(
        "최소 30자 이상 \n"
    ),
    height = 200,
    max_chars=1500,
)

st.info(
    "입력한 내용은 AI 분석을 위해 외부 AI 서비스로 "
    "전송될 수 있습니다. 실명, 연락처, 학번, 학교명 등 "
    "개인을 식별할 수 있는 정보는 입력하지 마세요."
)



recommend_button = st.button(
    "내 고민에 맞는 책 찾기",
    type="primary",
    use_container_width=True,
)


if recommend_button:
    cleaned_concern = concern.strip()

    if len(cleaned_concern) < 30:
        st.warning(
            "고민을 30자 이상 작성해 주세요. "
            "상황과 현재 느끼는 점을 함께 작성하면 "
            "더 적절한 추천을 받을 수 있습니다. "
        )
        st.stop()

    
    try:
        with st.status(
            "고민과 도서를 분석하고 있습니다...",
            expanded=True,
        ) as status:
            st.write(
                "1단계: 고민의 핵심 내용과 "
                "필요한 독서 방향을 분석합니다."
            )

            result = run_recommendation(
                concern=cleaned_concern,
                books=books,
                ai_service=ai_service,
            )

            st.write(
                "2단계: 자관 도서 50권을 "
                "두 그룹으로 나누어 평가했습니다."
            )

            st.write(
                "3단계: 상위 후보를 다시 비교하여 "
                "최종 3권을 선정했습니다."
            )

            status.update(
                label="도서 추천이 완료되었습니다.",
                state="complete",
                expanded=False,
            )

        # 현재 브라우저 세션에만 결과 저장
        st.session_state["recommendation_result"] = (
            result
        )

    except Exception as error:
        st.error(
            "추천도서 처리 중 오류가 발생했습니다."
        )

        with st.expander(
            "오류 내용 확인"
        ):
            st.code(str(error))


if "recommendation_result" in st.session_state:
    display_result(
        st.session_state[
            "recommendation_result"
        ]
    )


st.divider()    

st.caption(
    "본 추천은 등록된 도서 소개문을 기준으로 생성됩니다. "
    "도서의 전체 내용을 분석한 결과가 아니며, "
    "실시간 대출 상태는 도서관 상세 페이지에서 "
    "확인해야 합니다."
)