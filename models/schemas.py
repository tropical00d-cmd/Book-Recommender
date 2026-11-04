from pydantic import BaseModel, Field

class ConcernAnalysis(BaseModel):
    """사용자 고민 분석 결과"""

    summary: str = Field(
        description="사용자의 고민을 한 문장으로 요약"
    )

    surface_problem : str = Field(
        description="사용자가 직접 언급한 표면적 고민"
    )

    underlying_needs : list[str] = Field(
        description="고민에서 추론되는 잠재적 독서 요구"
    )

    emotions: list[str] = Field(
        description="고민글에서 드러나는 감정"
    )

    desired_help: list[str] = Field(
        description="사용자가 독서를 통해 얻을 필요가 있는 도움"
    )

    avoid: list[str] = Field(
        description="현재 사용자에게 적합하지 않을 수 있는 도서 특성"
    )


class BookEvaluation(BaseModel):
    """한 권의 독서에 대한 AI 평가"""

    book_id: str = Field(
        description="입력으로 제공된 도서 식별번호"
    )

    problem_relevance: int = Field(
        ge=0,
        le=30,
        description="핵심 고민 관련성, 0~30점"
    )

    emotional_fit: int = Field(
        ge=0,
        le=20,
        description="사용자의 감정 상태와의 적합성, 0~20점"
    )

    underlying_need_fit: int = Field(
        ge=0,
        le=25,
        description="잠재적 독서 요구 적합성, 0~25점"
    )

    helpfulness: int = Field(
        ge=0,
        le=25,
        description="관점 전환 또는 실천 도움 가능성, 0~25점"
    )

    risk: int = Field(
        ge=0,
        le=20,
        description="압박이나 부정적 감정을 강화할 위험, 0~20점"
    )

    matched_need: str = Field(
        description="이 도서가 대응하는 주요 독서 요구"
    )

    reason: str = Field(
        description="고민과 도서 소개문을 연결한 평가 이유"
    )


class BatchEvaluation(BaseModel):
    """한 그룹에서 선정된 상위 도서"""

    results: list[BookEvaluation] =Field(
        min_length=5,
        max_length=5, 
        description="해당 그룹에서 적합성이 높은 상위 5권"
    )


class FinalRecommendation(BaseModel):
    """최종 추천 도서"""

    book_id: str = Field(
        description="입력 후보에 포함된 도서 식별번호"
    )

    recommendation_role: str = Field(
        description="고민 이해, 관점 전환, 실천 지원 중 이 책의 역할"
    )

    matched_concern: str = Field(
        description="이 책이 대응하는 사용자의 고민 요소"
    )

    reason: str = Field(
        description="고민과 도서 소개문을 연결한 구체적인 추천 이유"
    )

    expected_help: str = Field(
        description="책을 통해 얻을 수 있는 관점이나 도움"
    )

    reading_question: str = Field(
        description="책을 읽으면서 생각해 볼 질문"
    )


class FinalRecommendations(BaseModel):
    """최종 추천 결과"""

    recommendations: list[FinalRecommendation] = Field(
        min_length=3,
        max_length=3,
        description="최종 추천 도서 3권"
    )