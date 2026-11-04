from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "book_id",
    "title",
    "author",
    "description",
    "call_number",
    "location",
    "catalog_url",
]

OPTIONAL_COLUMNS = [
    "publisher",
    "pages",
    "publication_year",
    "cover_url",
]


def load_books(csv_path: Path) -> pd.DataFrame:
    """
    자관 도서 CSV를 읽고 필수 필드와 데이터 중복을 검사
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"도서 데이터 파일을 찾을 수 없습니다: {csv_path}"
        )

    books = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8"
    ).fillna("")

    #열 이름의 앞뒤 공백 제거
    books.columns = [
        column.strip()
        for column in books.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in books.columns
    ]

    if missing_columns: 
        raise ValueError(
            "CSV에 다음 필수 열이 없습니다: "
            +", ".join(missing_columns)
        )

    #부가적인 열이 없으면 빈 문자열로 생성
    for column in OPTIONAL_COLUMNS:
        if column not in books.columns:
            books[column] = ""

    #모든 문자열 앞뒤 공백 제거
    for column in books.columns:
        books[column] = (
            books[column]
            .astype(str)
            .str.strip()
        )

    #book_id가 비어 있는 행 검사
    empty_id_rows = books[
        books["book_id"] == ""
    ]

    if not empty_id_rows.empty:
        raise ValueError(
            "book_id가 비어 있는 데이터가 있습니다."
        )

    #book_id 중복 검사
    duplicated_ids = books[
        books["book_id"].duplicated(keep=False)
    ]["book_id"].tolist()

    if duplicated_ids: 
        raise ValueError(
            "중복된 book_id가 있습니다: "
            +", ".join(sorted(set(duplicated_ids)))
        )

    #필수 텍스트 검사
    for column in ["title", "author", "description"]:
        empty_rows = books[
            books[column] == ""
        ]

        if not empty_rows.empty:
            problem_ids = empty_rows["book_id"].tolist()

            raise ValueError(
                f"{column} 값이 비어 있는 도서가 있습니다: "
                +", ".join(problem_ids)
            )
    
    if len(books) < 3:
        raise ValueError(
            "추천 도서 선정을 위해 최소 3개 이상의 도서 데이터가 필요합니다."
        )

    return books.reset_index(drop=True)



def split_books(
    books: pd.DataFrame
) -> list[pd.DataFrame]:
    """
    도서 목록을 무작위로 섞은 뒤 두 그룹으로 분리함

    random_state를 고정하여 실행할 때마다 
    동일한 그룹이 만들어지도록 함
    """

    shuffled = books.sample(
        frac = 1,
        random_state=42
    ).reset_index(drop=True)

    middle = (len(shuffled) + 1)//2

    first_group = shuffled.iloc[:middle].copy()
    second_group = shuffled.iloc[middle:].copy()

    return [first_group, second_group]


def books_to_records(
    books: pd.DataFrame
) -> list[dict]:
    """
    AI에게 전달할 도서 데이터를 구성
    개인정보나 불필요한 서지정보는 전달하지 않음
    """

    records = []

    for _, book in books.iterrows():
        records.append({
            "book_id": book["book_id"],
            "title": book["title"],
            "author": book["author"],
            "publisher": book["publisher"],
            "description": book["description"][:1000],
        })

    return records