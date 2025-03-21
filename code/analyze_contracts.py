import json
import pandas as pd
import os
import sys

# 사용자 문서와 참조 문서의 필드를 비교하여 유사도 순위를 찾는 함수
# 사용자 문서의 제목, 장, 절, 조와 참조 문서의 동일 필드를 비교해 일치하는 경우 유사도 순위를 반환합니다.
# 입력: user_title (str) - 사용자 문서의 제목
#       user_chapter (str) - 사용자 문서의 장
#       user_section (str) - 사용자 문서의 절
#       user_article (str) - 사용자 문서의 조
#       reference_metadata (list) - 참조 문서의 메타데이터 리스트
# 출력: int | None - 일치하는 경우 가장 낮은 유사도 순위, 없으면 None
def find_matching_similarity_rank(user_title, user_chapter, user_section, user_article, reference_metadata):
    matches = []
    for ref in reference_metadata:
        ref_title = str(ref.get('제목', 'unknown')).strip()
        ref_chapter = str(ref.get('장', 'unknown')).strip()
        ref_section = str(ref.get('절', 'unknown')).strip()
        ref_article = str(ref.get('조', 'unknown')).strip()
        similarity_rank = ref.get('유사도_순위', -1)

        user_title = str(user_title).strip()
        user_chapter = str(user_chapter).strip()
        user_section = str(user_section).strip()
        user_article = str(user_article).strip()

        if (user_title == ref_title and 
            user_chapter == ref_chapter and 
            user_section == ref_section and 
            user_article == ref_article):
            matches.append(similarity_rank)
    
    return min(matches) if matches else None

# JSON 분석 결과를 읽고 요약 정보를 추출하여 Excel 파일로 저장합니다.
# 페이지별 문제 여부와 참조 문서 유사도를 분석하며, 통계를 출력합니다.
# 입력: json_path (str) - 분석할 JSON 파일 경로
# 출력: 없음 (Excel 파일로 저장되고 콘솔에 요약 출력)
def analyze_json_file(json_path):
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터 추출
    extracted_data = []
    for page in data["analysis_results"]:
        page_info = page.get("page_info", {})
        page_number = str(page_info.get("쪽", "unknown"))
        user_title = str(page_info.get("제목", ""))
        user_chapter = str(page_info.get("장", ""))
        user_section = str(page_info.get("절", ""))
        user_article = str(page_info.get("조", ""))

        # has_issues가 누락된 경우 감지
        has_issues = page.get("has_issues")
        if has_issues is None:
            print(f"Warning: 'has_issues' missing in page {page_number}")
            has_issues = False

        reference_metadata = page.get("reference_metadata", [])

        similarity_rank = find_matching_similarity_rank(
            user_title, user_chapter, user_section, user_article, reference_metadata
        )

        extracted_data.append({
            "page_number": page_number,
            "title": user_title,
            "chapter": user_chapter,
            "section": user_section,
            "article": user_article,
            "has_issues": has_issues,
            "matching_similarity_rank": similarity_rank,
            "reference_metadata": reference_metadata
        })

    # 데이터프레임으로 변환
    df = pd.DataFrame(extracted_data)

    # 분석 요약
    total_pages = len(df)
    problem_pages = len(df[df['has_issues'] == True])
    matched_pages = len(df[df['matching_similarity_rank'].notna()])
    matched_problem_pages = len(df[(df['has_issues'] == True) & (df['matching_similarity_rank'].notna())])

    print(f"전체 페이지 수: {total_pages}")
    print(f"문제 있는 페이지 수: {problem_pages}")
    print(f"참조 문서와 필드가 일치하는 페이지 수: {matched_pages}")
    print(f"문제 있고 참조 문서와 필드가 일치하는 페이지 수: {matched_problem_pages}")

    # 결과를 Excel로 저장
    output_excel = os.path.splitext(json_path)[0] + "_summary.xlsx"
    display_cols = ['page_number', 'has_issues', 'matching_similarity_rank']
    df[display_cols].to_excel(output_excel, index=False)
    print(f"분석 결과가 '{output_excel}'에 저장되었습니다.")

# 분석할 JSON 파일을 사용자가 선택하도록 인터페이스를 제공합니다.
# 지정된 폴더에서 JSON 파일 목록을 보여주고, 사용자가 번호로 선택한 파일 경로를 반환합니다.
# 입력: analysis_folder (str, 기본값 "../analysis_results") - JSON 파일이 있는 폴더 경로
# 출력: str - 선택된 JSON 파일의 전체 경로
def select_json_file(analysis_folder="../analysis_results"):
    # analysis_results 폴더에서 JSON 파일 목록 가져오기
    if not os.path.exists(analysis_folder):
        print(f"오류: '{analysis_folder}' 폴더가 존재하지 않습니다.")
        sys.exit(1)

    json_files = [f for f in os.listdir(analysis_folder) if f.endswith('.json')]
    if not json_files:
        print(f"오류: '{analysis_folder}' 폴더에 JSON 파일이 없습니다.")
        sys.exit(1)

    # 파일 목록 출력
    print("분석할 JSON 파일을 선택하세요:")
    for i, json_file in enumerate(json_files, 1):
        print(f"{i}. {json_file}")

    # 사용자 입력 받기
    while True:
        try:
            choice = int(input("파일 번호를 입력하세요 (1-{}): ".format(len(json_files))))
            if 1 <= choice <= len(json_files):
                return os.path.join(analysis_folder, json_files[choice - 1])
            else:
                print(f"잘못된 입력입니다. 1에서 {len(json_files)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

if __name__ == "__main__":
    # JSON 파일 선택
    selected_json = select_json_file()
    print(f"선택된 파일: {selected_json}")
    
    # 분석 실행
    analyze_json_file(selected_json)