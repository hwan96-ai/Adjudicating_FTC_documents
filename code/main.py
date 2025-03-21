from document_analyzer import DocumentAnalyzer
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 메인 함수: 사용자로부터 파일 경로를 받아 계약서 분석을 실행합니다.
# 입력 경로를 처리하고, 파일이 존재하면 DocumentAnalyzer로 분석을 수행한 뒤 결과를 저장합니다.
# 입력: 없음 (사용자 입력으로 경로 받음)
# 출력: 없음 (분석 결과는 파일로 저장되고 로그로 출력)
def main():
    analyzer = DocumentAnalyzer()

    # 프로젝트 루트와 기본 입력 디렉토리 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input_dir = os.path.join(project_root, "data", "file")
    
    # 사용자 입력 받기
    prompt = f"분석할 PDF 또는 JSON 파일 경로를 입력하세요\n(예: '화학업종.pdf' 또는 'data/file/화학업종.pdf', 기본 디렉토리: {default_input_dir}): "
    input_path = input(prompt).strip()
    
    # 경로 처리
    if not os.path.isabs(input_path):
        # 파일 이름만 입력된 경우, 기본 디렉토리에서 찾기
        if os.path.splitext(input_path)[1] in ['.pdf', '.json'] and '/' not in input_path and '\\' not in input_path:
            input_path = os.path.join(default_input_dir, input_path)
        else:
            input_path = os.path.join(project_root, input_path)
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_path):
        logger.error(f"파일을 찾을 수 없습니다: {input_path}")
        logger.info(f"확인: {default_input_dir} 디렉토리를 확인하거나, 절대 경로를 입력하세요.")
        return
    
    logger.info(f"사용자 문서 분석 시작: {input_path}")
    result = analyzer.analyze_contract(input_path)
    if "error" not in result:
        analyzer.save_results(result, input_path)
        if result.get("problem_objects"):
            logger.info(f"문제가 있는 객체 인덱스: {result['problem_objects']}")
    else:
        logger.error(f"분석 오류: {result['error']}")

if __name__ == "__main__":
    main()