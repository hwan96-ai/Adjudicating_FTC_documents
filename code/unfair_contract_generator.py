import os
import json
import boto3
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import logging
from document_processor import DocumentProcessor
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
# 로깅 설정: 기본적으로 CRITICAL만 출력
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
# langchain_aws 로그 억제
logging.getLogger("langchain_aws").setLevel(logging.CRITICAL)



# AWS Bedrock 클라이언트를 설정합니다.
# .env 파일에서 AWS 자격 증명을 읽어와 boto3 클라이언트를 초기화하며,
# 오류 발생 시 예외를 발생시킵니다.
# 입력: 없음
# 출력: boto3.client - 초기화된 AWS Bedrock 클라이언트 객체
def setup_aws_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
# LLM(Large Language Model)을 설정합니다.
# AWS Bedrock 클라이언트를 사용해 지정된 모델 ID로 ChatBedrock 객체를 생성합니다.
# 입력: model_id (str, 기본값 "us.anthropic.claude-3-7-sonnet-20250219-v1:0") - 사용할 모델 ID
# 출력: ChatBedrock - 초기화된 LLM 모델 객체
def setup_llm_model(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
    client = setup_aws_client()
    model_kwargs = {"temperature": 0.3, "top_k": 0, "top_p": 1, "stop_sequences": ["\n\nHuman"]}
    return ChatBedrock(model_id=model_id, client=client, model_kwargs=model_kwargs, streaming=False)

# 주어진 계약서 항목을 불공정하게 수정합니다.
# LLM을 사용해 계약서 텍스트를 불공정하게 변형하며, 수정 힌트를 함께 반환합니다.
# 입력: entry (Dict) - 수정할 계약서 항목 (title, chapter, section, article, text 포함)
#       llm (ChatBedrock) - 텍스트 생성에 사용할 LLM 모델
#       idx (int) - 항목 인덱스, 수정 유형 결정에 사용
# 출력: Tuple[str, str] - (불공정한 텍스트, 수정 힌트)
def generate_unfair_text(entry: Dict, llm, idx: int) -> tuple[str, str]:
    change_type = ["changed_clauses", "missing_clauses", "added_clauses", "original"][idx % 4]

    if change_type == "original":
        return entry["text"], "변경 없음: 원본 계약서 내용 그대로 사용"

    prompts = {
        "changed_clauses": """
            당신은 하도급 계약서 작성 전문가입니다. 수급사업자에게 불리하도록 계약서 조항 하나를 변경하세요.
            변경된 내용을 "힌트: [바뀐 부분]"을 추가로 반환하세요。
        """,
        "missing_clauses": """
            당신은 하도급 계약서 작성 전문가입니다. 주어진 계약서 조항에서 수급사업자에게 불리하도록 하나의 조항을 삭제하세요.
            변경된 내용을 "힌트: [바뀐 부분]"을 추가로 반환하세요。
        """,
        "added_clauses": """
            당신은 하도급 계약서 작성 전문가입니다. 주어진 계약서 조항에서 수급사업자에게 새로운 법적 의무나 제한이 되는 조항 하나만 추가하세요.
            변경된 내용을 "힌트: [바뀐 부분]"을 추가로 반환하세요。
        """
    }

    prompt = ChatPromptTemplate.from_template(f"""
    {prompts[change_type]}

    원래 계약서 조항:
    - 제목: {{title}}
    - 장: {{chapter}}
    - 절: {{section}}
    - 조: {{article}}
    - 내용: {{text}}

    수정된 내용만 텍스트로 반환하며, JSON 형식이나 코드 블록(```)은 포함시키지 마세요.
    변경 내용 설명은 "힌트: " 뒤에 이어서 작성하세요.
    """)

    chain = prompt | llm
    try:
        response = chain.invoke({
            "title": entry["title"],
            "chapter": entry["chapter"],
            "section": entry["section"],
            "article": entry["article"],
            "text": entry["text"]
        })
        content = response.content.strip()
        
        if "힌트:" in content:
            text, hint = content.split("힌트:", 1)
            text = text.strip()
            hint = hint.strip()
        else:
            text = content
            hint = "변경 내용 설명 없음"
        
        return text, f"{change_type}: {hint}"
    except Exception as e:
        return f"오류 발생: {str(e)}", f"{change_type}: 오류로 인해 변경 실패"

# 폴더 내 모든 PDF 파일을 처리하여 불공정 계약서를 생성합니다.
# PDF를 JSON으로 변환하고, 각 항목을 불공정하게 수정한 뒤 결과를 저장합니다.
# 입력: input_folder (str) - PDF 파일이 있는 입력 폴더 경로
#       temp_folder (str) - 중간 JSON 파일을 저장할 임시 폴더 경로
#       output_folder (str) - 불공정 계약서 JSON을 저장할 출력 폴더 경로
# 출력: 없음 (파일로 저장됨)
def process_all_pdfs_in_folder(input_folder: str, temp_folder: str, output_folder: str):
    processor = DocumentProcessor()
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return

    # 1단계: PDF → JSON 변환
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        temp_json_path = os.path.join(temp_folder, f"{base_name}.json")
        unfair_json_path = os.path.join(output_folder, f"unfair_{base_name}.json")

        original_json = processor.process_pdf_to_json(pdf_path, temp_json_path)

        unfair_structure = [
            {
                "title": entry["title"],
                "chapter": entry["chapter"],
                "section": entry["section"],
                "article": entry["article"],
                "text": entry["text"],
                "page": entry["page"],
                "hint": ""
            } for entry in original_json
        ]
        with open(unfair_json_path, 'w', encoding='utf-8') as f:
            json.dump(unfair_structure, f, ensure_ascii=False, indent=4)

    # 2단계: LLM으로 불공정 내용 채우기
    llm = setup_llm_model()
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        unfair_json_path = os.path.join(output_folder, f"unfair_{base_name}.json")

        with open(unfair_json_path, 'r', encoding='utf-8') as f:
            unfair_json = json.load(f)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [(executor.submit(generate_unfair_text, entry, llm, idx), idx) for idx, entry in enumerate(unfair_json)]
            for future, idx in tqdm(futures, desc=f"불공정 내용 생성 중 ({base_name})", total=len(unfair_json)):
                text, hint = future.result()
                unfair_json[idx]["text"] = text
                unfair_json[idx]["hint"] = hint

        with open(unfair_json_path, 'w', encoding='utf-8') as f:
            json.dump(unfair_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, "data", "file")
    temp_folder = os.path.join(project_root, "temp", "temp_json")
    output_folder = os.path.join(project_root, "data", "unfair_contracts")
    process_all_pdfs_in_folder(input_folder, temp_folder, output_folder)