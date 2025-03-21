import pdfplumber
import json
import boto3
import re
import os
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
class DocumentClassifier:
    def __init__(self, mapping_file: str = "data/doc_mapping.json"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mapping_file = os.path.join(self.project_root, mapping_file)
        self.document_mapping = self._load_document_mapping()
        self.bedrock_runtime = self._get_bedrock_client()
        self.model_ids = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-7-sonnet-20250219-v1:0"
        ]
        self.prompt_template = ChatPromptTemplate.from_template("""
        다음은 하도급 문서의 일부분입니다. 이 문서가 어떤 유형인지 판단해주세요.
        [문서 제목 정보]
        {document_title}
        [문서 시작 부분]
        {start_text}
        [문서 중간 부분]
        {middle_text}
        [문서 끝 부분]
        {end_text}
        문서 유형 매핑:
        {document_mapping}
        위 정보를 바탕으로, 이 문서가 어떤 번호의 문서 유형에 해당하는지 판단하고 해당 번호만 응답해주세요.
        어떤 설명도 포함하지 말고 숫자만 정확하게 응답해주세요.
        """)

    def _load_document_mapping(self) -> Dict[str, str]:
        try:
            if not os.path.exists(self.mapping_file):
                raise FileNotFoundError(f"매핑 파일 '{self.mapping_file}'이 없습니다.")
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 파일 로드 오류: {e}")
            return {}

    def _get_bedrock_client(self) -> Any:
        try:
            return boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
        except Exception as e:
            print(f"Bedrock 클라이언트 생성 오류: {e}")
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"PDF 추출 오류: {e}")
        return text

    def _extract_text_from_json(self, json_path: str) -> str:
        text = ""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text += item.get("text", "") + "\n"
                elif isinstance(data, dict):
                    text += data.get("text", "") + "\n"
        except Exception as e:
            print(f"JSON 추출 오류: {e}")
        return text

    def _extract_title_from_file(self, file_path: str) -> str:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_path.lower().endswith('.pdf'):
            potential_title = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    if pdf.pages:
                        first_page_text = pdf.pages[0].extract_text() or ""
                        first_300_chars = first_page_text[:300]
                        lines = first_300_chars.split('\n')
                        title_keywords = ["표준하도급계약서", "계약서", "협약서", "특약서"]
                        if lines and len(lines[0]) < 100:
                            potential_title = lines[0].strip()
                        else:
                            for i in range(min(5, len(lines))):
                                for keyword in title_keywords:
                                    if keyword in lines[i]:
                                        potential_title = lines[i].strip()
                                        break
                                if potential_title:
                                    break
            except Exception as e:
                print(f"제목 추출 오류 (PDF): {e}")
            full_title = f"파일명: {file_name}"
            if potential_title:
                full_title += f" | 문서 제목: {potential_title}"
            return full_title
        elif file_path.lower().endswith('.json'):
            return f"파일명: {file_name}"
        return f"파일명: {file_name}"

    def _get_document_samples(self, text: str) -> Dict[str, str]:
        samples = {"시작 부분": "", "중간 부분": "", "끝 부분": ""}
        text_length = len(text)
        if text_length > 0:
            samples["시작 부분"] = text[:min(1000, text_length)]
        if text_length > 1500:
            middle_start = max(0, text_length // 2 - 250)
            middle_end = min(text_length, middle_start + 500)
            samples["중간 부분"] = text[middle_start:middle_end]
        if text_length > 500:
            samples["끝 부분"] = text[max(0, text_length - 500):]
        return samples

    def classify(self, file_path: str) -> str:
        if not self.document_mapping:
            print("문서 매핑 정보를 로드할 수 없습니다.")
            return "0"
        if not self.bedrock_runtime:
            print("Bedrock 클라이언트를 사용할 수 없습니다.")
            return "0"
        
        # 파일 형식에 따라 텍스트 추출
        if file_path.lower().endswith('.pdf'):
            full_text = self._extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.json'):
            full_text = self._extract_text_from_json(file_path)
        else:
            print(f"지원하지 않는 파일 형식: {file_path}")
            return "0"
        
        if not full_text:
            print(f"파일에서 텍스트를 추출할 수 없습니다: {file_path}")
            return "0"
        
        document_title = self._extract_title_from_file(file_path)
        text_samples = self._get_document_samples(full_text)
        input_data = {
            "document_title": document_title,
            "start_text": text_samples["시작 부분"],
            "middle_text": text_samples["중간 부분"],
            "end_text": text_samples["끝 부분"],
            "document_mapping": json.dumps(self.document_mapping, ensure_ascii=False)
        }
        try:
            for model_index, model_id in enumerate(self.model_ids):
                print(f"[시도 {model_index + 1}] 모델 {model_id} 사용 중...")
                try:
                    chat = ChatBedrock(model_id=model_id, client=self.bedrock_runtime)
                    chain = self.prompt_template | chat
                    result = chain.invoke(input_data)
                    result_text = result.content.strip()
                    number_match = re.search(r'^\d+$', result_text)
                    if number_match:
                        number = number_match.group(0)
                        print(f"분류 완료: {file_path} -> 유형 #{number} (모델: {model_id})")
                        return number
                    any_number_match = re.search(r'\d+', result_text)
                    if any_number_match:
                        number = any_number_match.group(0)
                        print(f"분류 완료(숫자 추출): {file_path} -> 유형 #{number} (모델: {model_id})")
                        return number
                    print(f"모델 {model_id}에서 숫자를 찾을 수 없음: {result_text}")
                    if model_index == len(self.model_ids) - 1:
                        print(f"분류 실패: {file_path} - 모든 모델 시도 후에도 숫자를 찾을 수 없음")
                        return "0"
                    print(f"다음 모델 {self.model_ids[model_index + 1]}로 시도합니다...")
                except Exception as e:
                    print(f"모델 {model_id} 호출 오류: {e}")
                    if model_index == len(self.model_ids) - 1:
                        print("모든 모델 시도 실패")
                        return "0"
            return "0"
        except Exception as e:
            print(f"문서 분류 오류: {e}")
            return "0"

if __name__ == "__main__":
    classifier = DocumentClassifier()
    pdf_dir = os.path.join(classifier.project_root, "data", "file")
    import glob
    for file_path in glob.glob(f"{pdf_dir}/*.[pj][ds][fo][n]"):  # .pdf와 .json 모두 처리
        document_number = classifier.classify(file_path)
        print(f"{os.path.basename(file_path)}: 유형 #{document_number}")