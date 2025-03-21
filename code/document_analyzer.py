import os
import boto3
import json
from typing import List, Dict, Any
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings
import re
from document_processor import DocumentProcessor
from document_classifier import DocumentClassifier
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
# Bedrock 관련 로거 억제
logging.getLogger('langchain_aws').setLevel(logging.WARNING)

class DocumentAnalyzer:
    # DocumentAnalyzer 클래스의 인스턴스를 초기화합니다.
    # 프로젝트 경로, 문서 처리기, 분류기, 벡터 DB, LLM을 설정하여 계약서 분석 환경을 준비합니다.
    # 입력: 없음
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processor = DocumentProcessor()
        self.classifier = DocumentClassifier()
        self.doc_mapping_path = os.path.join(self.project_root, "data", "doc_mapping.json")
        self.vectorstore = self._setup_retrieval_system()
        self.llm = self._setup_llm_model()
    # AWS Bedrock 클라이언트를 설정합니다.
    # .env 파일에서 AWS 자격 증명을 읽어 Bedrock 런타임 클라이언트를 생성합니다.
    # 입력: 없음
    # 출력: boto3.client - 초기화된 AWS Bedrock 클라이언트 객체
    def _setup_aws_client(self):
        return boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    # HuggingFace 임베딩 모델을 설정합니다.
    # BGE-m3-ko 모델을 사용하며, CUDA 장치에서 정규화된 임베딩을 생성합니다.
    # 입력: 없음
    # 출력: HuggingFaceEmbeddings - 초기화된 임베딩 모델 객체
    def _setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="dragonkue/BGE-m3-ko",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
    # LLM(Large Language Model)을 설정합니다.
    # AWS Bedrock에서 Claude-3-7-Sonnet 모델을 사용하며, 온도와 토큰 설정을 적용합니다.
    # 입력: model_id (str, 기본값 "us.anthropic.claude-3-7-sonnet-20250219-v1:0") - 사용할 모델 ID
    # 출력: ChatBedrock - 초기화된 LLM 모델 객체
    def _setup_llm_model(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        client = self._setup_aws_client()
        model_kwargs = {"temperature": 0, "top_k": 0, "top_p": 1, "stop_sequences": ["\n\nHuman"]}
        return ChatBedrock(model_id=model_id, client=client, model_kwargs=model_kwargs, streaming=False)
    # 벡터 검색 시스템(ChromaDB)을 설정합니다.
    # 기존 벡터 DB 디렉토리를 사용해 임베딩 기반 검색 시스템을 초기화합니다.
    # 입력: 없음
    # 출력: Chroma - 초기화된 Chroma 벡터 검색 객체 
    def _setup_retrieval_system(self):
        embeddings = self._setup_embeddings()
        return Chroma(
            persist_directory=os.path.join(self.project_root, "data", "vectordb"),
            embedding_function=embeddings,
            collection_name="contracts_collection"
        )

    # 사용자 문서를 JSON 형식으로 변환합니다.
    # PDF는 JSON으로 변환 후 로드하고, JSON 파일은 바로 로드하며, 오류 시 빈 리스트를 반환합니다.
    # 입력: input_path (str) - 처리할 PDF 또는 JSON 파일 경로
    # 출력: List[Dict] - 변환된 JSON 데이터 리스트 (오류 시 빈 리스트)  
    def _process_user_document_to_json(self, input_path: str) -> List[Dict]:
        if input_path.lower().endswith('.pdf'):
            output_folder = os.path.join(self.project_root, "temp", "temp_json")
            os.makedirs(output_folder, exist_ok=True)
            json_file_name = os.path.splitext(os.path.basename(input_path))[0] + ".json"
            json_path = os.path.join(output_folder, json_file_name)
            self.processor.process_pdf_to_json(input_path, json_path)
        elif input_path.lower().endswith('.json'):
            json_path = input_path
        else:
            print(f"오류: 지원하지 않는 파일 형식입니다: {input_path}")
            return []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 파일 로드 중 오류: {e}")
            return []

    # 문서 유형에 해당하는 표준 계약서 제목을 반환합니다.
    # doc_mapping.json에서 유형 번호에 맞는 제목을 찾아 반환하며, 오류 시 빈 문자열을 반환합니다.
    # 입력: doc_type (str) - 문서 유형 번호
    # 출력: str - 해당 유형의 제목 (찾지 못하면 빈 문자열)
    def _get_document_title_from_type(self, doc_type: str) -> str:
        try:
            with open(self.doc_mapping_path, 'r', encoding='utf-8') as f:
                doc_mapping = json.load(f)
            for title, type_id in doc_mapping.items():
                if type_id == doc_type:
                    return title
            print(f"문서 유형 {doc_type}에 해당하는 제목을 찾을 수 없습니다.")
            return ""
        except Exception as e:
            print(f"문서 제목 매핑 오류: {e}")
            return ""

    # 주어진 유형에 맞는 표준 계약서를 벡터 DB에서 검색합니다.
    # 문서 제목을 기반으로 유사한 표준 계약서를 찾아 리스트로 반환하며, 오류 시 빈 리스트를 반환합니다.
    # 입력: doc_type (str) - 검색할 문서 유형 번호
    # 출력: List[Dict] - 표준 계약서 데이터 리스트 (id, content, metadata 포함)
    def _find_standard_document(self, doc_type: str) -> List[Dict]:
        doc_title = self._get_document_title_from_type(doc_type)
        if not doc_title:
            return []
        print(f"검색할 표준 계약서: {doc_title}")
        try:
            filter_dict = {"source_file": doc_title.lower()}
            results = self.vectorstore.similarity_search(query=doc_title, filter=filter_dict, k=3)
            return [{"id": doc.metadata.get('id', ''), "content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            print(f"표준 계약서 검색 오류: {e}")
            return []

    # JSON 객체를 표준 계약서와 비교하여 문제점을 분석합니다.
    # LLM을 사용해 변경/누락/추가된 조항을 분류하고, 유사 문서를 참조하여 결과를 반환합니다.
    # 입력: json_obj (Dict) - 분석할 JSON 객체 (title, chapter, section, article, text, page 포함)
    #       standard_docs (List[Dict]) - 비교할 표준 계약서 리스트
    # 출력: Dict - 분석 결과 (title, chapter, section, article, page, analysis, has_issues, reference_metadata 포함)

    def _analyze_json_object(self, json_obj: Dict, standard_docs: List[Dict]) -> Dict:
        obj_content = json_obj["text"]
        obj_title = json_obj["title"]
        obj_page = json_obj.get("page", "")
        obj_chapter = json_obj.get("chapter", "")
        obj_section = json_obj.get("section", "")
        obj_article = json_obj.get("article", "")

        standard_sources = set(doc['metadata'].get('source_file', '').lower() for doc in standard_docs)
        filter_dict = {"source_file": {"$in": list(standard_sources)}} if standard_sources else None

        query = f"{obj_title} {obj_content}"
        similar_docs = self.vectorstore.similarity_search(query=query, filter=filter_dict, k=1)
        
        standard_content = "\n\n".join([
            f"--- 표준 계약서 부분 {i+1} ---\n제목: {doc.metadata.get('title', 'unknown')}\n장: {doc.metadata.get('chapter', 'unknown')}\n절: {doc.metadata.get('section', 'unknown')}\n조: {doc.metadata.get('article', 'unknown')}\n내용: {doc.page_content}"
            for i, doc in enumerate(similar_docs)
        ])
        reference_metadata = [
            {'제목': doc.metadata.get('title', 'unknown'), '장': doc.metadata.get('chapter', 'unknown'), '절': doc.metadata.get('section', 'unknown'), '조': doc.metadata.get('article', 'unknown'), '쪽': doc.metadata.get('page', 'unknown'), '유사도_순위': i + 1}
            for i, doc in enumerate(similar_docs)
        ]

        prompt = ChatPromptTemplate.from_template("""
        당신은 하도급 계약서 검토 전문가입니다. 사용자의 계약서와 표준 계약서를 비교하여 다음 세 가지 카테고리로만 문제점을 분류하세요:
        # 원본 표준 계약서 정보: {standard_content}
        # 사용자 계약서 정보: 제목: {obj_title}, 장: {obj_chapter}, 절: {obj_section}, 조: {obj_article}, 페이지: {obj_page}, 내용: {user_content}
        ## 문서 동일성 평가 (최우선 지침): 먼저, 두 문서가 실질적으로 동일한지 철저히 평가하세요. 실질적으로 동일하다면 모든 카테고리에 빈 배열을 반환하고 분석을 종료하세요.
        ## 분석 지시사항: 실질적인 차이가 있는 경우만 다음 카테고리로 분류하세요:
        1. 변경된 조항 (changed_clauses): 표준계약서와 비교했을 때 수급사업자에게 불리하게 바뀐 조항
        2. 누락된 조항 (missing_clauses): 표준계약서에 있는 중요 보호조항이 사용자 문서에서 삭제된 경우
        3. 추가된 조항 (added_clauses): 표준계약서에 없는 새로운 법적 의무나 제한이 추가된 경우
        ## 응답 형식: 정확히 다음 JSON 형식으로만 응답하세요:
        {{"changed_clauses": [], "missing_clauses": [], "added_clauses": []}}
        """)
        chain = prompt | self.llm | JsonOutputParser()

        try:
            result = chain.invoke({
                "standard_content": standard_content,
                "user_content": obj_content,
                "obj_title": obj_title,
                "obj_chapter": obj_chapter,
                "obj_section": obj_section,
                "obj_article": obj_article,
                "obj_page": obj_page if obj_page else "정보 없음"
            })
        except Exception as e:
            result = {"changed_clauses": [], "missing_clauses": [], "added_clauses": [], "error": str(e)}

        has_issues = any(len(result.get(key, [])) > 0 for key in ["changed_clauses", "missing_clauses", "added_clauses"])
        return {
            "title": obj_title,
            "chapter": obj_chapter,
            "section": obj_section,
            "article": obj_article,
            "page": obj_page,
            "analysis": result,
            "has_issues": has_issues,
            "reference_metadata": reference_metadata
        }
    

    # 여러 JSON 객체를 병렬로 분석합니다.
    # ThreadPoolExecutor를 사용해 다중 스레드로 분석 작업을 수행하며, 진행 상황을 표시합니다.
    # 입력: json_objects (List[Dict]) - 분석할 JSON 객체 리스트
    #       standard_docs (List[Dict]) - 비교할 표준 계약서 리스트
    #       max_workers (int, 기본값 10) - 병렬 처리 스레드 수
    # 출력: List[Dict] - 각 객체의 분석 결과 리스트
    def _analyze_json_objects_in_parallel(self, json_objects: List[Dict], standard_docs: List[Dict], max_workers: int = 10) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._analyze_json_object, obj, standard_docs) for obj in json_objects]
            for future in tqdm(futures, desc="JSON 객체 분석 중", total=len(json_objects)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"분석 중 오류: {str(e)}")
                    results.append({"error": str(e), "has_issues": False})
        return results

    # 주어진 계약서를 분석합니다.
    # 입력 파일을 JSON으로 변환하고, 유형을 분류한 뒤 표준 계약서와 비교하여 분석 결과를 반환합니다.
    # 입력: input_path (str) - 분석할 PDF 또는 JSON 파일 경로
    # 출력: Dict - 분석 결과 (문서 유형, 제목, 객체 수, 문제 객체 인덱스, 분석 결과, 소요 시간 포함)
    def analyze_contract(self, input_path: str) -> Dict:
        start_time = time.time()
        print(f"계약서 분석 시작: {input_path}")
        json_objects = self._process_user_document_to_json(input_path)
        if not json_objects:
            return {"error": "입력 파일에서 JSON 객체를 추출할 수 없습니다."}
        
        doc_type = self.classifier.classify(input_path)
        print(f"문서 유형 분류 결과: {doc_type}")
        doc_title = self._get_document_title_from_type(doc_type)
        print(f"표준계약서 제목: {doc_title}")
        
        standard_docs = self._find_standard_document(doc_type)
        if not standard_docs:
            return {"error": "해당 유형의 표준계약서를 찾을 수 없습니다."}

        print("\nJSON 객체 분석 시작...")
        analysis_results = self._analyze_json_objects_in_parallel(json_objects, standard_docs)
        
        problematic_indices = [i for i, obj in enumerate(analysis_results) if obj.get("has_issues", False)]
        elapsed_time = time.time() - start_time
        result = {
            "document_type": doc_type,
            "document_title": doc_title,
            "total_objects": len(json_objects),
            "problem_objects": problematic_indices,
            "analysis_results": analysis_results,
            "processing_time": f"{elapsed_time:.2f}초"
        }
        print(f"\n계약서 분석 완료! 소요 시간: {elapsed_time:.2f}초")
        return result

    # 분석 결과를 JSON 파일로 저장합니다.
    # 원본 데이터와 분석 결과를 결합하여 결과를 파일에 저장하며, 저장 경로를 반환합니다.
    # 입력: analysis_result (Dict) - 저장할 분석 결과
    #       input_path (str) - 원본 파일 경로 (파일명 생성에 사용)
    # 출력: str - 저장된 파일 경로
    def save_results(self, analysis_result: Dict, input_path: str):
        output_dir = os.path.join(self.project_root, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{filename}_analysis.json")
        
        original_data = {}
        if input_path.lower().endswith('.json'):
            json_path = input_path
        else:
            json_file_name = filename + ".json"
            json_path = os.path.join(self.project_root, "temp", "temp_json", json_file_name)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data_list = json.load(f)
                for item in original_data_list:
                    original_data[str(item.get('page', ''))] = item
        except Exception as e:
            print(f"원본 데이터 로드 실패: {e}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            simple_copy = {key: value for key, value in analysis_result.items()}
            simple_copy["analysis_results"] = [
                {
                    "page_info": {
                        "제목": obj.get("title", original_data.get(str(obj.get("page", "")), {}).get("title", "")),
                        "장": obj.get("chapter", original_data.get(str(obj.get("page", "")), {}).get("chapter", "")),
                        "절": obj.get("section", original_data.get(str(obj.get("page", "")), {}).get("section", "")),
                        "조": obj.get("article", original_data.get(str(obj.get("page", "")), {}).get("article", "")),
                        "쪽": obj.get("page", "정보 없음")
                    },
                    "has_issues": obj.get("has_issues", False),
                    "analysis": obj.get("analysis", {}),
                    "reference_metadata": obj.get("reference_metadata", [])
                }
                for obj in analysis_result["analysis_results"]
            ]
            json.dump(simple_copy, f, ensure_ascii=False, indent=2)
        print(f"분석 결과가 {output_file}에 저장되었습니다.")
        return output_file

if __name__ == "__main__":
    import sys
    analyzer = DocumentAnalyzer()
    input_path = sys.argv[1] if len(sys.argv) > 1 else input("분석할 PDF 또는 JSON 파일 경로를 입력하세요: ")
    if not os.path.exists(input_path):
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    result = analyzer.analyze_contract(input_path)
    if "error" not in result:
        analyzer.save_results(result, input_path)
        if result.get("problem_objects"):
            print(f"\n문제가 있는 객체 인덱스: {result['problem_objects']}")
    else:
        print(f"\n오류: {result['error']}")