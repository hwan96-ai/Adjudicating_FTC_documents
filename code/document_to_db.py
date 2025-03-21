import os
import json
import torch
import gc
import shutil
import time
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentToDB:

    def __init__(self, merged_json_path: str = "data/merge/merged_contracts.json", vectordb_dir: str = "data/vectordb"):
    # DocumentToDB 클래스의 인스턴스를 초기화합니다.
    # JSON 파일 경로와 벡터 DB 저장 경로를 설정하고, 임베딩 모델을 초기화합니다.
    # 입력: merged_json_path (str, 기본값 "data/merge/merged_contracts.json") - 병합된 JSON 파일 경로
    #       vectordb_dir (str, 기본값 "data/vectordb") - 벡터 DB 저장 디렉토리 경로
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.merged_json_path = os.path.join(self.project_root, merged_json_path)
        self.vectordb_dir = os.path.join(self.project_root, vectordb_dir)
        self.collection_name = "contracts_collection"
        self.embeddings = self._setup_embeddings()
        if not self.embeddings:
            raise RuntimeError("임베딩 모델 초기화에 실패했습니다.")

    def _setup_embeddings(self):
    # HuggingFace 임베딩 모델을 설정합니다.
    # CUDA가 사용 가능하면 GPU를 활용하고, 실패 시 None을 반환합니다.
    # 입력: 없음
    # 출력: HuggingFaceEmbeddings 객체 또는 None (초기화 실패 시)
        model_name = "dragonkue/BGE-m3-ko"
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"임베딩 모델 초기화 완료 - 모델: {model_name}, 디바이스: {model_kwargs['device']}")
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 오류: {str(e)}")
            return None

    def _load_merged_json(self) -> list:
    # 병합된 JSON 파일을 로드합니다.
    # 지정된 경로에서 JSON 파일을 읽고, 오류 발생 시 빈 리스트를 반환합니다.
    # 입력: 없음 (self.merged_json_path 사용)
    # 출력: list - 로드된 JSON 데이터 리스트 (오류 시 빈 리스트)
        logger.info(f"병합된 JSON 파일 '{self.merged_json_path}' 로드 중...")
        try:
            with open(self.merged_json_path, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            logger.info(f"총 {len(merged_data)}개 항목을 로드했습니다.")
            return merged_data
        except Exception as e:
            logger.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
            return []

    def _convert_json_to_documents(self, merged_data: list) -> list:
    # JSON 데이터를 LangChain Document 객체로 변환합니다.
    # 각 JSON 항목에서 메타데이터와 콘텐츠를 추출하여 Document 객체 리스트로 만듭니다.
    # 입력: merged_data (list) - 변환할 JSON 데이터 리스트
    # 출력: list - Document 객체 리스트
        logger.info("Document 객체 생성 시작...")
        documents = []
        
        for item in tqdm(merged_data, desc="Document 객체 생성 중"):
            metadata = {
                "title": item.get("title", ""),
                "chapter": item.get("chapter", ""),
                "section": item.get("section", ""),
                "article": item.get("article", ""),
                "page": item.get("page", 0),
                "source_file": item.get("source_file", "")
            }
            if "source_file" in item:
                metadata["doc"] = item["source_file"]
            elif "doc" in item:
                metadata["doc"] = item["doc"]
            content = item.get("text", "")
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"총 {len(documents)}개의 Document 객체가 생성되었습니다.")
        return documents

    def create_vector_db(self, documents: list, batch_size: int = 1) -> bool:
    # Document 객체 리스트를 ChromaDB에 저장하여 벡터 데이터베이스를 생성합니다.
    # 기존 벡터 DB 디렉토리를 삭제하고 새로 생성하며, 배치 단위로 문서를 추가합니다.
    # 입력: documents (list) - 저장할 Document 객체 리스트
    #       batch_size (int, 기본값 1) - 한 번에 처리할 배치 크기 > 필자는 배치 2만가도 gpu 과부화가 옴
    # 출력: bool - 생성 성공 여부 (True: 성공, False: 실패)
        logger.info("ChromaDB 전체 처리 시작...")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if os.path.exists(self.vectordb_dir):
            try:
                shutil.rmtree(self.vectordb_dir)
                logger.info(f"기존 디렉토리 제거: {self.vectordb_dir}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"기존 디렉토리 제거 중 오류: {str(e)}")
        os.makedirs(self.vectordb_dir, exist_ok=True)
        
        if not self.embeddings:
            logger.error("임베딩 모델이 초기화되지 않았습니다.")
            return False
        
        try:
            chroma_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectordb_dir,
                collection_name=self.collection_name
            )
            
            for i in tqdm(range(0, len(documents), batch_size), desc="ChromaDB에 문서 추가 중"):
                batch = documents[i:i + batch_size]
                chroma_db.add_documents(batch)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            logger.info("ChromaDB 저장 완료")
        except Exception as e:
            logger.error(f"ChromaDB 생성 실패: {str(e)}")
            return False
        
        total_time = time.time() - start_time
        logger.info(f"ChromaDB 생성 완료: 총 소요 시간 {total_time:.2f}초, 처리된 문서 수 {len(documents)}개, 저장 위치 {self.vectordb_dir}")
        return True

    def process_to_db(self) -> bool:
    # JSON 데이터를 로드하고 벡터 DB로 변환하는 전체 프로세스를 실행합니다.
    # JSON 로드, Document 변환, 벡터 DB 생성 과정을 순차적으로 수행하며,
    # 각 단계에서 오류 발생 시 False를 반환합니다.
    # 입력: 없음
    # 출력: bool - 전체 처리 성공 여부 (True: 성공, False: 실패)
        merged_data = self._load_merged_json()
        if not merged_data:
            logger.error("JSON 데이터를 로드할 수 없습니다.")
            return False
        
        documents = self._convert_json_to_documents(merged_data)
        if not documents:
            logger.error("Document 객체를 생성할 수 없습니다.")
            return False
        
        success = self.create_vector_db(documents)
        return success

if __name__ == "__main__":
    try:
        db_processor = DocumentToDB()
        success = db_processor.process_to_db()
        
        if success:
            logger.info("모든 처리가 완료되었습니다.")
        else:
            logger.error("처리가 완료되지 않았습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")