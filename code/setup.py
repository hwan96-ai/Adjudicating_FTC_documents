import os
from document_processor import DocumentProcessor
from document_to_db import DocumentToDB
from unfair_contract_generator import process_all_pdfs_in_folder
import logging

# 로깅 비활성화
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

def setup():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectordb_dir = os.path.join(project_root, "data", "vectordb")
    input_folder = os.path.join(project_root, "data", "file")
    temp_folder = os.path.join(project_root, "temp", "temp_json")
    unfair_output_folder = os.path.join(project_root, "data", "unfair_contracts")

    if os.path.exists(vectordb_dir):
        os.system(f"chmod -R 777 {vectordb_dir}")
    else:
        os.makedirs(vectordb_dir, exist_ok=True)
        os.system(f"chmod -R 777 {vectordb_dir}")

    # 1. PDF를 JSON으로 변환하고 병합
    processor = DocumentProcessor()
    processor.process_pdfs_in_folder()
    processor.merge_json_files()

    # 2. JSON을 ChromaDB에 저장
    db_processor = DocumentToDB()
    db_processor.process_to_db()

    # 3. 불공정 계약서 생성
    process_all_pdfs_in_folder(input_folder, temp_folder, unfair_output_folder)

if __name__ == "__main__":
    setup()