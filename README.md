# Contract Analysis and Unfair Clause Generator

이 프로젝트는 PDF 또는 JSON 형식의 하도급 계약서를 분석하고, 표준 계약서와 비교하여 불공정 조항을 식별하거나 생성하는 도구입니다. AWS Bedrock의 LLM(Claude 모델), Chroma 벡터 DB, HuggingFace 임베딩 모델을 활용하여 계약서를 처리하고 분석합니다.

## 주요 기능
1. **계약서 분석**: 사용자 계약서를 표준 계약서와 비교하여 변경된 조항, 누락된 조항, 추가된 조항을 식별.
2. **불공정 조항 생성**: 표준 계약서를 기반으로 불공정 조항을 생성하여 테스트 데이터로 활용 가능.
3. **문서 분류**: 입력 문서의 유형을 자동으로 분류.
4. **결과 요약**: 분석 결과를 JSON 및 Excel 형식으로 저장하고 요약 통계 제공.

## 요구 사항
- Python 3.8 이상
- AWS 계정 및 Bedrock 접근 권한
- CUDA 지원 GPU (임베딩 모델 실행 시 권장)

## 설치 방법

1. **리포지토리 클론**
```bash
   git clone https://github.com/hwan96-ai/Adjudicating_FTC_documents.git
   cd Adjudicating_FTC_documents
```
3. **가상 환경 생성 및 활성화**
```bash 
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows 
```
3. **의존성 설치**
    `pip install -r requirements.txt`
4. **환경 변수 설정**
   프로젝트 루트에 .env 파일을 생성하고 다음 내용을 추가하세요:

```bash    
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_REGION=us-east-1
```


## 사용법
1. **원본 계약서 VectorDB 생성 및 불공정 계약서 생성성**
   setup.py를 통해 data/file에 있는 원본 하도급 PDF 처리, 벡터 DB 생성, 불공정 계약서 생성을 한 번에 실행할 수 있습니다
   `python setup.py`   
3. **전체 파이프라인 실행**
   main.py를 실행하여 사용자의 PDF 또는 JSON 파일을 분석합니다.
   `python main.py`
5. **분석 결과 요약**
    analyze_json.py를 실행하여 분석된 JSON 파일을 요약하고 Excel로 저장합니다.

    `python analyze_json.py`

    analysis_results/ 폴더에서 JSON 파일을 선택합니다.
    요약 통계와 Excel 파일이 생성됩니다.

## 프로젝트 구조
    contract-analyzer/
    ├── data/
    │   ├── file/              # 입력 PDF 파일
    │   ├── json/              # 변환된 JSON 파일
    │   ├── merge/             # 병합된 JSON 파일
    │   ├── vectordb/          # Chroma 벡터 DB
    │   ├── temp/              # 임시 JSON 파일
    │   ├── doc_mapping.json   # 문서 유형 매핑
    │   └── analysis_results/  # 분석 결과 JSON 및 Excel
    ├── document_processor.py  # PDF를 JSON으로 변환
    ├── document_to_db.py      # JSON을 벡터 DB로 변환
    ├── document_classifier.py # 문서 유형 분류
    ├── document_analyzer.py   # 계약서 분석
    ├── unfair_contract_generator.py # 불공정 조항 생성 (구현 필요)
    ├── analyze_json.py        # 분석 결과 요약
    ├── main.py                # 메인 실행 스크립트
    ├── setup.py               # 전체 파이프라인 실행 (구현 필요)
    ├── .env                   # 환경 변수
    ├── requirements.txt       # 의존성 목록
    └── README.md              # 프로젝트 설명

## 주요 모듈 설명
- document_processor.py: PDF 파일을 구조화된 JSON으로 변환합니다.
- document_classifier.py: 입력 문서의 유형을 LLM으로 분류합니다.
- document_to_db.py: JSON 데이터를 Chroma 벡터 DB로 변환합니다.
- document_analyzer.py: 사용자 문서를 표준 계약서와 비교하여 문제점을 분석합니다.
- unfair_contract_generator.py: 불공정 조항을 생성합니다 (구현 필요 시 확인).
- analyze_json.py: 분석 결과를 요약하고 Excel로 출력합니다.
- main.py: 사용자 인터페이스로 분석을 실행합니다.

## 문제 해결
- AWS 인증 오류: .env 파일의 자격 증명을 확인하세요.
- CUDA 관련 오류: GPU가 없으면 model_kwargs={"device": "cpu"}로 수정하세요.
- 파일 경로 오류: 절대 경로를 사용하거나 data/file/에 파일을 배치하세요.


## 연락처
    문의 사항은 hjh1210@saltware.co.kr로 연락 주세요.

### 개선 포인트
1. **Hwp to PDF** [공정거래위원회](https://www.ftc.go.kr/www/selectBbsNttList.do?bordCd=202&key=203) 에서 다운 받은 하도급 문서들 hwp to pdf 구현 
2. **OCR 기능 추가가**: OCR을 통해 실제 사용자의 문서를 넣고 불공정 계약인지 파악하는 기능 추가가
3. **법률 조항 추가가**: 불공정 계약이라 판단된 곳은 법률 조항까지 넣어서 설명해주는 기능 추가가

