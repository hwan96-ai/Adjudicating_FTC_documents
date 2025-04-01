import pdfplumber
import re
import json
import os
from typing import List, Dict
import logging
import warnings

class DocumentProcessor:
    def __init__(self):
        """초기화: pdfplumber 로깅 레벨 설정 및 프로젝트 루트 경로 설정"""
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
        # 수정 전: self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _extract_titles_from_all_pages(self, pdf_path: str) -> tuple[Dict[str, str], int]:
        """PDF의 모든 페이지에서 제목을 추출하여 딕셔너리와 페이지 수를 반환"""
        with pdfplumber.open(pdf_path) as pdf:
            titles = {}
            chapter_pattern = r'제\s*(\d+)\s*장'
            section_pattern = r'제\s*(\d+)\s*절'

            for i, page in enumerate(pdf.pages, start=1):
                page_width, page_height = page.width, page.height
                is_landscape = page_width > page_height * 1.2

                if is_landscape:
                    halves = [
                        {"name": "left", "bbox": (0, 0, page_width/2, page_height)},
                        {"name": "right", "bbox": (page_width/2, 0, page_width, page_height)}
                    ]
                    for j, half in enumerate(halves, start=1):
                        half_title = self._extract_title_from_section(page, half["bbox"], chapter_pattern, section_pattern)
                        if half_title:
                            titles[f"{i}-{j}"] = half_title
                else:
                    page_title = self._extract_title_from_section(page, None, chapter_pattern, section_pattern)
                    if page_title:
                        titles[str(i)] = page_title

            return titles, len(pdf.pages)

    def _extract_title_from_section(self, page, bbox=None, chapter_pattern=None, section_pattern=None) -> str | None:
        """페이지 또는 특정 섹션에서 제목을 추출"""
        # 1단계: bbox 처리
        if bbox:
            section = page.crop(bbox)
            words = section.extract_words()
            section_left = bbox[0]
            section_right = bbox[2]
            section_width = section_right - section_left
            section_center_x = (section_left + section_right) / 2
        else:
            words = page.extract_words()
            section_left = 0
            section_right = page.width
            section_width = page.width
            section_center_x = page.width / 2

        if not words:
            return None

        # 2단계: 단어 그룹화
        lines = {}
        for word in words:
            y = round(word['top'])
            if y not in lines:
                lines[y] = []
            lines[y].append(word)
        sorted_lines = sorted(lines.items())

        # 3단계: 첫 번째 라인 분석
        if sorted_lines:
            first_line_words = sorted_lines[0][1]
            first_line_text = ' '.join(w['text'] for w in first_line_words)
            attachment_pattern = r'^【첨부\s*\d*】'
            appendix_pattern = r'^【별첨\s*\d*】'
            if re.match(attachment_pattern, first_line_text) or re.match(appendix_pattern, first_line_text):
                lines_to_check = sorted_lines[:2]
            else:
                lines_to_check = sorted_lines[:1]
        else:
            lines_to_check = []

        # 4단계: 제목 후보 선정
        potential_titles = []
        for i, (y, line_words) in enumerate(lines_to_check):
            has_large_gap = False
            if len(line_words) >= 2:
                sorted_words = sorted(line_words, key=lambda w: w['x0'])
                for j in range(len(sorted_words) - 1):
                    gap = sorted_words[j+1]['x0'] - sorted_words[j]['x1']
                    char_width = (sorted_words[j]['x1'] - sorted_words[j]['x0']) / max(1, len(sorted_words[j]['text']))
                    if gap > char_width * 2:
                        has_large_gap = True
                        break
            if has_large_gap:
                continue

            line_text = ' '.join(w['text'] for w in sorted(line_words, key=lambda w: w['x0']))
            if chapter_pattern and section_pattern:
                if re.search(chapter_pattern, line_text) or re.search(section_pattern, line_text):
                    continue

            if line_words:
                leftmost = min(w['x0'] for w in line_words)
                rightmost = max(w['x1'] for w in line_words)
                line_width = rightmost - leftmost
                line_center = leftmost + (line_width / 2)
                is_centered = abs(line_center - section_center_x) < (section_width * 0.1)
                if is_centered:
                    potential_titles.append((i, line_text))

        # 5단계: 최종 제목 선택
        if potential_titles:
            selected_title = potential_titles[0][1]
            selected_title = re.sub(r'\s+', ' ', selected_title).strip()
            allowed_endings = ('계약서', '계약서(본문)', '계약서(전문)', '계약서(표지)', '명세서', '합의서', '동표')
            if selected_title.endswith(allowed_endings):
                return selected_title
            return None
        return None

    def _merge_json_entries(self, json_list: List[Dict]) -> List[Dict]:
        """제목이 동일한 항목을 병합"""
        merged_list = []
        i = 0
        while i < len(json_list):
            current = json_list[i]
            text_without_title = current['text'].replace(current['title'], '').strip()
            if len(text_without_title) <= 25 and i + 1 < len(json_list) and current['title'] == json_list[i + 1]['title']:
                next_entry = json_list[i + 1]
                merged_list.append({
                    "title": current['title'],
                    "chapter": next_entry['chapter'] or current['chapter'],
                    "section": next_entry['section'] or current['section'],
                    "article": next_entry['article'] or current['article'],
                    "text": f"{current['text']} {next_entry['text']}",
                    "page": current['page']
                })
                i += 2
            else:
                merged_list.append(current)
                i += 1
        return merged_list

    def _process_page_section(self, page, bbox, titles, section_key, structured_data, state):
        """페이지 섹션을 처리하여 구조화된 데이터를 업데이트"""
        section = page.crop(bbox) if bbox else page
        text = section.extract_text()
        if not text:
            return state

        current_title, current_chapter, current_section, current_article, current_content, start_page = state
        chapter_pattern = re.compile(r'제\s*\d+\s*장\s*[^\n]+')
        section_pattern = re.compile(r'제\s*\d+\s*절\s*[^\n]+')
        article_pattern = re.compile(r'제\s*\d+\s*조(의\d+)?\s*\([^\)]+\)\s+')

        if section_key in titles:
            if current_content.strip():
                structured_data.append({
                    "title": current_title, "chapter": current_chapter, "section": current_section,
                    "article": current_article.rstrip(), "text": current_content.strip(), "page": start_page
                })
            current_title, current_chapter, current_section, current_article, current_content = titles[section_key], "", "", "", ""
            start_page = section_key

        for line in [line.strip() for line in text.split('\n') if line.strip()]:
            if chapter_pattern.match(line):
                if current_content.strip():
                    structured_data.append({
                        "title": current_title, "chapter": current_chapter, "section": current_section,
                        "article": current_article.rstrip(), "text": current_content.strip(), "page": start_page
                    })
                current_chapter, current_section, current_article, current_content = line, "", "", ""
                start_page = section_key
            elif section_pattern.match(line):
                if current_content.strip():
                    structured_data.append({
                        "title": current_title, "chapter": current_chapter, "section": current_section,
                        "article": current_article.rstrip(), "text": current_content.strip(), "page": start_page
                    })
                current_section, current_article, current_content = line, "", ""
                start_page = section_key
            elif article_pattern.match(line):
                if current_content.strip():
                    structured_data.append({
                        "title": current_title, "chapter": current_chapter, "section": current_section,
                        "article": current_article.rstrip(), "text": current_content.strip(), "page": start_page
                    })
                article_match = article_pattern.match(line)
                current_article, current_content = article_match.group(0), line[len(article_match.group(0)):].strip()
                start_page = section_key
            else:
                current_content += " " + line

        return current_title, current_chapter, current_section, current_article, current_content, start_page

    def process_pdf_to_json(self, pdf_path: str, output_json_path: str) -> List[Dict]:
        """PDF를 JSON으로 변환"""
        titles, total_pages = self._extract_titles_from_all_pages(pdf_path)
        print(titles, total_pages)

        structured_data = []
        state = ("", "", "", "", "", None)

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_width, page_height = page.width, page.height
                is_landscape = page_width > page_height * 1.2
                if is_landscape:
                    halves = [
                        {"name": "left", "bbox": (0, 0, page_width/2, page_height)},
                        {"name": "right", "bbox": (page_width/2, 0, page_width, page_height)}
                    ]
                    for j, half in enumerate(halves, start=1):
                        state = self._process_page_section(page, half["bbox"], titles, f"{i}-{j}", structured_data, state)
                else:
                    state = self._process_page_section(page, None, titles, str(i), structured_data, state)

            if state[4].strip():
                structured_data.append({
                    "title": state[0], "chapter": state[1], "section": state[2],
                    "article": state[3].rstrip(), "text": state[4].strip(), "page": state[5]
                })

        merged_data = self._merge_json_entries(structured_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        return merged_data

    def process_pdfs_in_folder(self, input_folder: str = "data/file", output_folder: str = "data/json"):
        """폴더 내 모든 PDF를 JSON으로 변환"""
        input_folder = os.path.join(self.project_root, input_folder)
        output_folder = os.path.join(self.project_root, output_folder)
        os.makedirs(output_folder, exist_ok=True)
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"'{input_folder}' 폴더에 PDF 파일이 없습니다.")
            return
        
        print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            json_path = os.path.join(output_folder, os.path.splitext(pdf_file)[0] + ".json")
            print(f"처리 중: {pdf_file} -> {os.path.basename(json_path)}")
            try:
                self.process_pdf_to_json(pdf_path, json_path)
            except Exception as e:
                print(f"오류 발생: {pdf_file} 처리 중 - {str(e)}")

    def merge_json_files(self, json_folder: str = "data/json", merge_folder: str = "data/merge", output_filename: str = "merged_contracts.json"):
        """JSON 폴더 내 모든 JSON 파일을 읽어 하나의 파일로 병합합니다."""
        json_folder = os.path.join(self.project_root, json_folder)
        merge_folder = os.path.join(self.project_root, merge_folder)
        os.makedirs(merge_folder, exist_ok=True)  # 이미 있어도 에러 안 나게
        
        json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
        if not json_files:
            print(f"'{json_folder}' 폴더에 JSON 파일이 없습니다.")
            return
        
        print(f"총 {len(json_files)}개의 JSON 파일을 병합합니다.")
        all_data = []
        for json_file in json_files:
            json_path = os.path.join(json_folder, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"경고: {json_file}은 리스트 형식이 아님, 건너뜀")
                        continue
                    if not data:
                        print(f"경고: {json_file}이 비어 있음, 건너뜀")
                        continue
                    for item in data:
                        item['source_file'] = os.path.splitext(json_file)[0]
                    all_data.extend(data)
            except Exception as e:
                print(f"오류 발생: {json_file} 읽기 중 - {str(e)}")
        
        if not all_data:
            print("병합할 데이터가 없습니다.")
            return
        
        merged_file_path = os.path.join(merge_folder, output_filename)
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"병합 완료: {len(all_data)}개 항목이 '{merged_file_path}'에 저장되었습니다.")
