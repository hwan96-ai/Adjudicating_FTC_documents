import pdfplumber
import re
import json
import os
from pathlib import Path
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _extract_titles_from_all_pages(self, pdf_path: str) -> tuple[Dict[str, str], int]:
        #PDF의 모든 페이지를 순회하며 가로/세로 페이지 구분
        #가로 페이지면 반으로 나눔
        #_extract_title_from_section호출해서 제목 추출
        #페이지 번호 매핑, 가로 페이지도 페이지 매핑
        #총 페이지 계산
        with pdfplumber.open(pdf_path) as pdf:
            titles = {}
            sequential_page = 1
            chapter_pattern = r'제\s*(\d+)\s*장'
            section_pattern = r'제\s*(\d+)\s*절'
            for page in pdf.pages:
                page_width = page.width
                page_height = page.height
                is_landscape = page_width > page_height * 1.2
                if is_landscape:
                    halves = [
                        {"name": "left", "bbox": (0, 0, page_width/2, page_height)},
                        {"name": "right", "bbox": (page_width/2, 0, page_width, page_height)}
                    ]
                    for half in halves:
                        half_title = self._extract_title_from_section(page, half["bbox"], chapter_pattern, section_pattern)
                        if half_title:
                            titles[str(sequential_page)] = half_title
                        sequential_page += 1
                else:
                    page_title = self._extract_title_from_section(page, None, chapter_pattern, section_pattern)
                    if page_title:
                        titles[str(sequential_page)] = page_title
                    sequential_page += 1
            total_pages = sequential_page - 1
            return titles, total_pages

    def _extract_title_from_section(self, page, bbox=None, chapter_pattern=None, section_pattern=None) -> str | None:
    # 페이지 또는 특정 섹션에서 제목을 추출
    # 단어를 줄 단위로 그룹화하고, 중앙에 위치하며 허용된 형식의 제목만을 반환.
    # 제목 뽑는 조건문    
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
        lines = {}
        for word in words:
            y = round(word['top'])
            if y not in lines:
                lines[y] = []
            lines[y].append(word)
        sorted_lines = sorted(lines.items())
        if sorted_lines:
            first_line_words = sorted(sorted_lines[0][1], key=lambda w: w['x0'])
            first_line_text = ' '.join(w['text'] for w in first_line_words)
            attachment_pattern = r'^【첨부\s*\d*】'
            appendix_pattern = r'^【별첨\s*\d*】'
            if re.match(attachment_pattern, first_line_text) or re.match(appendix_pattern, first_line_text):
                lines_to_check = sorted_lines[:2]
            else:
                lines_to_check = sorted_lines[:1]
        else:
            lines_to_check = []
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
                if is_centered and len(line_text.strip()) <= 40:
                    potential_titles.append((i, line_text))
        if potential_titles:
            selected_title = potential_titles[0][1]
            selected_title = re.sub(r'\s+', ' ', selected_title).strip()
            allowed_endings = ('계약서', '계약서(본문)', '계약서(전문)', '계약서(표지)', '명세서', '합의서', '동표')
            if selected_title.endswith(allowed_endings):
                return selected_title
            return None
        return None

    def _merge_json_entries(self, json_list: List[Dict]) -> List[Dict]:
    # JSON 데이터 리스트에서 제목이 동일한 항목을 병합합니다.
    # 제목을 제외한 텍스트가 25자 이하이고 다음 항목과 제목이 같으면 텍스트를 합치며,
    # chapter, section, article은 빈 값이 아닌 것을 우선 사용합니다.
    # 입력: json_list (List[Dict]) - 병합할 JSON 데이터 리스트, 각 항목은 title, chapter, section, article, text, page 포함
    # 출력: List[Dict] - 병합된 JSON 데이터 리스트
        merged_list = []
        i = 0
        while i < len(json_list):
            current = json_list[i]
            text_without_title = current['text'].replace(current['title'], '').strip()
            if len(text_without_title) <= 25 and i + 1 < len(json_list):
                next_entry = json_list[i + 1]
                if current['title'] == next_entry['title']:
                    chapter = next_entry['chapter'] if next_entry['chapter'] != "" else current['chapter']
                    section = next_entry['section'] if next_entry['section'] != "" else current['section']
                    article = next_entry['article'] if next_entry['article'] != "" else current['article']
                    merged_text = current['text'] + " " + next_entry['text']
                    merged_entry = {
                        "title": current['title'],
                        "chapter": chapter,
                        "section": section,
                        "article": article,
                        "text": merged_text,
                        "page": current['page']
                    }
                    merged_list.append(merged_entry)
                    i += 2
                else:
                    merged_list.append({
                        "title": current['title'],
                        "chapter": current['chapter'],
                        "section": current['section'],
                        "article": current['article'],
                        "text": current['text'],
                        "page": current['page']
                    })
                    i += 1
            else:
                merged_list.append({
                    "title": current['title'],
                    "chapter": current['chapter'],
                    "section": current['section'],
                    "article": current['article'],
                    "text": current['text'],
                    "page": current['page']
                })
                i += 1
        return merged_list

    def process_pdf_to_json(self, pdf_path: str, output_json_path: str) -> List[Dict]:
    # PDF 파일을 읽어 제목, 장, 절, 조로 구조화된 JSON 데이터로 변환하고 파일에 저장합니다.
    # 가로 페이지는 좌우로 분할하며, 페이지별 제목을 기준으로 데이터를 분리하고,
    # 정규식을 사용해 장, 절, 조를 인식한 뒤 병합된 데이터를 생성합니다.
    # 입력: pdf_path (str) - PDF 파일 경로, output_json_path (str) - JSON 저장 경로
    # 출력: List[Dict] - 구조화된 데이터 리스트 (title, chapter, section, article, text, page 포함)
        titles = self._extract_titles_from_all_pages(pdf_path)
        chapter_pattern = re.compile(r'제\s*\d+\s*장\s*[^\n]+')
        section_pattern = re.compile(r'제\s*\d+\s*절\s*[^\n]+')
        article_pattern = re.compile(r'제\s*\d+\s*조(의\d+)?\s*\([^\)]+\)\s+')
        structured_data = []
        current_title = ""
        current_chapter = ""
        current_section = ""
        current_article = ""
        current_content = ""
        start_page = 1
        with pdfplumber.open(pdf_path) as pdf:
            sequential_page = 1
            for page in pdf.pages:
                page_width = page.width
                page_height = page.height
                is_landscape = page_width > page_height * 1.2
                if is_landscape:
                    halves = [
                        {"name": "left", "bbox": (0, 0, page_width/2, page_height)},
                        {"name": "right", "bbox": (page_width/2, 0, page_width, page_height)}
                    ]
                    for half in halves:
                        section = page.crop(half["bbox"])
                        text = section.extract_text()
                        if not text:
                            sequential_page += 1
                            continue
                        if str(sequential_page) in titles:
                            if current_content.strip():
                                structured_data.append({
                                    "title": current_title,
                                    "chapter": current_chapter,
                                    "section": current_section,
                                    "article": current_article.rstrip(),
                                    "text": current_content.strip(),
                                    "page": start_page
                                })
                            current_title = titles[str(sequential_page)]
                            current_chapter = ""
                            current_section = ""
                            current_article = ""
                            current_content = ""
                            start_page = sequential_page
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            if chapter_pattern.match(line):
                                if current_content.strip():
                                    structured_data.append({
                                        "title": current_title,
                                        "chapter": current_chapter,
                                        "section": current_section,
                                        "article": current_article.rstrip(),
                                        "text": current_content.strip(),
                                        "page": start_page
                                    })
                                current_chapter = line
                                current_section = ""
                                current_article = ""
                                current_content = ""
                                start_page = sequential_page
                            elif section_pattern.match(line):
                                if current_content.strip():
                                    structured_data.append({
                                        "title": current_title,
                                        "chapter": current_chapter,
                                        "section": current_section,
                                        "article": current_article.rstrip(),
                                        "text": current_content.strip(),
                                        "page": start_page
                                    })
                                current_section = line
                                current_article = ""
                                current_content = ""
                                start_page = sequential_page
                            elif article_pattern.match(line):
                                if current_content.strip():
                                    structured_data.append({
                                        "title": current_title,
                                        "chapter": current_chapter,
                                        "section": current_section,
                                        "article": current_article.rstrip(),
                                        "text": current_content.strip(),
                                        "page": start_page
                                    })
                                article_match = article_pattern.match(line)
                                current_article = article_match.group(0)
                                current_content = line[len(article_match.group(0)):].strip()
                                start_page = sequential_page
                            else:
                                current_content += " " + line.strip()
                        sequential_page += 1
                else:
                    text = page.extract_text()
                    if not text:
                        sequential_page += 1
                        continue
                    if str(sequential_page) in titles:
                        if current_content.strip():
                            structured_data.append({
                                "title": current_title,
                                "chapter": current_chapter,
                                "section": current_section,
                                "article": current_article.rstrip(),
                                "text": current_content.strip(),
                                "page": start_page
                            })
                        current_title = titles[str(sequential_page)]
                        current_chapter = ""
                        current_section = ""
                        current_article = ""
                        current_content = ""
                        start_page = sequential_page
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if chapter_pattern.match(line):
                            if current_content.strip():
                                structured_data.append({
                                    "title": current_title,
                                    "chapter": current_chapter,
                                    "section": current_section,
                                    "article": current_article.rstrip(),
                                    "text": current_content.strip(),
                                    "page": start_page
                                })
                            current_chapter = line
                            current_section = ""
                            current_article = ""
                            current_content = ""
                            start_page = sequential_page
                        elif section_pattern.match(line):
                            if current_content.strip():
                                structured_data.append({
                                    "title": current_title,
                                    "chapter": current_chapter,
                                    "section": current_section,
                                    "article": current_article.rstrip(),
                                    "text": current_content.strip(),
                                    "page": start_page
                                })
                            current_section = line
                            current_article = ""
                            current_content = ""
                            start_page = sequential_page
                        elif article_pattern.match(line):
                            if current_content.strip():
                                structured_data.append({
                                    "title": current_title,
                                    "chapter": current_chapter,
                                    "section": current_section,
                                    "article": current_article.rstrip(),
                                    "text": current_content.strip(),
                                    "page": start_page
                                })
                            article_match = article_pattern.match(line)
                            current_article = article_match.group(0)
                            current_content = line[len(article_match.group(0)):].strip()
                            start_page = sequential_page
                        else:
                            current_content += " " + line.strip()
                    sequential_page += 1
        if current_content.strip():
            structured_data.append({
                "title": current_title,
                "chapter": current_chapter,
                "section": current_section,
                "article": current_article.rstrip(),
                "text": current_content.strip(),
                "page": start_page
            })
        merged_data = self._merge_json_entries(structured_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

        return merged_data

    def process_pdfs_in_folder(self, input_folder: str = "data/file", output_folder: str = "data/json"):
    # 지정된 폴더 내 모든 PDF 파일을 JSON으로 변환합니다.
    # 출력 폴더가 없으면 생성하고, 각 PDF 파일을 처리하여 JSON 파일로 저장하며,
    # 오류 발생 시 메시지를 출력합니다.
    # 입력: input_folder (str, 기본값 "data/file") - PDF 파일이 있는 폴더,
    #       output_folder (str, 기본값 "data/json") - JSON 파일을 저장할 폴더
    # 출력: 없음 (파일로 저장됨)
        input_folder = os.path.join(self.project_root, input_folder)
        output_folder = os.path.join(self.project_root, output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"출력 폴더 '{output_folder}'를 생성했습니다.")
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"'{input_folder}' 폴더에 PDF 파일이 없습니다.")
            return
        print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            json_file = os.path.splitext(pdf_file)[0] + ".json"
            json_path = os.path.join(output_folder, json_file)
            print(f"처리 중: {pdf_file} -> {json_file}")
            try:
                self.process_pdf_to_json(pdf_path, json_path)
            except Exception as e:
                print(f"오류 발생: {pdf_file} 처리 중 - {str(e)}")

    def merge_json_files(self, json_folder: str = "data/json", merge_folder: str = "data/merge"):
        # 폴더 내 모든 JSON 파일을 읽어 하나의 JSON 파일로 병합합니다.
        json_folder = os.path.join(self.project_root, json_folder)
        merge_folder = os.path.join(self.project_root, merge_folder)
        if not os.path.exists(merge_folder):
            os.makedirs(merge_folder)
            print(f"병합 폴더 '{merge_folder}'를 생성했습니다.")
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
                    for item in data:
                        item['source_file'] = os.path.splitext(json_file)[0]
                    all_data.extend(data)
            except Exception as e:
                print(f"오류 발생: {json_file} 읽기 중 - {str(e)}")
        merged_file_path = os.path.join(merge_folder, "merged_contracts.json")
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"병합 완료: {len(all_data)}개 항목이 '{merged_file_path}'에 저장되었습니다.")