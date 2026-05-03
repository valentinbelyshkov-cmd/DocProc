import re
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

DOCUMENT_TYPES = {
    'Счет-фактура': {
        'patterns': [
            r'счет[_\s]фактура',
            r'счет\s+фактура',
            r'счет-фактура',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:счф?|invoice)\s*(?:№|no\.?|number|#)\s*[:\-]?\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(?:от\s*)?(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})', r'(?:дата|date)\s*[:\-]?\s*(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Покупатель', 'patterns': [r'покупатель\s*[:\-]?\s*(.+)', r'(?:buyer|customer)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Продавец', 'patterns': [r'продавец\s*[:\-]?\s*(.+)', r'(?:seller|supplier)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН продавца', 'patterns': [r'инн\s*(?:продавца)?\s*[:\-]?\s*(\d{10,12})'], 'required': True},
            {'name': 'КПП продавца', 'patterns': [r'кпп\s*(?:продавца)?\s*[:\-]?\s*(\d{9})'], 'required': False},
            {'name': 'Всего к оплате', 'patterns': [r'(?:всего|итого|total|sum)\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+)', r'([\d\s,]+)\s*(?:руб|₽|rur)'], 'required': True},
        ]
    },
    'УПД': {
        'patterns': [
            r'универсальный\s*передаточный\s*документ',
            r'упд',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Покупатель', 'patterns': [r'покупатель\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Продавец', 'patterns': [r'продавец\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН продавца', 'patterns': [r'инн\s*(?:продавца)?\s*[:\-]?\s*(\d{10,12})'], 'required': True},
            {'name': 'КПП продавца', 'patterns': [r'кпп\s*(?:продавца)?\s*[:\-]?\s*(\d{9})'], 'required': False},
            {'name': 'Основание', 'patterns': [r'(?:основание|basis)\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Всего к оплате', 'patterns': [r'(?:всего|итого)\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+)'], 'required': True},
        ]
    },
    'Акт': {
        'patterns': [
            r'\bакт\b.*(?:выполнен(?:ны|ых?)|оказанн(?:ая|ых?)|работ)',
            r'(?:акт|act)\s*(?:сдачи|при[её]мки|выполненн(?:ых?|ой))',
            r'акт\s+№',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:акт|act)\s*(?:№|no\.?|#)\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Исполнитель', 'patterns': [r'исполнитель\s*[:\-]?\s*(.+)', r'(?:исп|executor)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН исполнителя', 'patterns': [r'инн\s*(?:исполнителя)?\s*[:\-]?\s*(\d{10,12})'], 'required': False},
            {'name': 'Основание', 'patterns': [r'основание\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Итого', 'patterns': [r'итого\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+)'], 'required': True},
        ]
    },
    'Счет': {
        'patterns': [
            r'счет\s+(?:на\s+)?оплату?',
            r'(?:счёт|счет)\s*(?:на\s+оплату?)?',
            r'(?:invoice|bill)\s*(?:no\.?|number)?',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:счет|счёт)\s*(?:№|no\.?|#)\s*[:\-]?\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Наименование банка', 'patterns': [r'(?:банк|bank)\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Наименование получателя', 'patterns': [r'(?:получатель|recipient)\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Счет', 'patterns': [r'(?:счет|счёт)\s*(?:получател[а-я])?\s*[:\-]?\s*(\d{20})'], 'required': False},
            {'name': 'БИК', 'patterns': [r'(?:бик|bik)\s*[:\-]?\s*(\d{9})'], 'required': False},
            {'name': 'ИНН', 'patterns': [r'инн\s*[:\-]?\s*(\d{10,12})'], 'required': False},
            {'name': 'Основание', 'patterns': [r'основание\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Итого', 'patterns': [r'итого\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+)'], 'required': True},
        ]
    }
}


def detect_document_type(text: str) -> Tuple[Optional[str], float]:
    text_lower = text.lower()
    best_match = None
    best_confidence = 0.0
    
    for doc_type, config in DOCUMENT_TYPES.items():
        matches = 0
        for pattern in config['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        if matches > 0:
            confidence = min(matches / len(config['patterns']), 1.0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = doc_type
    
    return best_match, best_confidence


def extract_field_value(text: str, field_config: Dict) -> Tuple[Optional[str], float]:
    value = None
    confidence = 0.0
    
    for pattern in field_config['patterns']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.groups():
                value = match.group(1).strip()
            else:
                value = match.group(0).strip()
            
            if value:
                confidence = 0.7 + (0.3 * min(len(value) / 20, 1.0))
                
                if field_config['name'] in ['ИНН продавца', 'КПП продавца', 'ИНН исполнителя', 'ИНН', 'БИК', 'Счет']:
                    confidence = min(confidence + 0.1, 0.95)
                break
    
    return value, confidence


def parse_document_fields(text: str, doc_type: str) -> List[Dict]:
    if doc_type not in DOCUMENT_TYPES:
        return []
    
    results = []
    config = DOCUMENT_TYPES[doc_type]
    
    for field_config in config['fields']:
        value, confidence = extract_field_value(text, field_config)
        results.append({
            'field': field_config['name'],
            'value': value or '',
            'confidence': confidence
        })
    
    return results


def find_tables_in_text(text: str) -> List[List[List[str]]]:
    tables = []
    lines = text.split('\n')
    
    table_pattern = re.compile(r'^\s*[\d۰۱]+[\s,.\t]+[\d۰۱]+')
    
    for i, line in enumerate(lines):
        if re.search(r'\d+\s+\d+', line) and len(line.split()) >= 4:
            potential_table_lines = [line]
            
            for j in range(i + 1, min(i + 20, len(lines))):
                next_line = lines[j]
                
                if re.search(r'(?:итого|всего|total|sum)', next_line, re.IGNORECASE):
                    potential_table_lines.append(next_line)
                    break
                
                if re.search(r'\d', next_line):
                    cols = len(re.findall(r'\S+', next_line))
                    if 2 <= cols <= 15:
                        potential_table_lines.append(next_line)
                    else:
                        break
                else:
                    break
            
            if len(potential_table_lines) >= 2:
                table_data = []
                for table_line in potential_table_lines:
                    cells = re.split(r'[\t,|]+', table_line)
                    cells = [c.strip() for c in cells if c.strip()]
                    if cells:
                        table_data.append(cells)
                
                if table_data and any(len(row) >= 2 for row in table_data):
                    tables.append(table_data)
    
    return tables


def extract_numerical_tables(text: str) -> List[List[List[str]]]:
    tables = []
    lines = text.split('\n')
    
    table_started = False
    current_table = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_table and len(current_table) >= 2:
                tables.append(current_table)
                current_table = []
            table_started = False
            continue
        
        numbers = re.findall(r'[\d\s,]+(?:\.\d+)?', line)
        words = re.findall(r'\S+', line)
        
        has_numbers = any(n.strip() for n in numbers)
        has_words = len([w for w in words if re.search(r'[а-яА-Яa-zA-Z]', w)]) > 0
        
        if has_numbers and len(words) >= 2:
            if not table_started and current_table and len(current_table) >= 2:
                if not any(re.search(r'\d', t) for t in current_table[-1]):
                    continue
            
            table_started = True
            cells = re.split(r'[\t]{1,4}|[\s]{2,}', line)
            cells = [c.strip() for c in cells if c.strip()]
            
            if cells:
                current_table.append(cells)
        else:
            if current_table and len(current_table) >= 2:
                tables.append(current_table)
                current_table = []
            table_started = False
    
    if current_table and len(current_table) >= 2:
        tables.append(current_table)
    
    return tables


def parse_ocr_result(pages: List[Dict]) -> Dict[str, Any]:
    full_text = ""
    for page in pages:
        full_text += page.get('markdown', '') + "\n"
        if 'result_json' in page and page['result_json']:
            for p_data in page['result_json']:
                for block in p_data.get('blocks', []):
                    full_text += block.get('text', '') + "\n"
    
    doc_type, type_confidence = detect_document_type(full_text)
    
    if not doc_type:
        return {
            'document_type': None,
            'type_confidence': 0.0,
            'fields': [],
            'tables': [],
            'raw_text': full_text[:5000]
        }
    
    fields = parse_document_fields(full_text, doc_type)
    
    tables = extract_numerical_tables(full_text)
    
    return {
        'document_type': doc_type,
        'type_confidence': type_confidence,
        'fields': fields,
        'tables': tables,
        'raw_text': full_text[:5000]
    }
