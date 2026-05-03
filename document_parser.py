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
            {'name': 'Тип документа', 'patterns': [r'((?:Счет|Sчeт)-фактура)', r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:счет-фактура|счф?|invoice)\s*(?:№|no\.?|number|#)\s*[:\-]?\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(?:счет-фактура|счф?|invoice)\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)', r'(?:от\s*)?(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})', r'(?:дата|date)\s*[:\-]?\s*(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Покупатель', 'patterns': [r'покупатель\s*[:\-]?\s*(.+)', r'(?:buyer|customer)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Продавец', 'patterns': [r'продавец\s*[:\-]?\s*(.+)', r'(?:seller|supplier)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН продавца', 'patterns': [r'инн\s*(?:продавца)?\s*[:\-]?\s*(\d{10,12})'], 'required': True},
            {'name': 'КПП продавца', 'patterns': [r'кпп\s*(?:продавца)?\s*[:\-]?\s*(\d{9})'], 'required': False},
            {'name': 'Всего к оплате', 'patterns': [r'(?:всего|итого|total|sum)\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)', r'([\d\s,]+)\s*(?:руб|₽|rur)'], 'required': True},
        ]
    },
    'УПД': {
        'patterns': [
            r'универсальный\s*передаточный\s*документ',
            r'упд',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(универсальный\s*передаточный\s*документ)', r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:универсальный\s*передаточный\s*документ|номер|no\.?)\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(?:№|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)', r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Покупатель', 'patterns': [r'покупатель\s*[:\-]?\s*(.+)', r'получатель\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Продавец', 'patterns': [r'продавец\s*[:\-]?\s*(.+)', r'поставщик\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН продавца', 'patterns': [r'инн\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{10,12})'], 'required': True},
            {'name': 'КПП продавца', 'patterns': [r'кпп\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{9})'], 'required': False},
            {'name': 'Основание', 'patterns': [r'(?:основание|basis)\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Всего к оплате', 'patterns': [r'(?:всего|итого)\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)'], 'required': True},
        ]
    },
    'Акт': {
        'patterns': [
            r'\bакт\b.*(?:выполнен(?:ны|ых?)|оказанн(?:ая|ых?)|работ)',
            r'(?:акт|act)\s*(?:сдачи|при[её]мки|выполненн(?:ых?|ой))',
            r'акт\s+№',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'(Акт\s+(?:приемки-сдачи|выполненных\s+работ|оказанных\s+услуг)?)', r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:акт|act)\s*(?:№|no\.?|#)\s*[:\-]?\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(?:акт|act)\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)', r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Исполнитель', 'patterns': [r'исполнитель\s*[:\-]?\s*(.+)', r'(?:исп|executor)\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'ИНН исполнителя', 'patterns': [r'инн\s*(?:исполнителя)?\s*[:\-]?\s*(\d{10,12})'], 'required': False},
            {'name': 'Основание', 'patterns': [r'основание\s*[:\-]?\s*(.+)'], 'required': False},
            {'name': 'Итого', 'patterns': [r'итого\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)'], 'required': True},
        ]
    },
    'Счет': {
        'patterns': [
            r'счет\s+(?:на\s+)?оплату?',
            r'(?:счёт|счет)\s*(?:на\s+оплату?)?',
            r'(?:invoice|bill)\s*(?:no\.?|number)?',
        ],
        'fields': [
            {'name': 'Тип документа', 'patterns': [r'((?:Счет|Sчeт)\s+на\s+оплату)', r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'], 'required': True},
            {'name': 'Номер документа', 'patterns': [r'(?:счет|счёт)(?:\s+на\s+оплату)?\s*(?:№|no\.?|#)\s*[:\-]?\s*(\S+)', r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'], 'required': True},
            {'name': 'Дата документа', 'patterns': [r'(?:счет|счёт)(?:\s+на\s+оплату)?\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-яё]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)', r'от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})', r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'], 'required': True},
            {'name': 'Наименование получателя', 'patterns': [
                r'поставщик\s*[:\-]?\s*(?:["\']?)(ООО\s+"[^"]+"|АО\s+"[^"]+"|ПАО\s+"[^"]+")',
                r'поставщик\s*[:\-]?\s*(?:ooo|ооо|ао|пао)?\s*["\']?([\w\s"-]+?)(?:["\']?\s*,|\s*$|\s*инн)',
            ], 'required': True},
            {'name': 'ИНН', 'patterns': [
                r'(?:инн|inn)\s*[:\-]?\s*(\d{10,12})',
            ], 'required': True},
            {'name': 'БИК', 'patterns': [
                r'(?:бик|bik)\s*[:\-]?\s*(\d{9})',
                r'\b(\d{9})\b(?=\s*(?:кпп|инн|р/с|$))',
                r'(?<!\d)(\d{9})(?!\d)',
            ], 'required': False},
            {'name': 'Наименование банка', 'patterns': [
                r'(?:банк\s+получателя|банк получателя)\s*[:\-]?\s*([^\n]+)',
                r'([А-ЯЁ][\w\s"-]{0,30}Банк[\w\s"-]{0,20})',
                r'(?:Банк\s+(?:ПАО\s+|ПAO\s+|ООО\s+|АО\s+))?([^\n]+)',
                r'(ООО\s+"[^"]+"|ПАО\s+"[^"]+")(?=\s+[Г|g]\.|$)',
            ], 'required': False},
            {'name': 'Счет', 'patterns': [
                r'(?:р/с|расч[её]тный\s+сч[её]т|лицевой\s+сч[её]т)\s*[:\-]?\s*(\d{20})',
                r'(?:^|\n)\s*(407\d{17})\s*(?:\n|$)',
                r'(?:^|\n)\s*(\d{20})\s*(?:\n|$)',
            ], 'required': False},
            {'name': 'Основание', 'patterns': [
                r'(?:основание|договор|контракт)\s*[:\-]?\s*(.+)',
                r'(\d{5,}\s+(?:от|OT)\s+\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
            ], 'required': False},
            {'name': 'Итого', 'patterns': [
                r'(?:всего|итого)\s*(?:к\s*оплате|по\s*счету)?\s*[:\-]?\s*([\d\s]+[.,]\d{2})\s*(?:руб|₽|rur)?',
                r'([\d\s]+[.,]\d{2})\s*(?:руб|₽|rur)',
            ], 'required': True},
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
                
                if field_config['name'] == 'Дата документа' and re.search(r'[а-я]{3,}', value, re.IGNORECASE):
                    confidence = min(confidence + 0.2, 0.98)
                    
                break
    
    return value, confidence


def parse_document_fields(text: str, doc_type: str) -> List[Dict]:
    if doc_type not in DOCUMENT_TYPES:
        return []
    
    lines = text.split('\n')
    
    table_start_idx = len(lines)
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if re.match(r'^\s*№?\s*(?:наименование|товар|описание|ед\.|кол-во|количество|сумма)\s*', line_lower, re.IGNORECASE):
            table_start_idx = i
            break
        if re.match(r'^\s*\d+\s+\d+\s+\d+', line.strip()):
            table_start_idx = i
            break
    
    provider_idx = len(lines)
    for i, line in enumerate(lines):
        if re.search(r'поставщик\s*[:\-]?\s*', line, re.IGNORECASE):
            provider_idx = i
            break
    
    bank_section = '\n'.join(lines[:provider_idx]) if provider_idx > 0 else ''
    provider_section = '\n'.join(lines[provider_idx:table_start_idx]) if provider_idx < table_start_idx else ''
    
    results = []
    config = DOCUMENT_TYPES[doc_type]
    
    for field_config in config['fields']:
        value = None
        confidence = 0.0
        
        field_name = field_config['name'].lower()
        
        if field_name == 'наименование получателя':
            search_text = provider_section
        elif field_name == 'инн':
            search_text = provider_section if provider_section.strip() else bank_section
        elif field_name in ['бик', 'наименование банка', 'счет']:
            search_text = bank_section
        else:
            search_text = text
        
        for pattern in field_config['patterns']:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                if match.groups():
                    value = match.group(1).strip()
                else:
                    value = match.group(0).strip()
                
                if value and len(value) > 1:
                    confidence = 0.7 + (0.3 * min(len(value) / 20, 1.0))
                    
                    if field_config['name'] in ['ИНН продавца', 'КПП продавца', 'ИНН исполнителя', 'ИНН', 'БИК', 'Счет']:
                        confidence = min(confidence + 0.1, 0.95)
                    
                    if field_config['name'] == 'Дата документа' and re.search(r'[а-я]{3,}', value, re.IGNORECASE):
                        confidence = min(confidence + 0.2, 0.98)
                    
                    if field_config['name'] == 'Наименование банка':
                        bad_patterns = ['реквизиты', 'получателя', 'счет', 'банк$']
                        if any(re.search(p, value.lower()) for p in bad_patterns):
                            continue
                        if not re.search(r'(банк|bank|точка|открытие|сбер)', value, re.IGNORECASE):
                            if not re.search(r'(пао|ооо|ао)\s+', value, re.IGNORECASE):
                                continue
                    
                    break
        
        results.append({
            'field': field_config['name'],
            'value': value or '',
            'confidence': confidence
        })
    
    return results


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