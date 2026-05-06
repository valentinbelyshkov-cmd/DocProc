"""
Handler for Invoice (Счет-фактура) documents.
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from processors.document_handler import BaseDocumentHandler


class InvoiceHandler(BaseDocumentHandler):
    """
    Handler for Счет-фактура (Invoice) documents.

    Document type patterns:
    - счет-фактура
    - счет фактура
    - invoice

    Key fields:
    - Номер документа
    - Дата документа
    - Продавец
    - Покупатель
    - ИНН продавца/покупателя
    - Итого сумма
    """

    DOCUMENT_TYPE = "Счет-фактура"
    DOCUMENT_TYPE_DISPLAY = "Счет-фактура"

    DETECTION_PATTERNS = [
        r'счет[_\s]фактура',
        r'счет\s+фактура',
        r'счет-фактура',
        r'\bсчф?\b.*(?:фактура|универсальный)',
    ]

    REQUIRED_FIELDS = [
        {
            'name': 'Тип документа',
            'patterns': [r'((?:Счет|Sчeт)-фактура)', r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Номер документа',
            'patterns': [
                r'(?:счет-фактура|счф?|invoice)\s*(?:№|no\.?|number|#)\s*[:\-]?\s*(\S+)',
                r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Дата документа',
            'patterns': [
                r'(?:счет-фактура|счф?|invoice)\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
                r'(?:от\s*)?(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
                r'(?:дата|date)\s*[:\-]?\s*(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Продавец',
            'patterns': [
                r'продавец\s*[:\-]?\s*(.+)',
                r'(?:seller|supplier)\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'ИНН продавца',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:продавца)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'КПП продавца',
            'patterns': [
                r'(?:кпп|kpp|kпп|кpp)\s*(?:продавца)?\s*[:\-]?\s*(\d{9})',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Покупатель',
            'patterns': [
                r'покупатель\s*[:\-]?\s*(.+)',
                r'(?:buyer|customer)\s*[:\-]?\s*(.+)',
                r'получатель\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'ИНН покупателя',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:покупателя)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'Итого к оплате',
            'patterns': [
                r'(?:всего|итого|total|sum)\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'([\d\s,]+(?:[.,]\d{2})?)\s*(?:руб|₽|rur)',
                r'всего\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'сумма\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)'
            ],
            'required': True,
            'region': 'footer'
        },
    ]

    OPTIONAL_FIELDS = [
        {
            'name': 'КПП покупателя',
            'patterns': [
                r'(?:кпп|kpp|kпп|кpp)\s*(?:покупателя)?\s*[:\-]?\s*(\d{9})',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'customer'
        },
        {
            'name': 'Номер грузоотправителя',
            'patterns': [r'грузоотправитель\s*[:\-]?\s*(\d+)'],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Грузополучатель',
            'patterns': [r'грузополучатель\s*[:\-]?\s*(.+)'],
            'required': False,
            'region': 'customer'
        },
    ]

    FIELD_REGIONS = {
        'header': ['Тип документа', 'Номер документа', 'Дата документа'],
        'provider': ['Продавец', 'ИНН продавца', 'КПП продавца'],
        'customer': ['Покупатель', 'ИНН покупателя', 'КПП покупателя', 'Грузополучатель'],
        'bank': [],  # Usually not present in invoices
        'footer': ['Итого к оплате'],
        'table': ['Наименование товара', 'Количество', 'Цена', 'Сумма'],
    }

    def get_prompt(self) -> str:
        return """Извлеките текст и таблицы из счета-фактуры.

Требования к извлечению:
1. Извлеките ВЕСЬ текст документа без изменений (русский, английский, цифры)
2. Таблицы должны быть представлены в виде структурированных данных
3. Сохраните порядок строк и колонок в таблицах
4. Укажите номера строк в таблицах

Формат ответа:
- Текст: построчно
- Таблицы: списком списков, где каждый внутренний список - строка таблицы

ОтветЬТЕ ТОЛЬКО на русском языке, используя русские буквы и арабские цифры."""

    def get_post_process_prompt(self) -> str:
        return """Из документа извлеките следующие поля:

1. Номер счета-фактуры
2. Дата выставления
3. Наименование продавца (полное)
4. ИНН продавца
5. КПП продавца
6. Наименование покупателя (полное)
7. ИНН покупателя
8. Итого сумма к оплате

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "продавец": "значение",
    "инн_продавца": "значение",
    "кпп_продавца": "значение",
    "покупатель": "значение",
    "инн_покупателя": "значение",
    "итого": "значение"
}

Если поле не найдено, укажите null."""

    def extract_fields(self, text: str, regions: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract fields from invoice text."""
        results = []
        lines = text.split('\n')

        # Find table start
        table_start_idx = self.detect_table_start(lines)

        for field_config in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            value = None
            confidence = 0.0
            field_name = field_config['name']

            # Determine search region
            region = field_config.get('region', 'all')
            if region == 'header':
                search_text = '\n'.join(lines[:table_start_idx])
            elif region == 'provider':
                search_text = regions.get('provider', '')
            elif region == 'customer':
                search_text = regions.get('customer', '')
            elif region == 'bank':
                search_text = regions.get('bank', '')
            elif region == 'footer':
                search_text = '\n'.join(lines[table_start_idx:])
            else:
                search_text = text

            # Search for field value
            for pattern in field_config['patterns']:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    if match.groups():
                        value = match.group(1).strip()
                    else:
                        value = match.group(0).strip()

                    if value and len(value) > 1:
                        is_valid, confidence = self.validate_field(field_name, value)
                        if is_valid:
                            value = self.clean_field_value(field_name, value)
                            break
                    else:
                        value = None

            results.append({
                'field': field_name,
                'value': value or '',
                'confidence': confidence,
                'required': field_config.get('required', False)
            })

        return results
