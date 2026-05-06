"""
Handler for UPD (Универсальный передаточный документ) documents.
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from processors.document_handler import BaseDocumentHandler


class UPDHandler(BaseDocumentHandler):
    """
    Handler for УПД (Universal Transfer Document) documents.

    Document type patterns:
    - универсальный передаточный документ
    - упд

    Key fields:
    - Номер документа
    - Дата документа
    - Продавец/Поставщик
    - Покупатель/Получатель
    - ИНН продавца/покупателя
    - Основание
    - Итого сумма
    """

    DOCUMENT_TYPE = "УПД"
    DOCUMENT_TYPE_DISPLAY = "Универсальный передаточный документ"

    DETECTION_PATTERNS = [
        r'универсальный\s*передаточн?\s*документ',
        r'\bупд\b',
        r'универсальный\s+документ\s+(?:о|об)\s+передаче',
    ]

    REQUIRED_FIELDS = [
        {
            'name': 'Тип документа',
            'patterns': [
                r'(универсальный\s*передаточн?\s*документ)',
                r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Номер документа',
            'patterns': [
                r'(?:универсальный\s*передаточн?\s*документ|номер|no\.?)\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)',
                r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Дата документа',
            'patterns': [
                r'(?:№|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
                r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
                r'(?:от\s+)?(\d{1,2}\s+[а-яё]+\s+\d{4})'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Продавец',
            'patterns': [
                r'продавец\s*[:\-]?\s*(.+)',
                r'поставщик\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'ИНН продавца',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'КПП продавца',
            'patterns': [
                r'(?:кпп|kpp|kпп|кpp)\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{9})',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Покупатель',
            'patterns': [
                r'покупатель\s*[:\-]?\s*(.+)',
                r'получатель\s*[:\-]?\s*(.+)',
                r'(?:buyer|customer)\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'ИНН покупателя',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:покупателя|получателя)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'Основание',
            'patterns': [
                r'(?:основание|basis)\s*[:\-]?\s*(.+)',
                r'(\d{5,}\s+(?:от|OT)\s+\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
                r'(?:договор|контракт)\s*(?:№|no\.?)?\s*[:\-]?\s*(\S+)'
            ],
            'required': False,
            'region': 'header'
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
                r'(?:кпп|kpp|kпп|кpp)\s*(?:покупателя|получателя)?\s*[:\-]?\s*(\d{9})',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'customer'
        },
        {
            'name': 'Грузоотправитель',
            'patterns': [
                r'грузоотправитель\s*[:\-]?\s*(.+)'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Грузополучатель',
            'patterns': [
                r'грузополучатель\s*[:\-]?\s*(.+)'
            ],
            'required': False,
            'region': 'customer'
        },
    ]

    FIELD_REGIONS = {
        'header': ['Тип документа', 'Номер документа', 'Дата документа', 'Основание'],
        'provider': ['Продавец', 'Поставщик', 'ИНН продавца', 'КПП продавца', 'Грузоотправитель'],
        'customer': ['Покупатель', 'Получатель', 'ИНН покупателя', 'КПП покупателя', 'Грузополучатель'],
        'bank': [],
        'footer': ['Итого к оплате'],
        'table': ['Наименование', 'Кол-во', 'Сумма'],
    }

    def get_prompt(self) -> str:
        return """Извлеките текст и таблицы из универсального передаточного документа (УПД).

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
        return """Из документа УПД извлеките следующие поля:

1. Номер документа
2. Дата выставления
3. Наименование продавца/поставщика (полное)
4. ИНН продавца
5. КПП продавца
6. Наименование покупателя/получателя (полное)
7. ИНН покупателя
8. Основание (договор, контракт)
9. Итого сумма к оплате

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "продавец": "значение",
    "инн_продавца": "значение",
    "кпп_продавца": "значение",
    "покупатель": "значение",
    "инн_покупателя": "значение",
    "основание": "значение",
    "итого": "значение"
}

Если поле не найдено, укажите null.

Используйте ТОЛЬКО русские буквы и арабские цифры."""

    def extract_fields(self, text: str, regions: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract fields from UPD text."""
        results = []
        lines = text.split('\n')

        table_start_idx = self.detect_table_start(lines)

        for field_config in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            value = None
            confidence = 0.0
            field_name = field_config['name']

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
