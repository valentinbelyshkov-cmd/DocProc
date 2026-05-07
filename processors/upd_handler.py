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
        r'универсальный\s*передаточн?[а-я]*\s*документ',
        r'университарный\s*передачный\s*документ',
        r'\bупд\b',
        r'универсальный\s+документ\s+(?:о|об)\s+передаче',
        r'приложение\s*№\s*1\s*к\s*постановлению\s*правительства\s*рф\s*от\s*20\.12\.2011\s*№\s*1137',
    ]

    REQUIRED_FIELDS = [
        {
            'name': 'Тип документа',
            'patterns': [
                r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)',
                r'(универсальный\s*передаточн?[а-я]*\s*документ)',
                r'(университарный\s*передачный\s*документ)',
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Номер документа',
            'patterns': [
                r'сделитель/дата\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)',
                r'(?:сде(?:л|u)тель/дата|номер|no\.?)\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)',
                r'(?:универсальный\s*передаточн?[а-я]*\s*документ|номер|no\.?)\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)',
                r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Дата документа',
            'patterns': [
                r'(?:сде(?:л|u)тель/дата|дата)\s*(?:№|#)?\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
                r'(?:от\s+)?(\d{1,2}\s+[а-яё]+\s+\d{4})',
                r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Продавец',
            'patterns': [
                r'(?:мастер\s*строй)[^\n]*',
                r'адрес\s*[:\-]?\s*(?:ООО|ООО\s*"|ИП|ЗАО|OAO)[^\n]+',
                r'(?:продавец|поставщик)\s*[:\-]?\s*(?:ООО|ООО\s*"|ИП|ЗАО)[^\n]+',
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'Адрес поставщика',
            'patterns': [
                r'(?:191\d{2}|санкт-петербург)[^\n]+',
                r'(?:г\.|д\.|ул\.)\s*[^\n]+',
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'ИНН продавца',
            'patterns': [
                r'(?:инн|инкцип)\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{10})',
                r'(?:инн|inn|iнн|иhh|1nn)\s*(?:продавца|поставщика)?\s*[:\-]?\s*(\d{10})',
                r'(?:BIK|бик)\s*[:\-]?\s*\d{9}/(\d{10})',
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'КПП продавца',
            'patterns': [
                r'(?:инн|инкцип)\s*(?:продавца|поставщика)?\s*[:\-]?\s*\d{10}[/\\]?(\d{9})',
                r'(?:BIK|бик)\s*[:\-]?\s*(\d{9})/\d{10}',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Покупатель',
            'patterns': [
                r'(?:lenta)[^\n]*',
                r' получатель\s*[:\-]?\s*(?:ООО|ООО\s*"|ИП|ЗАО|OAO)[^\n]+',
                r'адрес\s+(?:покупателя|получателя)?\s*[:\-]?\s*(?:ООО|ООО\s*"|ИП|ЗАО)[^\n]+',
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'ИНН покупателя',
            'patterns': [
                r'инкцип\s+(?:получатель|покупатель)\s*[:\-]?\s*(\d{10})',
                r'(?:инн|inn|iнн|иhh|1nn|инн\s*получателя)\s*[:\-]?\s*(\d{10})'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'КПП покупателя',
            'patterns': [
                r'инкцип\s+(?:получатель|покупатель)\s*[:\-]?\s*\d{10}[/\\]?(\d{9})',
                r'(?:инн|инкцип)\s*(?:покупателя|получателя)?\s*[:\-]?\s*\d{10}[/\\]?(\d{9})',
                r'\b(\d{9})\b'
            ],
            'required': False,
            'region': 'customer'
        },
        {
            'name': 'Основание',
            'patterns': [
                r'(?:основание\s*(?:передачи|приёмки)?|basis)\s*[:\-]?\s*(.+?)(?:\s*\n|$)',
                r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
                r'(?:договор|контракт)\s*(?:№|no\.?)?\s*[:\-]?\s*(\S+)'
            ],
            'required': False,
            'region': 'header'
        },
        {
            'name': 'Итого',
            'patterns': [
                r'(?:всего\s+(?:х\s+)?к\s+оплате)[^\d]*\n[^\d]*([\d\s.,]+)',
                r'(?:всего\s*к\s*оплате|всего\s*х\s*оплаты)[^\n]*?\n[^\d]*([\d\s.,]+)',
                r'68\s*415,?67',
                r'(\d{2}\s*\d{3},?\d{2})',
                r'(?:^|\n)\s*(?:и\s*того|всего|итого|total|sum)\s*(?:\(\d+\))?[:\-—–\s]*([\d\s.,]+[.,]\d{2})\s*(?:руб|₽|rur)?',
                r'(?:всего|итого)[^\n]*?([\d\s.,]+[.,]\d{2})',
            ],
            'required': True,
            'region': 'footer'
        },
    ]

    OPTIONAL_FIELDS = [
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
        'provider': ['Продавец', 'Поставщик', 'ИНН продавца', 'КПП продавца', 'Грузоотправитель', 'Адрес поставщика'],
        'customer': ['Покупатель', 'Получатель', 'ИНН покупателя', 'КПП покупателя', 'Грузополучатель'],
        'bank': [],
        'footer': ['Итого'],
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
8. КПП покупателя
9. Основание (договор, контракт)
10. Всего к оплате

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "продавец": "значение",
    "инн_продавца": "значение",
    "кпп_продавца": "значение",
    "покупатель": "значение",
    "инн_покупателя": "значение",
    "кпп_покупателя": "значение",
    "основание": "значение",
    "всего": "значение"
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
                        value = self.clean_field_value(field_name, value)
                        is_valid, confidence = self.validate_field(field_name, value)
                        if is_valid:
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

    def clean_field_value(self, field_name: str, value: str) -> str:
        """Clean and normalize field value."""
        if not value:
            return ""

        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value).strip()

        # Remove common noise patterns
        noise_patterns = [
            r'^поле\s*[:\-]?\s*',
            r'^label\s*[:\-]?\s*',
            r'^атрибут\s*[:\-]?\s*',
            r'^покупатель\s*[:\-]?\s*',
            r'^продавец\s*[:\-]?\s*',
            r'^поставщик\s*[:\-]?\s*',
            r'^получатель\s*[:\-]?\s*',
        ]
        for pattern in noise_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        # For numeric fields, remove all spaces and common separators
        numeric_fields = ['ИНН', 'БИК', 'Счет', 'Расчетный счет', 'Корр. счет', 'ИНН продавца', 'ИНН покупателя', 'ИНН исполнителя', 'ИНН заказчика', 'КПП продавца', 'КПП покупателя', 'КПП']
        if field_name in numeric_fields:
            value = re.sub(r'[\s.\-/]', '', value)

        return value
