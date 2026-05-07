"""
Handler for Act (Акт) documents.
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from processors.document_handler import BaseDocumentHandler


class ActHandler(BaseDocumentHandler):
    """
    Handler for Акт (Act of Work Completion) documents.

    Document type patterns:
    - акт выполненных работ
    - акт оказанных услуг
    - акт сдачи-приёмки

    Key fields:
    - Номер документа
    - Дата документа
    - Исполнитель
    - ИНН исполнителя
    - Основание
    - Итого сумма
    """

    DOCUMENT_TYPE = "Акт"
    DOCUMENT_TYPE_DISPLAY = "Акт выполненных работ / оказанных услуг"

    DETECTION_PATTERNS = [
        r'\bакт\b.*(?:выполнен(?:ны|ых?)|оказанн(?:ая|ых?)|работ)',
        r'(?:акт|act)\s*(?:сдачи|при[её]мки|выполненн(?:ых?|ой))',
        r'акт\s+(?:сдачи|приёмки|выполненн)',
        r'акт\s+№',
    ]

    REQUIRED_FIELDS = [
        {
            'name': 'Тип документа',
            'patterns': [
                r'(Акт\s+(?:прие[е]мки-сдачи|выполненных\s+работ|оказанных\s+услуг)?)',
                r'(?:^|\n)\s*(?:тип\s+)?документа\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Номер документа',
            'patterns': [
                r'(?:акт|act)\s*(?:№|no\.?|#)\s*[:\-]?\s*(\S+)',
                r'(?:номер|no\.?)\s*[:\-]?\s*(\S+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Дата документа',
            'patterns': [
                r'(?:акт|act)\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-я]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
                r'от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})',
                r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Исполнитель',
            'patterns': [
                r'исполнитель\s*[:\-]?\s*(.+)',
                r'(?:исп|executor)\s*[:\-]?\s*(.+)',
                r'исполнителю?\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'ИНН исполнителя',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:исполнителя)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Заказчик',
            'patterns': [
                r'заказчик\s*[:\-]?\s*(.+)',
                r'(?:заказ|customer)\s*[:\-]?\s*(.+)',
                r'получатель\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'ИНН заказчика',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*(?:заказчика)?\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b'
            ],
            'required': False,
            'region': 'customer'
        },
        {
            'name': 'Основание',
            'patterns': [
                r'(?:основание|basis)\s*[:\-]?\s*(.+)',
                r'основание\s*[:\-]?\s*(.+)',
                r'(?:договор|контракт)\s*(?:№|no\.?)?\s*[:\-]?\s*(\S+)',
                r'договор\s+(?:№|no\.?)?\s*[:\-]?\s*(\d+\s+от\s+\d{1,2}[.,]\d{1,2}[.,]\d{2,4})',
                r'(?:по\s+)?договору\s+(?:№|no\.?)?\s*[:\-]?\s*(\d+)\s+от\s+(\d{1,2}\s+[а-я]+\s+\d{4})',
                r'(?:по\s+)?договору\s+(?:№|no\.?)?\s*[:\-]?\s*(\d+\s+(?:от\s+)?\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'
            ],
            'required': False,
            'region': 'all'
        },
        {
            'name': 'Итого',
            'patterns': [
                r'(?:^|\n)\s*(?:и\s*того|всего|итого|total|sum)\s*(?:к\s*оплате|по\s*акту)?\s*[:\-—–\s]+\s*([\d\s.,]+[.,]\d{2})\s*(?:руб|₽|rur)?',
                r'(?:^|\n)\s*([\d\s.,]+[.,]\d{2})\s*(?:руб|₽|rur)\s*$',
                r'сумма\s*[:\-—–\s]+\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'(?:всего|итого)[^\d\n]*([\d\s]+[.,]\d{2})',
            ],
            'required': True,
            'region': 'footer'
        },
    ]

    OPTIONAL_FIELDS = [
        {
            'name': 'Место составления',
            'patterns': [
                r'место\s*(?:составления)?\s*[:\-]?\s*([^:\n]{2,50}?)',
                r'(?:г\.|город)\s*[:\-]?\s*([^\n,]{2,50})'
            ],
            'required': False,
            'region': 'header'
        },
    ]

    FIELD_REGIONS = {
        'header': ['Тип документа', 'Номер документа', 'Дата документа', 'Основание', 'Место составления'],
        'provider': ['Исполнитель', 'ИНН исполнителя'],
        'customer': ['Заказчик', 'ИНН заказчика'],
        'bank': [],
        'footer': ['Итого'],
        'table': ['Наименование', 'Кол-во', 'Цена', 'Сумма', 'Единица'],
    }

    def get_prompt(self) -> str:
        return """Извлеките текст и таблицы из акта выполненных работ / оказанных услуг.

Требования к извлечению:
1. Извлеките ВЕСЬ текст документа без изменений (русский, английский, цифры)
2. Таблицы должны быть представлены в виде структурированных данных
3. Сохраните порядок строк и колонок в таблицах (список услуг/работ с ценами)
4. Укажите номера строк в таблицах

Формат ответа:
- Текст: построчно
- Таблицы: списком списков, где каждый внутренний список - строка таблицы

ОтветЬТЕ ТОЛЬКО на русском языке, используя русские буквы и арабские цифры."""

    def get_post_process_prompt(self) -> str:
        return """Из акта выполненных работ извлеките следующие поля:

1. Номер акта
2. Дата составления
3. Наименование исполнителя (полное)
4. ИНН исполнителя
5. Наименование заказчика (полное)
6. ИНН заказчика
7. Основание (номер и дата договора)
8. Итого сумма

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "исполнитель": "значение",
    "инн_исполнителя": "значение",
    "заказчик": "значение",
    "инн_заказчика": "значение",
    "основание": "значение",
    "итого": "значение"
}

Если поле не найдено, укажите null.

Используйте ТОЛЬКО русские буквы и арабские цифры."""

    def extract_fields(self, text: str, regions: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract fields from Act text."""
        results = []
        lines = text.split('\n')

        table_start_idx = self.detect_table_start(lines)
        provider_text = regions.get('provider', '')

        for field_config in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            value = None
            confidence = 0.0
            field_name = field_config['name']

            region = field_config.get('region', 'all')
            if region == 'header':
                search_text = '\n'.join(lines[:table_start_idx])
            elif region == 'provider':
                search_text = provider_text
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
                        # Если несколько групп (например, номер и дата договора) - объединяем
                        groups = [g.strip() for g in match.groups() if g and g.strip()]
                        if len(groups) > 1:
                            value = ' '.join(groups)
                        else:
                            value = groups[0] if groups else match.group(1).strip()
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