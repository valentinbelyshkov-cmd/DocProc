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
                r'(?:тип\s+)?документа\s*[:\-]?\s*(.+)'
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
                r'инн\s*(?:исполнителя)?\s*[:\-]?\s*(\d{10,12})'
            ],
            'required': False,
            'region': 'provider'
        },
        {
            'name': 'Заказчик',
            'patterns': [
                r'заказчик\s*[:\-]?\s*(.+)',
                r'(?:заказ|customer)\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'customer'
        },
        {
            'name': 'ИНН заказчика',
            'patterns': [
                r'инн\s*(?:заказчика)?\s*[:\-]?\s*(\d{10,12})'
            ],
            'required': False,
            'region': 'customer'
        },
        {
            'name': 'Основание',
            'patterns': [
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
                r'итого\s*(?:к\s*оплате)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'(?:всего|итого)\s*(?:к\s*оплате|по\s*акту)?\s*[:\-]?\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'([\d\s,]+(?:[.,]\d{2})?)\s*(?:руб|₽|rur)'
            ],
            'required': True,
            'region': 'footer'
        },
    ]

    OPTIONAL_FIELDS = [
        {
            'name': 'Место составления',
            'patterns': [
                r'место\s*(?:составления)?\s*[:\-]?\s*(.+)',
                r'(?:г\.|город)\s*[:\-]?\s*([^\n,]+)'
            ],
            'required': False,
            'region': 'header'
        },
        {
            'name': 'КПП исполнителя',
            'patterns': [
                r'кпп\s*(?:исполнителя)?\s*[:\-]?\s*(\d{9})'
            ],
            'required': False,
            'region': 'provider'
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
5. КПП исполнителя
6. Наименование заказчика (полное)
7. ИНН заказчика
8. Основание (номер и дата договора)
9. Итого сумма

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "исполнитель": "значение",
    "инн_исполнителя": "значение",
    "кпп_исполнителя": "значение",
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
        is_ip = bool(re.search(r'\bИП\b', provider_text, re.IGNORECASE))

        for field_config in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            value = None
            confidence = 0.0
            field_name = field_config['name']

            # Для ИП не извлекаем КПП
            if field_name == 'КПП исполнителя' and is_ip:
                results.append({
                    'field': field_name,
                    'value': '',
                    'confidence': 0.0,
                    'required': field_config.get('required', False)
                })
                continue

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