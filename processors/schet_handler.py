"""
Handler for Payment Document (Счет) documents.
"""
from typing import Dict, List, Tuple, Optional, Any
import re
import logging
from processors.document_handler import BaseDocumentHandler

logger = logging.getLogger(__name__)

# Bank BIK to correspondent account prefix mapping
BANK_BIK_TO_CORR_ACC_PREFIX = {
    '044525411': '3010181014525',
    '044525593': '30101810400000000593',
    '044030653': '30101810300000000653',
}

VALID_BANK_INDICATORS = [
    'банк', 'bank', 'точка', 'открытие', 'сбер', 'альфа', 'тбанк', 'втб',
    'газпром', 'россельхоз', 'райффайзен', 'юникредит', 'бинбанк', 'промсвязь',
    'мосОбл', 'авангард', 'ситибанк', 'инвест', 'форбанк', 'русслав', 'вест',
    'уралсиб', 'росбанк', 'мкб', 'бкс', 'открытие', 'санкт-петербург',
]


def normalize_account_number(acc: str) -> str:
    """Remove all non-digit characters from account number."""
    return re.sub(r'\D', '', acc)


def validate_bank_accounts(bik: str, corr_account: str, rec_texts: list) -> Tuple[str, float]:
    """Validate bank account numbers and return corrected value with confidence."""
    if not corr_account:
        return corr_account, 0.0

    bik_clean = normalize_account_number(bik)
    corr_clean = normalize_account_number(corr_account)

    if len(corr_clean) < 19 or len(corr_clean) > 20:
        return corr_account, 0.0

    if corr_clean.startswith('301'):
        # This is likely a correspondent account, which is fine if found in a general search,
        # but we should flag it if it's supposed to be a settlement account.
        return corr_account, 0.8

    if len(bik_clean) == 9 and bik_clean.startswith('04'):
        correct_prefix = BANK_BIK_TO_CORR_ACC_PREFIX.get(bik_clean)
        if correct_prefix and not corr_clean.startswith('4'):
            logger.warning(f"Обнаружен счёт не начинающийся с 4. БИК={bik_clean}")
            return f"[ПРОВЕРЬТЕ] {corr_account}", 0.3

    return corr_account, 0.95


def is_warning_text(text: str) -> bool:
    """Check if text contains warning phrases."""
    warning_phrases = [
        r'внимание!?.?', r'оплата данного', r'означает согласие',
        r'в противном случае', r'не гарантируется', r'уведомление',
        r'обязательно', r'на складе', r'поставки\s+товара',
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in warning_phrases)


def is_valid_bank_name(text: str) -> bool:
    """Check if text contains valid bank name indicators."""
    text_lower = text.lower()
    # Strip HTML tags for validation
    text_lower = re.sub(r'<[^>]+>', '', text_lower)
    
    if is_warning_text(text_lower):
        return False

    if re.search(r'(реквизиты|получател[а-я]*|счет\s*$|bank$|^вним)', text_lower):
        return False

    has_bank_word = any(word in text_lower for word in VALID_BANK_INDICATORS)

    if re.search(r'(пао|ооо|ао|зао)\s+["\']?\s*\w', text_lower):
        return True

    return has_bank_word


def clean_bank_name(raw_text: str, rec_texts: list) -> Tuple[str, float]:
    """Clean and validate bank name from raw OCR text."""
    if not raw_text or len(raw_text) < 3:
        return '', 0.0

    # Strip HTML tags
    raw_text = re.sub(r'<[^>]+>', '', raw_text).strip()

    if is_warning_text(raw_text):
        for text in rec_texts:
            if is_valid_bank_name(text):
                return text, 0.8

        correct_bik = None
        for bik, prefix in BANK_BIK_TO_CORR_ACC_PREFIX.items():
            for rt in rec_texts:
                if bik in rt or prefix in rt:
                    correct_bik = bik
                    break
            if correct_bik:
                break

        if correct_bik:
            for rt in rec_texts:
                if re.search(r'(?:банк\s*(?:получателя|\s*$)|(?:пао|ооо|ао)\s+[А-ЯЁа-яё]+(?:\s*(?:банк|банк\s+[А-Я]))?)', rt, re.IGNORECASE):
                    if is_valid_bank_name(rt):
                        return rt, 0.7
        return '', 0.0

    if is_valid_bank_name(raw_text):
        return raw_text, 0.85

    return '', 0.0


class SchetHandler(BaseDocumentHandler):
    """
    Handler for Счет (Payment Invoice) documents.

    Document type patterns:
    - счет на оплату
    - счёт на оплату
    - invoice

    Key fields:
    - Номер документа
    - Дата документа
    - Поставщик
    - ИНН
    - БИК
    - Наименование банка
    - Расчетный счет
    - Корр. счет
    - Основание
    - Итого сумма
    """

    DOCUMENT_TYPE = "Счет"
    DOCUMENT_TYPE_DISPLAY = "Счет на оплату"

    DETECTION_PATTERNS = [
        r'счет\s+(?:на\s+)?оплату(?!-фактура)',
        r'(?<!-)(?:счёт|счет)\s+на\s+оплату(?!-фактура)',
        r'(?<!-)(?:счёт|счет)\s*№\s*\d+',
        r'(?:invoice|bill)\s*(?:no\.?|number)?',
    ]

    REQUIRED_FIELDS = [
        {
            'name': 'Тип документа',
            'patterns': [
                r'((?:Счет|Sчeт)\s+на\s+оплату)',
                r'(?:^|\n)\s*(?:тип\s+)?документа\s*[:\-]?\s*(.+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Номер документа',
            'patterns': [
                r'(?:счет|счёт)(?:\s+на\s+оплату)?\s*(?:№|no\.?|#)\s*[:\-]?\s*(\S+)',
                r'(?:номер|no\.?)\s*(?:№|no\.?|#)?\s*[:\-]?\s*(\S+)'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Дата документа',
            'patterns': [
                r'(?:счет|счёт)(?:\s+на\s+оплату)?\s*(?:№|no\.?|#)\s*\S+\s+от\s+(\d{1,2}(?:\s+[а-яё]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
                r'от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})',
                r'(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})'
            ],
            'required': True,
            'region': 'header'
        },
        {
            'name': 'Поставщик',
            'patterns': [
                r'(?:поставщик|исполнитель|продавец)\s*[:\-]?\s*(?:["\']?)(ООО\s+"[^"]+"|АО\s+"[^"]+"|ПАО\s+"[^"]+"|ИП\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)',
                r'(?:поставщик|исполнитель|продавец)\s*[:\-]?\s*(?:ooo|ооо|ао|пао|ип|ит)?\s*["\']?([\w\s"-]+?)(?:["\']?\s*,|\s*$|\s*инн)',
                r'(?:ИП|ИТ)\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)',
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'ИНН',
            'patterns': [
                r'(?:инн|inn|inн|iнн|иhh|1nn)\s*[:\-]?\s*(\d{10,12})',
                r'\b(\d{10,12})\b',
            ],
            'required': True,
            'region': 'provider'
        },
        {
            'name': 'БИК',
            'patterns': [
                r'(?:бик|bik)[\s\n:<>/td]+(\d{9})',
                r'\b(\d{9})\b(?=\s*(?:кпп|инн|р/с|сч|$))',
                r'\b(\d{9})\b',
            ],
            'required': False,
            'region': 'bank'
        },
        {
            'name': 'Наименование банка',
            'patterns': [
                r'(?:банк\s+(?:получателя)?|банк получателя)[\s\n:<>/td]+([А-ЯЁ\w\s"-]+?[Б|b]анк[А-ЯЁ\w\s"-]*)',
                r'([А-ЯЁ][\w\s"-]{0,30}[Б|b]анк[\w\s"-]{0,30})',
                r'(?:[Б|b]анк\s+(?:ПАО\s+|ПAO\s+|ООО\s+|АО\s+))([^\n<]+)',
                r'(ПАО\s+"[^"]+"|ПAO\s+"[^"]+")',
                r'([А-ЯЁa-z]+(?:\s+[А-ЯЁa-z]+)?\s+\((?:ПАО.АО|ООО|ЗАО)\))',
            ],
            'required': False,
            'region': 'bank'
        },
        {
            'name': 'Расчетный счет',
            'patterns': [
                r'(?:р/с|расч[её]тный\s+сч[её]т|лицевой\s+сч[её]т|сч\.?\s*№)[\s\n:<>/td]+(4\d{18,19})',
                r'(?<!\d)(4\d{18,19})(?!\d)',
                r'\b(4\d{19})\b',
            ],
            'required': False,
            'region': 'bank'
        },
        {
            'name': 'Корр. счет',
            'patterns': [
                r'(?:корр[.,]?\s*сч[её]т|к/с|сч\.?\s*№)[\s\n:<>/td]+(301\d{16,17})',
                r'(?<!\d)(301\d{16,17})(?!\d)',
                r'\b(301\d{17})\b',
            ],
            'required': False,
            'region': 'bank'
        },
        {
            'name': 'Основание',
            'patterns': [
                r'(?:основание|договор|контракт|basis|по\s+договору)\s*[:\-]?\s*(.+)',
                r'(?:^|\n)(?:[а-яё\s]+)?(?:№|#)?\s*(\d{5,}\s+(?:от|OT)\s+\d{1,2}(?:\s+[а-яё]+\s+|\.|\/)\d{2,4}(?:\s*г\.)?)',
            ],
            'required': False,
            'region': 'header'
        },
        {
            'name': 'Итого',
            'patterns': [
                r'(?:^|\n)\s*(?:и\s*того|всего|итого|total|sum)\s*(?:к\s*оплате|по\s*счету)?\s*[:\-—–\s]+\s*([\d\s.,]+(?:[.,]\d{2})?)\s*(?:руб|₽|rur)?',
                r'(?:^|\n)\s*([\d\s.,]+(?:[.,]\d{2})?)\s*(?:руб|₽|rur)\s*$',
                r'сумма\s*[:\-—–\s]+\s*([\d\s,]+(?:[.,]\d{2})?)',
                r'(?:всего|итого)[^\d\n]*([\d\s.,]+(?:[.,]\d{2})?)',
            ],
            'required': True,
            'region': 'footer'
        },
    ]

    OPTIONAL_FIELDS = [
        {
            'name': 'Адрес поставщика',
            'patterns': [
                r'(?:адрес|юридический\s+адрес)\s*[:\-]?\s*([^\n]+)',
                r'инн\s+\d{10,12}\s*,\s*([А-ЯЁ0-9][^\n]+)',
            ],
            'required': False,
            'region': 'provider'
        },
    ]

    FIELD_REGIONS = {
        'header': ['Тип документа', 'Номер документа', 'Дата документа', 'Основание'],
        'provider': ['Поставщик', 'ИНН', 'Адрес поставщика'],
        'bank': ['БИК', 'Наименование банка', 'Расчетный счет', 'Корр. счет'],
        'footer': ['Итого'],
        'table': ['Наименование', 'Кол-во', 'Цена', 'Сумма', 'Единица'],
    }

    TABLE_EXTRACTION_ENABLED = True

    def get_prompt(self) -> str:
        return """Извлеките текст и таблицы из счёта на оплату.

Требования к извлечению:
1. Извлеките ВЕСЬ текст документа без изменений (русский, английский, цифры)
2. Таблицы должны быть представлены в виде структурированных данных
3. Сохраните порядок строк и колонок в таблицах
4. Укажите номера строк в таблицах
5. ОБЯЗАТЕЛЬНО извлеките банковские реквизиты: БИК, наименование банка, расчетный счет, корреспондентский счет

Формат ответа:
- Текст: построчно
- Таблицы: списком списков, где каждый внутренний список - строка таблицы

ОтветЬТЕ ТОЛЬКО на русском языке, используя русские буквы и арабские цифры."""

    def get_post_process_prompt(self) -> str:
        return """Из счёта на оплату извлеките следующие поля:

1. Номер счёта
2. Дата выставления
3. Наименование поставщика (полное с ООО/АО/ПАО)
4. ИНН поставщика
5. БИК банка
6. Наименование банка
7. Расчетный счёт
8. Корреспондентский счёт
9. Основание (договор)
10. Итого сумма

ОтветЬТЕ ТОЛЬКО в формате JSON:
{
    "номер": "значение",
    "дата": "значение",
    "поставщик": "значение",
    "инн": "значение",
    "бик": "значение",
    "банк": "значение",
    "расчетный_счет": "значение",
    "корр_счет": "значение",
    "основание": "значение",
    "итого": "значение"
}

Если поле не найдено, укажите null.

Используйте ТОЛЬКО русские буквы и арабские цифры."""

    def extract_fields(self, text: str, regions: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract fields from payment document text."""
        results = []
        lines = text.split('\n')

        table_start_idx = self.detect_table_start(lines)
        header_start_idx = self._find_header_start(lines)

        # In Russian invoices, bank details are often at the top, above the title.
        if header_start_idx > 0:
            bank_section = '\n'.join(lines[:header_start_idx])
        else:
            bank_section = '\n'.join(lines[:table_start_idx])

        # Header contains document number and date, usually in the title line itself.
        if header_start_idx < len(lines):
            header_section = '\n'.join(lines[header_start_idx:table_start_idx])
        else:
            header_section = '\n'.join(lines[:table_start_idx])

        # Clean markdown noise from sections to avoid picking up headers as values
        def clean_md(t):
            t = re.sub(r'^\s*#+\s+.*$', '', t, flags=re.MULTILINE)
            t = re.sub(r'^\s*-{3,}\s*$', '', t, flags=re.MULTILINE)
            return t

        bank_section = clean_md(bank_section)
        header_section = clean_md(header_section)

        provider_section = regions.get('provider', '')
        rec_texts = regions.get('rec_texts', [])

        for field_config in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            value = None
            confidence = 0.0
            field_name = field_config['name']

            region = field_config.get('region', 'all')
            if region == 'header':
                search_texts = [header_section, text]
            elif region == 'provider':
                search_texts = [provider_section, header_section, text]
            elif region == 'bank':
                search_texts = [bank_section, text]
            elif region == 'footer':
                search_texts = ['\n'.join(lines[table_start_idx:]), text]
            else:
                search_texts = [text]

            found = False
            for search_text in search_texts:
                if not search_text:
                    continue
                for pattern in field_config['patterns']:
                    match = re.search(pattern, search_text, re.IGNORECASE)
                    if match:
                        if match.groups():
                            value = match.group(1).strip()
                        else:
                            value = match.group(0).strip()

                        if value and len(value) > 1:
                            # Strip HTML tags from value
                            value = re.sub(r'<[^>]+>', '', value).strip()
                            # Clean value before validation for better results
                            value = self.clean_field_value(field_name, value)
                            
                            is_valid, confidence = self.validate_field(field_name, value)
                            if is_valid:
                                # Special bank name validation
                                if field_name == 'Наименование банка':
                                    cleaned_name, conf = clean_bank_name(value, rec_texts)
                                    if cleaned_name:
                                        value = cleaned_name
                                        confidence = conf
                                    else:
                                        bad_patterns = ['реквизиты', 'получателя', 'счет', 'банк$']
                                        if any(re.search(p, value.lower()) for p in bad_patterns):
                                            continue
                                        if not re.search(r'(банк|bank|точка|открытие|сбер)', value, re.IGNORECASE):
                                            if not re.search(r'(пао|ооо|ао)\s+', value, re.IGNORECASE):
                                                continue
                                found = True
                                break
                        else:
                            value = None
                if found:
                    break

            results.append({
                'field': field_name,
                'value': value or '',
                'confidence': confidence,
                'required': field_config.get('required', False)
            })

        # Validate bank accounts and distinguish between Расчетный and Корр счет
        bik_val = ''
        for bf in results:
            if bf['field'] == 'БИК' and bf['value']:
                bik_val = bf['value']
                break

        for i, field in enumerate(results):
            if field['field'] in ['Расчетный счет', 'Корр. счет'] and field['value']:
                corrected_acc, conf = validate_bank_accounts(bik_val, field['value'], rec_texts)
                if corrected_acc != field['value']:
                    results[i]['value'] = corrected_acc
                    results[i]['confidence'] = conf

        return results

    def _find_header_start(self, lines: List[str]) -> int:
        """Find where the document header starts."""
        for i, line in enumerate(lines):
            if re.search(r'(?:счет\s+на\s+оплату| invoice|акт|упд|универсальный)', line, re.IGNORECASE):
                return i
        return 0
