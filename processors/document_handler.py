"""
Base class for document type processors.
Each document type (invoice, act, UPD, etc.) should have its own handler.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import re


class BaseDocumentHandler(ABC):
    """
    Abstract base class for document type handlers.
    Each specific document type (Счет, Акт, Счет-фактура, УПД) should inherit from this class.
    """

    # Document type name (should be overridden by subclasses)
    DOCUMENT_TYPE: str = "base"
    DOCUMENT_TYPE_DISPLAY: str = "Неизвестный документ"

    # Patterns for document type detection
    DETECTION_PATTERNS: List[str] = []

    # Required fields for the document
    REQUIRED_FIELDS: List[Dict[str, Any]] = []

    # Optional fields for the document
    OPTIONAL_FIELDS: List[Dict[str, Any]] = []

    # Search regions for fields (header, body, bank_section, etc.)
    FIELD_REGIONS: Dict[str, List[str]] = {}

    # Table extraction settings
    TABLE_MIN_ROWS: int = 2
    TABLE_MIN_COLUMNS: int = 2
    TABLE_EXTRACTION_ENABLED: bool = True

    @abstractmethod
    def get_prompt(self) -> str:
        """Return the OCR prompt for this document type."""
        pass

    @abstractmethod
    def get_post_process_prompt(self) -> str:
        """Return the post-processing prompt for field extraction."""
        pass

    @abstractmethod
    def extract_fields(self, text: str, regions: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract document fields from text."""
        pass

    def detect_document(self, text: str) -> Tuple[bool, float]:
        """
        Detect if the given text matches this document type.
        Returns (is_match, confidence).
        """
        text_lower = text.lower()
        matches = 0

        for pattern in self.DETECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1

        if matches == 0:
            return False, 0.0

        confidence = min(matches / len(self.DETECTION_PATTERNS), 1.0)
        return True, confidence

    def validate_field(self, field_name: str, value: str) -> Tuple[bool, float]:
        """
        Validate a field value.
        Returns (is_valid, confidence).
        """
        if not value:
            return False, 0.0

        confidence = 0.7 + (0.3 * min(len(value) / 20, 1.0))

        # Numeric validation for INN, BIK, account numbers
        numeric_fields = ['ИНН', 'БИК', 'Счет', 'Расчетный счет', 'Корр. счет', 'ИНН продавца', 'ИНН покупателя', 'ИНН исполнителя', 'ИНН заказчика']
        if field_name in numeric_fields:
            if re.match(r'^\d+$', value):
                confidence = min(confidence + 0.1, 0.95)
            else:
                return False, 0.0

        # Date validation
        if field_name == 'Дата документа':
            if re.search(r'[а-я]{3,}', value):
                confidence = min(confidence + 0.2, 0.98)

        return True, confidence

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
        ]
        for pattern in noise_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        return value

    def get_all_fields(self) -> List[Dict[str, Any]]:
        """Return all fields (required + optional) for this document type."""
        return self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS

    def get_required_field_names(self) -> List[str]:
        """Return names of required fields."""
        return [f['name'] for f in self.REQUIRED_FIELDS]

    def get_field_config(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific field."""
        for field in self.get_all_fields():
            if field['name'] == field_name:
                return field
        return None

    def get_field_region(self, field_name: str) -> Optional[str]:
        """Get the search region for a field."""
        for region, fields in self.FIELD_REGIONS.items():
            if field_name in fields:
                return region
        return None

    def is_table_field(self, field_name: str) -> bool:
        """Check if the field is typically found in tables."""
        table_keywords = ['наименование', 'количество', 'сумма', 'цена', 'ед']
        return any(kw in field_name.lower() for kw in table_keywords)

    def parse_table_line(self, line: str) -> List[str]:
        """Parse a table row into cells."""
        cells = re.split(r'[\t]{1,4}|[\s]{2,}', line)
        return [c.strip() for c in cells if c.strip()]

    def detect_table_start(self, lines: List[str]) -> int:
        """Detect where tables start in the document."""
        table_headers = [
            r'^\s*№?\s*(?:наименование|товар|описание|ед\.|кол-во|количество|сумма)',
            r'^\s*<t[dh]>.*?№.*?</t[dh]>',
            r'^\s*<t[dh]>.*?наименование.*?</t[dh]>',
            r'^\s*<t[dh]>.*?товар.*?</t[dh]>',
            r'^\s*\d+\s+\d+\s+\d+',
            r'^\s*(?:номер|№)\s*(?:наименование|товар)',
            r'^\s*<thead',
            r'^\s*<tr',
            r'^\s*\|',
        ]

        # First pass: try more specific patterns
        for i, line in enumerate(lines):
            for pattern in table_headers[:6]:
                if re.search(pattern, line.lower()):
                    return i

        # Second pass: more general patterns, but skip typical bank table headers
        bank_keywords = ['бик', 'сч. №', 'корр. счет', 'банк получателя']
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in bank_keywords) and i < 20: # Typically bank table is at the top
                continue
            for pattern in table_headers[6:]:
                if re.search(pattern, line_lower):
                    return i

        return len(lines)  # No table found, return end of document
