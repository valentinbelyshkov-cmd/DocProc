"""
Table extraction utilities.
Provides configurable table detection and extraction from OCR text.
"""
from typing import List, Dict, Tuple, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Configurable table extraction from text documents.
    Supports multiple table formats and provides filtering options.
    """

    def __init__(
        self,
        min_rows: int = 2,
        min_columns: int = 2,
        delimiter_patterns: List[str] = None,
        header_patterns: List[str] = None
    ):
        """
        Initialize table extractor.

        Args:
            min_rows: Minimum number of rows to consider as table
            min_columns: Minimum number of columns
            delimiter_patterns: Patterns for splitting columns
            header_patterns: Patterns indicating table header
        """
        self.min_rows = min_rows
        self.min_columns = min_columns

        # Default column delimiters
        self.delimiter_patterns = delimiter_patterns or [
            r'\t',
            r'\s{2,}',  # 2+ spaces
            r'\s*\|\s*',  # pipe-separated
            r'\s*;\s*',  # semicolon-separated
        ]

        # Table header patterns
        self.header_patterns = header_patterns or [
            r'^\s*№?\s*(?:наименование|товар|описание|ед\.?|кол-во|количество|сумма|цена)',
            r'^\s*\d+\s+\d+\s+\d+\s*$',
            r'^\s*(?:номер|№)\s*(?:наименование|товар)',
        ]

        # Patterns for numeric data validation
        self.numeric_pattern = r'[\d\s,]+(?:\.\d+)?'
        self.currency_patterns = [
            r'\d+\s*(?:руб|₽|rur|eur|usd|\$)',
            r'\d+[.,]\d{2}\s*(?:руб|₽)?',
        ]

    def detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect all tables in text.

        Returns:
            List of detected table metadata
        """
        lines = text.split('\n')
        tables = []
        current_table = []
        table_started = False
        table_start_line = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # End current table
                if current_table and len(current_table) >= self.min_rows:
                    tables.append({
                        'start_line': table_start_line,
                        'end_line': i,
                        'rows': current_table,
                        'num_rows': len(current_table),
                        'num_columns': self._count_columns(current_table)
                    })
                current_table = []
                table_started = False
                continue

            # Check if this is a table header line
            is_header = any(
                re.match(p, line.lower())
                for p in self.header_patterns
            )

            # Check if this line looks like table data
            is_data_row = self._is_table_row(line)

            if is_header and not table_started:
                table_started = True
                table_start_line = i
                current_table = [line]
            elif is_data_row and table_started:
                current_table.append(line)
            elif is_data_row and not table_started and current_table:
                # Data row without explicit header
                if len(current_table) >= 1:
                    table_started = True
                    current_table.append(line)
                else:
                    current_table.append(line)
            elif not is_data_row and table_started:
                # End of table
                if len(current_table) >= self.min_rows:
                    tables.append({
                        'start_line': table_start_line,
                        'end_line': i,
                        'rows': current_table,
                        'num_rows': len(current_table),
                        'num_columns': self._count_columns(current_table)
                    })
                current_table = []
                table_started = False

        # Check for final table
        if current_table and len(current_table) >= self.min_rows:
            tables.append({
                'start_line': table_start_line,
                'end_line': len(lines),
                'rows': current_table,
                'num_rows': len(current_table),
                'num_columns': self._count_columns(current_table)
            })

        return tables

    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Extract words and numbers
        words = re.findall(r'\S+', line)
        numbers = re.findall(self.numeric_pattern, line)

        # Must have at least 2 elements
        if len(words) < 2:
            return False

        # Must have some numbers
        has_numbers = any(n.strip() for n in numbers)
        if not has_numbers:
            return False

        # Must have some text content
        has_text = any(
            re.search(r'[а-яА-Яa-zA-Z]', w)
            for w in words if not re.match(r'[\d\s,.]+', w)
        )

        return has_numbers or has_text

    def _count_columns(self, rows: List[str]) -> int:
        """Count number of columns based on delimiter patterns."""
        if not rows:
            return 0

        max_columns = 0
        for row in rows:
            columns = self._parse_columns(row)
            max_columns = max(max_columns, len(columns))

        return max_columns

    def _parse_columns(self, line: str) -> List[str]:
        """Parse a line into columns using delimiter patterns."""
        cells = [line]

        for pattern in self.delimiter_patterns:
            new_cells = []
            for cell in cells:
                parts = re.split(pattern, cell)
                new_cells.extend(parts)

            if len(new_cells) > len(cells):
                cells = new_cells
                break

        # Clean cells
        cells = [c.strip() for c in cells if c.strip()]
        return cells

    def extract_tables(self, text: str) -> List[List[List[str]]]:
        """
        Extract tables as list of lists.

        Returns:
            List of tables, each table is a list of rows (list of strings)
        """
        tables = self.detect_tables(text)
        result = []

        for table in tables:
            if table['num_columns'] >= self.min_columns:
                parsed_rows = [
                    self._parse_columns(row)
                    for row in table['rows']
                ]
                result.append(parsed_rows)

        return result

    def extract_numerical_tables(self, text: str) -> List[List[List[str]]]:
        """
        Extract tables that contain primarily numerical data.
        Useful for invoice line items.
        """
        tables = self.detect_tables(text)
        result = []

        for table in tables:
            # Check if table has enough columns
            if table['num_columns'] < self.min_columns:
                continue

            parsed_rows = []
            for row in table['rows']:
                cells = self._parse_columns(row)

                # Check if this row contains numbers
                has_numbers = any(
                    re.search(self.numeric_pattern, cell)
                    for cell in cells
                )

                if has_numbers:
                    parsed_rows.append(cells)

            if len(parsed_rows) >= self.min_rows:
                result.append(parsed_rows)

        return result

    def filter_tables(
        self,
        tables: List[List[List[str]]],
        has_header: bool = True,
        min_total_value: float = 0,
        column_filters: Optional[Dict[int, List[str]]] = None
    ) -> List[List[List[str]]]:
        """
        Filter extracted tables based on criteria.

        Args:
            tables: List of tables
            has_header: Only include tables with header row
            min_total_value: Minimum sum of numerical columns
            column_filters: Dict mapping column index to allowed values
        """
        filtered = []

        for table in tables:
            # Skip if header required but missing
            if has_header and len(table) < 2:
                continue

            # Apply column filters
            if column_filters:
                skip_table = False
                for col_idx, allowed_values in column_filters.items():
                    if col_idx < len(table):
                        col_values = [str(row[col_idx]) for row in table if col_idx < len(row)]
                        if not any(v in allowed_values for v in col_values):
                            skip_table = True
                            break
                if skip_table:
                    continue

            filtered.append(table)

        return filtered

    def tables_to_markdown(self, tables: List[List[List[str]]]) -> str:
        """Convert tables to markdown format."""
        markdown = ""

        for i, table in enumerate(tables, 1):
            markdown += f"### Таблица {i}\n\n"

            for row in table:
                markdown += "| " + " | ".join([str(cell) for cell in row]) + " |\n"

            markdown += "\n"

        return markdown


# Default extractor instance
default_extractor = TableExtractor()