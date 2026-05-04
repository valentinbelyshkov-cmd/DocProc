"""
Document template loader and manager.
Loads and manages document structure templates for different document types.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Base directory for templates
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'document_templates')


@dataclass
class FieldDefinition:
    """Definition of a document field."""
    name: str
    key: str
    field_type: str
    required: bool = False
    label: Optional[str] = None
    length: Optional[int] = None
    validation: Optional[str] = None
    format: Optional[str] = None
    pattern: Optional[str] = None
    default: Optional[str] = None
    prefix: Optional[str] = None
    width: Optional[int] = None
    align: Optional[str] = None


@dataclass
class SectionDefinition:
    """Definition of a document section (header, provider, bank, etc.)."""
    title: str
    fields: List[FieldDefinition]


@dataclass
class TableColumnDefinition:
    """Definition of a table column."""
    name: str
    key: str
    width: int = 20
    align: str = "left"


@dataclass
class TableDefinition:
    """Definition of a document table."""
    enabled: bool
    title: str
    columns: List[TableColumnDefinition]


@dataclass
class SignatureDefinition:
    """Definition of a signature block."""
    name: str
    key: str
    position: Optional[str] = None


@dataclass
class DocumentTemplate:
    """Complete document template."""
    document_type: str
    document_type_display: str
    description: str
    sections: Dict[str, SectionDefinition]
    table: Optional[TableDefinition]
    footer_fields: List[FieldDefinition]
    signatures: List[SignatureDefinition]
    styling: Dict[str, Any]
    is_tax_document: bool = False
    additional_info: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentTemplate':
        """Create template from dictionary."""
        # Parse sections
        sections = {}
        for section_key, section_data in data.get('fields', {}).items():
            fields = []
            for f in section_data.get('fields', []):
                fields.append(FieldDefinition(
                    name=f.get('name', ''),
                    key=f.get('key', ''),
                    field_type=f.get('type', 'string'),
                    required=f.get('required', False),
                    label=f.get('label'),
                    length=f.get('length'),
                    validation=f.get('validation'),
                    format=f.get('format'),
                    pattern=f.get('pattern'),
                    default=f.get('default'),
                    prefix=f.get('prefix'),
                    width=f.get('width'),
                    align=f.get('align')
                ))
            sections[section_key] = SectionDefinition(
                title=section_data.get('title', section_key),
                fields=fields
            )

        # Parse table
        table_def = None
        if 'table' in data:
            table_data = data['table']
            columns = []
            for c in table_data.get('columns', []):
                columns.append(TableColumnDefinition(
                    name=c.get('name', ''),
                    key=c.get('key', ''),
                    width=c.get('width', 20),
                    align=c.get('align', 'left')
                ))
            table_def = TableDefinition(
                enabled=table_data.get('enabled', True),
                title=table_data.get('title', 'Таблица'),
                columns=columns
            )

        # Parse footer fields
        footer_fields = []
        for f in data.get('footer', {}).get('fields', []):
            footer_fields.append(FieldDefinition(
                name=f.get('name', ''),
                key=f.get('key', ''),
                field_type=f.get('type', 'currency'),
                required=f.get('required', False)
            ))

        # Parse signatures
        signatures = []
        for sig in data.get('signatures', {}).get('signatories', []):
            signatures.append(SignatureDefinition(
                name=sig.get('name', ''),
                key=sig.get('key', ''),
                position=sig.get('position')
            ))

        return cls(
            document_type=data.get('document_type', ''),
            document_type_display=data.get('document_type_display', ''),
            description=data.get('description', ''),
            sections=sections,
            table=table_def,
            footer_fields=footer_fields,
            signatures=signatures,
            styling=data.get('styling', {}),
            is_tax_document=data.get('is_tax_document', False),
            additional_info=data.get('additional_info', {})
        )


class DocumentTemplateManager:
    """
    Manager for document templates.
    Loads and caches document templates from JSON files.
    """

    _templates: Dict[str, DocumentTemplate] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, template_dir: Optional[str] = None) -> None:
        """Load all templates from the template directory."""
        if cls._initialized:
            return

        template_dir = template_dir or TEMPLATE_DIR

        if not os.path.exists(template_dir):
            logger.warning(f"Template directory not found: {template_dir}")
            cls._initialized = True
            return

        for filename in os.listdir(template_dir):
            if filename.endswith('_template.json'):
                template_path = os.path.join(template_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    template = DocumentTemplate.from_dict(data)
                    cls._templates[template.document_type] = template
                    logger.info(f"Loaded template: {template.document_type}")

                except Exception as e:
                    logger.error(f"Failed to load template {filename}: {e}")

        cls._initialized = True
        logger.info(f"DocumentTemplateManager initialized with {len(cls._templates)} templates")

    @classmethod
    def get_template(cls, document_type: str) -> Optional[DocumentTemplate]:
        """Get template for a specific document type."""
        if not cls._initialized:
            cls.initialize()
        
        return cls._templates.get(document_type)

    @classmethod
    def get_all_templates(cls) -> Dict[str, DocumentTemplate]:
        """Get all loaded templates."""
        if not cls._initialized:
            cls.initialize()
        
        return cls._templates.copy()

    @classmethod
    def get_template_names(cls) -> List[str]:
        """Get list of available template names."""
        if not cls._initialized:
            cls.initialize()
        
        return list(cls._templates.keys())

    @classmethod
    def reload_templates(cls, template_dir: Optional[str] = None) -> None:
        """Reload all templates from disk."""
        cls._templates.clear()
        cls._initialized = False
        cls.initialize(template_dir)

    @classmethod
    def template_to_dict(cls, template: DocumentTemplate) -> Dict[str, Any]:
        """Convert template to dictionary format for serialization."""
        result = {
            'document_type': template.document_type,
            'document_type_display': template.document_type_display,
            'description': template.description,
            'fields': {},
            'table': None,
            'footer': {'fields': []},
            'signatures': {'signatories': []},
            'styling': template.styling,
            'is_tax_document': template.is_tax_document,
            'additional_info': template.additional_info or {}
        }

        # Convert sections
        for section_key, section in template.sections.items():
            result['fields'][section_key] = {
                'title': section.title,
                'fields': [
                    {
                        'name': f.name,
                        'key': f.key,
                        'type': f.field_type,
                        'required': f.required,
                        'label': f.label,
                        'length': f.length,
                        'validation': f.validation,
                        'format': f.format,
                        'pattern': f.pattern,
                        'default': f.default,
                        'prefix': f.prefix
                    }
                    for f in section.fields
                ]
            }

        # Convert table
        if template.table:
            result['table'] = {
                'enabled': template.table.enabled,
                'title': template.table.title,
                'columns': [
                    {
                        'name': c.name,
                        'key': c.key,
                        'width': c.width,
                        'align': c.align
                    }
                    for c in template.table.columns
                ]
            }

        # Convert footer fields
        for f in template.footer_fields:
            result['footer']['fields'].append({
                'name': f.name,
                'key': f.key,
                'type': f.field_type,
                'required': f.required
            })

        # Convert signatures
        for sig in template.signatures:
            result['signatures']['signatories'].append({
                'name': sig.name,
                'key': sig.key,
                'position': sig.position
            })

        return result


def load_all_templates() -> Dict[str, DocumentTemplate]:
    """Load all document templates."""
    DocumentTemplateManager.initialize()
    return DocumentTemplateManager.get_all_templates()


def get_template(document_type: str) -> Optional[DocumentTemplate]:
    """Get a specific document template."""
    return DocumentTemplateManager.get_template(document_type)


# Initialize on import
DocumentTemplateManager.initialize()