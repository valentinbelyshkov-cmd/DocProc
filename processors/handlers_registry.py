"""
Factory for document type handlers.
Provides a registry of all available document handlers.
"""
from typing import Dict, List, Type, Optional, Tuple
from processors.document_handler import BaseDocumentHandler
from processors.invoice_handler import InvoiceHandler
from processors.upd_handler import UPDHandler
from processors.act_handler import ActHandler
from processors.schet_handler import SchetHandler


class DocumentHandlerRegistry:
    """
    Registry for document type handlers.
    Manages registration and retrieval of handlers for different document types.
    """

    _handlers: Dict[str, Type[BaseDocumentHandler]] = {}

    @classmethod
    def register(cls, handler_class: Type[BaseDocumentHandler]) -> None:
        """Register a new document handler."""
        if not issubclass(handler_class, BaseDocumentHandler):
            raise TypeError(f"{handler_class} must inherit from BaseDocumentHandler")

        doc_type = handler_class.DOCUMENT_TYPE
        cls._handlers[doc_type] = handler_class

    @classmethod
    def get_handler(cls, doc_type: str) -> Optional[BaseDocumentHandler]:
        """Get an instance of handler for the specified document type."""
        handler_class = cls._handlers.get(doc_type)
        if handler_class:
            return handler_class()
        return None

    @classmethod
    def get_all_handlers(cls) -> Dict[str, BaseDocumentHandler]:
        """Get all registered handlers as instances."""
        return {
            doc_type: handler_class()
            for doc_type, handler_class in cls._handlers.items()
        }

    @classmethod
    def detect_document_type(cls, text: str) -> Tuple[Optional[str], float]:
        """
        Detect document type from text.
        Returns (document_type, confidence).
        """
        best_match = None
        best_confidence = 0.0

        for doc_type, handler_class in cls._handlers.items():
            handler = handler_class()
            is_match, confidence = handler.detect_document(text)

            if is_match and confidence > best_confidence:
                best_confidence = confidence
                best_match = doc_type

        return best_match, best_confidence

    @classmethod
    def get_document_types(cls) -> List[str]:
        """Get list of all registered document types."""
        return list(cls._handlers.keys())

    @classmethod
    def get_handler_info(cls, doc_type: str) -> Optional[Dict]:
        """Get information about a document type handler."""
        handler_class = cls._handlers.get(doc_type)
        if not handler_class:
            return None

        handler = handler_class()
        return {
            'document_type': doc_type,
            'display_name': handler.DOCUMENT_TYPE_DISPLAY,
            'required_fields': handler.get_required_field_names(),
            'table_extraction_enabled': handler.TABLE_EXTRACTION_ENABLED
        }


# Register all built-in handlers
DocumentHandlerRegistry.register(InvoiceHandler)
DocumentHandlerRegistry.register(UPDHandler)
DocumentHandlerRegistry.register(ActHandler)
DocumentHandlerRegistry.register(SchetHandler)