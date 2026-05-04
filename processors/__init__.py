# Document processors module
from processors.base_processor import BaseTask, BaseProcessor

# Lazy imports for processors with heavy dependencies
# These will be imported when first accessed
__all__ = [
    'BaseTask',
    'BaseProcessor',
]