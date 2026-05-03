import pandas as pd
from io import BytesIO
from docx import Document

def to_docx(pages):
    doc = Document()
    for page in pages:
        doc.add_heading(f"Страница {page['page_num']}", level=1)
        markdown_text = page.get('markdown', '')
        for line in markdown_text.split('\n'):
            if line.strip():
                doc.add_paragraph(line.strip())
        doc.add_page_break()
    
    mem = BytesIO()
    doc.save(mem)
    mem.seek(0)
    return mem

def to_xlsx(pages):
    rows = []
    for page in pages:
        if 'result_json' in page and page['result_json']:
            for p_data in page['result_json']:
                for block in p_data.get('blocks', []):
                    rows.append({
                        'Страница': page['page_num'],
                        'Текст': block.get('text', ''),
                        'Уверенность': block.get('confidence', 0)
                    })
        else:
            rows.append({
                'Страница': page['page_num'],
                'Текст': page['markdown'],
                'Уверенность': 1.0
            })
    
    df = pd.DataFrame(rows)
    mem = BytesIO()
    with pd.ExcelWriter(mem, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='OCR Result')
    mem.seek(0)
    return mem

def to_markdown(pages):
    full_markdown = ""
    for page in pages:
        full_markdown += f"# Страница {page['page_num']}\n\n{page['markdown']}\n\n---\n\n"
        
    mem = BytesIO()
    mem.write(full_markdown.encode('utf-8'))
    mem.seek(0)
    return mem
