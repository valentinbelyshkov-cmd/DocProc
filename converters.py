import pandas as pd
from io import BytesIO
from docx import Document
from document_parser import parse_ocr_result


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


def to_extracted_data_xlsx(pages):
    parsed = parse_ocr_result(pages)
    
    doc_type = parsed.get('document_type', 'Неизвестный документ')
    type_confidence = parsed.get('type_confidence', 0)
    fields = parsed.get('fields', [])
    tables = parsed.get('tables', [])
    
    mem = BytesIO()
    with pd.ExcelWriter(mem, engine='openpyxl') as writer:
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        fields_df = pd.DataFrame(fields)
        if not fields_df.empty:
            fields_df.columns = ['Поле', 'Значение', 'Процент уверенности']
            fields_df['Процент уверенности'] = fields_df['Процент уверенности'].apply(
                lambda x: f"{x:.0%}" if x > 0 else ""
            )
            
            fields_df.to_excel(writer, index=False, sheet_name='Данные документа')
            ws1 = writer.sheets['Данные документа']
            
            ws1.cell(1, 1, f'Тип документа: {doc_type} (уверенность: {type_confidence:.0%})')
            ws1.merge_cells('A1:C1')
            ws1.cell(1, 1).font = Font(bold=True, size=12)
            
            for row in ws1.iter_rows(min_row=3, max_row=ws1.max_row, min_col=1, max_col=3):
                for cell in row:
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='left', vertical='center')
            
            for cell in ws1[3]:
                cell.font = header_font
                cell.fill = header_fill
        
        if tables:
            for idx, table_data in enumerate(tables[:3], 1):
                table_df = pd.DataFrame(table_data)
                if table_df.shape[1] <= 10:
                    table_df.to_excel(writer, index=False, sheet_name=f'Таблица {idx}')
                    ws = writer.sheets[f'Таблица {idx}']
                    
                    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
                        for cell in row:
                            cell.border = thin_border
                            cell.alignment = Alignment(horizontal='center', vertical='center')
                    
                    if ws.max_row > 0:
                        for cell in ws[1]:
                            cell.font = header_font
                            cell.fill = header_fill
        else:
            pd.DataFrame({'Сообщение': ['Таблицы в документе не найдены']}).to_excel(
                writer, index=False, sheet_name='Таблица 1'
            )
    
    mem.seek(0)
    return mem


def get_extracted_data(pages):
    return parse_ocr_result(pages)
