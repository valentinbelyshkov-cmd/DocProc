from glmocr import GlmOcr, parse
import os

# Use an existing image from the repository for testing
# test_ocr.png is already present in the repository
image_path = "test_ocr.png"

print(f"Testing with image: {image_path}")

# --- Simple function API ---
# Note: a list is treated as pages of a single document.
try:
    result = parse(image_path)
    result.save(output_dir="./results")
    print("Simple parse successful.")
except Exception as e:
    print(f"Simple parse skipped or failed: {e}")

# --- Class-based API ---
try:
    with GlmOcr(layout_device="cpu") as parser:
        result = parser.parse(image_path)
        print("Standard OCR Result:", result.json_result)
        result.save()
except Exception as e:
    print(f"Class-based API skipped or failed: {e}")

# --- Document Parsing (Table Recognition) ---
print("\n--- Table Recognition ---")
try:
    with GlmOcr(layout_device="cpu") as parser:
        # Document Parsing tasks include: "Text Recognition:", "Formula Recognition:", "Table Recognition:"
        result = parser.parse(image_path, prompt="Table Recognition:")
        print("Table Recognition Result:", result.json_result)
except Exception as e:
    print(f"Table Recognition skipped or failed: {e}")

# --- Information Extraction ---
print("\n--- Information Extraction ---")
ie_prompt = """请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}"""

try:
    with GlmOcr(layout_device="cpu") as parser:
        # Information Extraction requires a strict JSON schema prompt
        result = parser.parse(image_path, prompt=ie_prompt)
        print("Information Extraction Result:", result.json_result)
except Exception as e:
    print(f"Information Extraction skipped or failed: {e}")

# --- Other examples from documentation ---
# result = parse(["img1.png", "img2.jpg"])
# result = parse("https://example.com/image.png")

# Place layout model on a specific GPU
# with GlmOcr(layout_device="cuda:1") as parser:
#     result = parser.parse(image_path)
