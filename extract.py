import zipfile

zip_file = "furits.zip"
extract_to = "furits"

print("Extracting dataset...")

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Extraction completed!")