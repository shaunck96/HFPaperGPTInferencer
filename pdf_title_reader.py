from pdfrw import PdfReader

reader = PdfReader(r'C:\Users\307164\Desktop\Huggingface_Paper_Extractor\pdfs\2024-03-15\2403.09055')
print(reader.Info.Title)
