
import google.generativeai as genai

genai.configure(api_key="AIzaSyAtqFJ73vCIXeIDpT3e_ImlmpNKlycrtio")

# Tampilkan model yang tersedia
for m in genai.list_models():
    print(m.name)
