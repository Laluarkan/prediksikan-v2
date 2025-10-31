
import google.generativeai as genai

genai.configure(api_key="AIzaSyBzDYkSQ9oQiC9jvwW5_v5bKewBtCSx68o")

# Tampilkan model yang tersedia
for m in genai.list_models():
    print(m.name)
