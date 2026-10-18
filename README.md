# AI Shorts Clipper

A local desktop-friendly Flask app that turns a long video + SRT subtitle file into **3–4 exactly 60-second vertical YouTube Shorts**, with AI-selected moments and burned-in captions.

The AI reads the **transcript only** and can pick moments anywhere in the video. FFmpeg handles cutting, center-cropping to 9:16, scaling to 1080×1920, and subtitle burning.

---

## 1. Requirements

- **Python 3.10+**
- **FFmpeg + FFprobe** on your PATH
- A **Groq** or **Gemini** API key

---

## 2. Windows PowerShell setup

### 2.1 Install Python dependencies

```powershell
cd path\to\your\project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt


run command 
cd "D:\vs codes\clipper"
& "C:\Users\m4 tech\AppData\Local\Programs\Python\Python310\python.exe" app.py
use is upper one not works
