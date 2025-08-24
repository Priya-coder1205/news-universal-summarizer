````md
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# 📰 Universal News Summarizer

An intelligent **news extractor + summarizer** that supports multiple sources, PDFs, and text files.  
Built with **Streamlit, NLP, and AI APIs** (OpenAI + Google Generative AI) for fast, reliable, and bias-aware news insights.  

---

## ✨ Features

- 🔗 Extract content from **multiple URLs** at once  
- 📄 Upload **PDF / TXT files** for summarization  
- 🧠 **AI-powered summarization** (with offline fallback using `newspaper3k` + `TextBlob`)  
- 📝 **Keyword extraction** for quick insights  
- 📊 **Sentiment & bias detection** using TextBlob + VADER  
- ⚡ Clean, interactive **Streamlit UI**  
- 🌐 Ready to deploy on **Streamlit Cloud**  

---

## 🚀 Demo

👉 [**Try it Live on Streamlit Cloud**](https://universal-news-summarizer.streamlit.app)  

---

## 📷 Screenshots

*(Optional – add screenshots of your app here. For example: homepage, sentiment analysis table, summary output, etc.)*  

---

## 🛠️ Installation & Setup

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/universal-news-summarizer.git
cd universal-news-summarizer
pip install -r requirements.txt
````

Run the app locally:

```bash
streamlit run app_streamlit.py
```

---

## ⚙️ Configuration

Create a `.env` file in the root folder with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_generative_ai_key
```

---

## 📖 Usage

* Paste one or more news article URLs
* Or upload a PDF / TXT file
* Select summarization length (short, medium, long)
* Click **Extract & Summarize**

View results:

* ✅ AI Summary
* ✅ Extracted Keywords
* ✅ Sentiment Analysis
* ✅ Bias Detection

---

## 🧰 Tech Stack

* Streamlit – UI framework
* newspaper3k – Article extraction
* TextBlob – Sentiment analysis
* NLTK + VADER – Bias/emotion detection
* OpenAI + Google Generative AI – Summarization
* Pandas, Scikit-learn – Data wrangling

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

```
```
