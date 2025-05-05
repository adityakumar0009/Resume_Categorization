# 🧠 Resume Categorization using Machine Learning and NLP

This project automates the classification of resumes into job-specific roles like **Data Scientist** or **Python Developer** using Natural Language Processing (NLP) and Machine Learning.

## 🚀 Objective
Automatically categorize resumes based on their content to streamline HR and recruitment processes.

## 🛠️ Tools & Technologies
- **Languages:** Python
- **Libraries:** Pandas, Scikit-learn, NLTK/spaCy, Joblib
- **NLP Techniques:** TF-IDF, Tokenization, Stopword Removal, Lemmatization
- **Models:** SVM, Random Forest
- **Deployment:** Flask (or similar), Render

### 📁 Project Structure
Resume_Categorization_Project/
│
├── Resume.csv # Labeled dataset of resumes
├── Resume_Categorization.ipynb # Notebook with full workflow
├── app.py # Web app script
├── model.pkl # Trained ML model
├── tfidf.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Python dependencies
└── categorized_resumes/ # Sample categorized resumes

## 🧪 Key Features
- Preprocesses resume text using NLP techniques.
- Trains and evaluates classification models (SVM, Random Forest).
- Predicts job roles from unseen resumes with high accuracy.
- Deployable as a web application.

## 📊 Results
Achieved high classification accuracy through experimentation with various ML algorithms and text processing techniques.

## 🌐 Live Demo
🔗 [Try the Web App](https://resume-categorization-4.onrender.com/)

## 📄 How to Run
1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run the app: `python app.py`

---
