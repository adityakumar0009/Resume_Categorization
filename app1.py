import os
import re
import pickle
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'categorized_resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load TF-IDF vectorizer and ML model
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Category mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
}

# Sample keywords per category for ATS scoring (expand as needed)
category_keywords = {
    "Python Developer": ["python", "django", "flask", "sql", "oop", "api"],
    "Java Developer": ["java", "spring", "hibernate", "oop", "sql"],
    "Data Science": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "data analysis"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "aws", "ci/cd", "linux"],
    # Add more categories here...
}

# Function to clean resume text
def clean_resume(txt):
    txt = re.sub(r'http\S+|\#\S+|@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# Improved ATS score calculation
def calculate_ats_score(text, category_name):
    text_lower = text.lower()
    keywords = category_keywords.get(category_name, [])
    if not keywords:
        return 50  # default score if no keywords defined
    matched = sum(1 for kw in keywords if kw in text_lower)
    total = len(keywords)
    score = int((matched / total) * 100)
    # Penalize very short resumes
    word_count = len(text.split())
    if word_count < 150:
        score = int(score * 0.7)
    elif word_count < 300:
        score = int(score * 0.85)
    return min(score, 100)

# Function to give improvement suggestions
def improvement_suggestions(text):
    suggestions = []
    if len(text.split()) < 200:
        suggestions.append("Add more details about your experience and projects.")
    if "python" not in text.lower():
        suggestions.append("Include technical skills related to Python if applicable.")
    if "project" not in text.lower():
        suggestions.append("Highlight your projects to showcase practical experience.")
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_files = request.files.getlist('files[]')
    results = []

    ats_score_total = 0
    improvement_list = []

    for uploaded_file in uploaded_files:
        if uploaded_file.filename.endswith('.pdf'):
            # Extract text
            reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            cleaned_resume = clean_resume(text)

            # ML prediction
            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Save categorized resume
            category_folder = os.path.join(app.config['UPLOAD_FOLDER'], category_name)
            os.makedirs(category_folder, exist_ok=True)
            target_path = os.path.join(category_folder, uploaded_file.filename)
            uploaded_file.save(target_path)

            # ATS score
            ats_score = calculate_ats_score(cleaned_resume, category_name)
            ats_score_total += ats_score

            # Improvement suggestions
            improvement_list += improvement_suggestions(cleaned_resume)

            # Append results
            results.append({
                'filename': uploaded_file.filename,
                'category': category_name
            })

    # Average ATS score
    ats_score_avg = int(ats_score_total / len(uploaded_files)) if uploaded_files else 0

    # Job match (realistic logic)
    job_match = "Matched with predicted job role" if ats_score_avg >= 60 else "Not matched"

    return jsonify({
        "resumes": results,
        "ats_score": ats_score_avg,
        "job_match": job_match,
        "suggestions": improvement_list
    })

if __name__ == '__main__':
    app.run(debug=True)
