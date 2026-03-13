import os
import re
import pickle
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader

app = Flask(__name__)
UPLOAD_FOLDER = 'categorized_resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def cleanResume(txt):
    cleanText = re.sub('http\\S+\\s', ' ', txt)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText.strip()

# Generate improvement suggestions
def generate_suggestions(text):
    suggestions = []

    # Check for missing sections
    if not re.search(r'skills', text, re.IGNORECASE):
        suggestions.append("Add a Skills section to highlight your abilities.")
    if not re.search(r'experience|work history', text, re.IGNORECASE):
        suggestions.append("Include Work Experience to showcase your professional background.")
    if not re.search(r'education', text, re.IGNORECASE):
        suggestions.append("Include your Education details for better ATS matching.")
    if len(text.split()) < 100:
        suggestions.append("Expand your resume with more details to improve ATS score.")

    # Example keyword suggestions
    required_keywords = ["Python", "Machine Learning", "Data Analysis"]
    missing_keywords = [kw for kw in required_keywords if kw.lower() not in text.lower()]
    if missing_keywords:
        suggestions.append(f"Include relevant keywords like {', '.join(missing_keywords)} to improve ATS score.")

    return suggestions

# Simple ATS score calculation
def calculate_ats_score(text):
    score = 50  # Base score
    if re.search(r'skills', text, re.IGNORECASE):
        score += 15
    if re.search(r'experience|work history', text, re.IGNORECASE):
        score += 15
    if re.search(r'education', text, re.IGNORECASE):
        score += 10
    if len(text.split()) > 300:
        score += 10
    return min(score, 100)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    uploaded_files = request.files.getlist('files[]')
    resumes = []

    for uploaded_file in uploaded_files:
        if uploaded_file.filename.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            cleaned_resume = cleanResume(text)

            # ML categorization
            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Save categorized resume
            category_folder = os.path.join(app.config['UPLOAD_FOLDER'], category_name)
            os.makedirs(category_folder, exist_ok=True)
            target_path = os.path.join(category_folder, uploaded_file.filename)
            uploaded_file.save(target_path)

            # Generate suggestions and ATS score
            suggestions = generate_suggestions(cleaned_resume)
            ats_score = calculate_ats_score(cleaned_resume)

            resumes.append({
                "filename": uploaded_file.filename,
                "category": category_name,
                "ats_score": ats_score,
                "suggestions": suggestions
            })

    # Overall ATS and job match (example: if any resume is Python Developer)
    overall_ats = sum([r["ats_score"] for r in resumes]) // len(resumes) if resumes else 0
    job_match = "Matched" if any(r["category"] == "Python Developer" for r in resumes) else "Not Matched"
    all_suggestions = sum([r["suggestions"] for r in resumes], [])

    return jsonify({
        "resumes": resumes,
        "ats_score": overall_ats,
        "job_match": job_match,
        "suggestions": all_suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)
