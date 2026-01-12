from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Job Description
job_description = """
Looking for a Python developer with experience in Machine Learning, Data Analysis,
Flask, and SQL. Knowledge of NLP is a plus.
"""

# Sample Resumes
resumes = [
    "I am a Python developer with experience in Machine Learning and Data Analysis. Worked with Flask.",
    "I am a Java developer with experience in Spring and Hibernate.",
    "Data scientist skilled in Python, NLP, Machine Learning, and SQL.",
    "Web developer with HTML, CSS, JavaScript and React."
]

# Combine JD and resumes
documents = [job_description] + resumes

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute similarity between job description and resumes
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Display results
print("📄 Resume Ranking Based on Job Description:\n")

for i, score in enumerate(similarity_scores[0]):
    print(f"Resume {i+1} Match Score: {round(score*100, 2)} %")

# Find best resume
best_resume = similarity_scores[0].argmax()
print(f"\n✅ Best matching resume is: Resume {best_resume + 1}")