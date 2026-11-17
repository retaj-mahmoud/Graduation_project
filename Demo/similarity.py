from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
import pdfplumber

with pdfplumber.open('GEO_Tool_Partner_Requirements.pdf') as pdf:
    for page in pdf.pages:
        problem = page.extract_text()

with pdfplumber.open('reseachPapers.pdf') as pdf:
    for paper in pdf.pages:
        research = paper.extract_text()

data = ([problem,research])

vectorizer = TfidfVectorizer()
vectorizerTfidf = vectorizer.fit_transform(data)

cosineSimilarity = cosine_similarity(vectorizerTfidf)

nlp = spacy.load('en_core_web_lg')

vector1 = nlp(problem)
vector2 = nlp(research)

textSimilarity = vector1.similarity(vector2)

print(f'cosine similarity = {cosineSimilarity[0,1] * 100} %')
print(f'text similarity using spacy = {textSimilarity * 100} %')


