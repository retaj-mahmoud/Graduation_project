from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
import pdfplumber

problem_text = pdfplumber.open('GEO_Tool_Partner_Requirements.pdf')
for page in problem_text.pages:
    problem = page.extract_text()

research_text = pdfplumber.open('reseachPapers.pdf')
for paper in research_text.pages:
    research = paper.extract_text()

data = ([problem,research])

vectorizer = TfidfVectorizer()
vectorizerTfidf = vectorizer.fit_transform(data)

cosineSimilarity = cosine_similarity(vectorizerTfidf)

nlp = spacy.load('en_core_web_lg')

vector1 = nlp(problem)
vector2 = nlp(research)

textSimilarity = vector1.similarity(vector2)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

problemEmbedding = model.encode(problem)
researchEmbedding = model.encode(research)

cosineSimilarity_embedding = cosine_similarity([problemEmbedding], [researchEmbedding])[0][0]

print(f'lexical similarity = {cosineSimilarity_tfidf[0,1] * 100} %')
print(f'text similarity using spacy = {textSimilarity * 100} %')
print(f'semantic similarity = {cosineSimilarity_embedding * 100} %')



