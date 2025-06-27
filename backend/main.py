from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import io
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store document content
current_document = ""
document_sentences = []
tfidf_vectorizer = None
sentence_vectors = None

class QuestionRequest(BaseModel):
    question: str

class AnswerRequest(BaseModel):
    question: str
    user_answer: str

class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_from_pdf(self, file_content):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                print(f"[DEBUG] Page Text: {page_text[:500]}")  # Print first 500 characters
                text += page_text + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

    def preprocess_text(self, text):
        # Clean and normalize text
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def generate_summary(self, text, max_words=150):
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text[:max_words*5]  # Rough character estimate
        
        # Simple extractive summarization using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.sum(sentence_vectors.toarray(), axis=1)
        
        # Get top sentences
        top_indices = np.argsort(sentence_scores)[-3:]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        summary = ' '.join(summary_sentences)
        
        # Truncate to word limit
        words = summary.split()
        if len(words) > max_words:
            summary = ' '.join(words[:max_words]) + "..."
        
        return summary
    
    def find_relevant_sentences(self, question, sentences, top_k=3):
        if not sentences:
            return []
        
        all_text = [question] + sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(all_text)
        
        question_vector = vectors[0]
        sentence_vectors = vectors[1:]
        
        similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_sentences = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_sentences.append({
                    'text': sentences[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return relevant_sentences
    
    def generate_answer(self, question, text):
        sentences = sent_tokenize(text)
        relevant_sentences = self.find_relevant_sentences(question, sentences)
        
        if not relevant_sentences:
            return "I couldn't find information relevant to your question in the document.", ""
        
        # Combine top relevant sentences for answer
        answer_parts = []
        justification_parts = []
        
        for i, sent_info in enumerate(relevant_sentences[:2]):  # Use top 2 sentences
            answer_parts.append(sent_info['text'])
            justification_parts.append(f"Reference {i+1}: \"{sent_info['text'][:100]}...\" (Sentence {sent_info['index']+1})")
        
        answer = ' '.join(answer_parts)
        justification = '\n'.join(justification_parts)
        
        return answer, justification
    
    def generate_challenges(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return []
        
        # Extract key information for questions
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Get important terms using frequency
        word_freq = Counter(words)
        important_terms = [word for word, freq in word_freq.most_common(20) if len(word) > 3]
        
        challenges = []
        
        # Generate different types of questions
        # Type 1: What/Who questions
        if important_terms:
            term = random.choice(important_terms)
            relevant_sents = [s for s in sentences if term in s.lower()]
            if relevant_sents:
                challenges.append({
                    'question': f"What is the significance of '{term}' in this document?",
                    'type': 'conceptual',
                    'reference_sentences': relevant_sents[:2]
                })
        
        # Type 2: Factual questions from sentences with numbers or specific facts
        fact_sentences = [s for s in sentences if re.search(r'\d+|percent|%|\$', s)]
        if fact_sentences:
            sent = random.choice(fact_sentences)
            challenges.append({
                'question': f"Based on the document, explain the context of: '{sent[:50]}...'",
                'type': 'factual',
                'reference_sentences': [sent]
            })
        
        # Type 3: Analytical questions
        if len(sentences) > 10:
            challenges.append({
                'question': "What are the main conclusions or findings presented in this document?",
                'type': 'analytical',
                'reference_sentences': sentences[-3:]  # Usually conclusions are at the end
            })
        
        return challenges[:3]  # Return max 3 challenges
    
    def evaluate_answer(self, question, user_answer, reference_sentences):
        if not user_answer.strip():
            return {
                'score': 0,
                'feedback': "Please provide an answer to evaluate.",
                'justification': ""
            }
        
        # Simple evaluation based on keyword matching and length
        reference_text = ' '.join(reference_sentences).lower()
        user_text = user_answer.lower()
        
        # Extract keywords from reference
        ref_words = set(word_tokenize(reference_text))
        ref_words = {word for word in ref_words if word.isalnum() and len(word) > 3}
        
        user_words = set(word_tokenize(user_text))
        user_words = {word for word in user_words if word.isalnum() and len(word) > 3}
        
        # Calculate overlap
        common_words = ref_words.intersection(user_words)
        if len(ref_words) > 0:
            overlap_score = len(common_words) / len(ref_words)
        else:
            overlap_score = 0
        
        # Length penalty for too short answers
        length_score = min(len(user_answer.split()) / 10, 1.0)
        
        final_score = (overlap_score * 0.7 + length_score * 0.3) * 100
        
        if final_score >= 70:
            feedback = "Excellent! Your answer demonstrates good understanding."
        elif final_score >= 50:
            feedback = "Good answer, but could be more comprehensive."
        elif final_score >= 30:
            feedback = "Partial understanding shown, but missing key points."
        else:
            feedback = "Your answer needs improvement. Please refer to the document more carefully."
        
        justification = f"Reference information: {reference_sentences[0][:200]}..."
        
        return {
            'score': round(final_score),
            'feedback': feedback,
            'justification': justification
        }

processor = DocumentProcessor()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_document, document_sentences, tfidf_vectorizer, sentence_vectors
    
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            text = processor.extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        current_document = processor.preprocess_text(text)
        document_sentences = sent_tokenize(current_document)
        
        # Generate summary
        summary = processor.generate_summary(current_document)
        
        return {
            'success': True,
            'summary': summary,
            'filename': file.filename,
            'text_length': len(current_document)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document uploaded")
    
    try:
        answer, justification = processor.generate_answer(request.question, current_document)
        
        return {
            'answer': answer,
            'justification': justification
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/generate-challenges")
async def generate_challenges():
    global current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document uploaded")
    
    try:
        challenges = processor.generate_challenges(current_document)
        
        return {
            'challenges': challenges
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating challenges: {str(e)}")

@app.post("/evaluate")
async def evaluate_answer(request: AnswerRequest):
    global current_document
    
    if not current_document:
        raise HTTPException(status_code=400, detail="No document uploaded")
    
    try:
        # Find relevant sentences for the question
        sentences = sent_tokenize(current_document)
        relevant_info = processor.find_relevant_sentences(request.question, sentences)
        reference_sentences = [info['text'] for info in relevant_info]
        
        if not reference_sentences:
            reference_sentences = sentences[:3]  # Fallback to first few sentences
        
        evaluation = processor.evaluate_answer(
            request.question, 
            request.user_answer, 
            reference_sentences
        )
        
        return evaluation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

@app.get("/")
async def root():
    return {"message": "GenAI Research Assistant API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)