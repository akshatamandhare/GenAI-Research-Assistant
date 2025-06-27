# 🤖 GenAI Research Assistant

A smart AI-powered document analysis tool that helps you understand, summarize, and interact with your research documents using advanced Machine Learning and Natural Language Processing techniques.

## ✨ Features

- **Document Upload**: Support for PDF and TXT files
- **Auto-Summary**: Get concise 150-word summaries of uploaded documents
- **Ask Anything**: Interactive Q&A system with contextual understanding
- **Challenge Mode**: AI-generated questions to test your comprehension
- **Intelligent Evaluation**: ML-powered answer evaluation with detailed feedback
- **Responsive Design**: Beautiful, modern UI that works on all devices
- **No API Dependencies**: Uses local ML/NLP libraries for processing

## 🏗️ Architecture

### Backend (FastAPI)
- **Document Processing**: PyPDF2 for PDF extraction, text preprocessing
- **NLP Pipeline**: NLTK for tokenization, POS tagging, and text analysis
- **ML Components**: 
  - TF-IDF vectorization for document similarity
  - Cosine similarity for relevant sentence extraction
  - Extractive summarization using sentence scoring
- **Question Generation**: Rule-based approach using linguistic patterns
- **Answer Evaluation**: Semantic similarity and keyword matching

### Frontend (React)
- **Modern UI**: Responsive design with glassmorphism effects
- **Tab Navigation**: Intuitive interface for different modes
- **File Handling**: Drag-and-drop file upload
- **Real-time Updates**: Dynamic content rendering
- **Loading States**: User-friendly feedback during processing

### ML/AI Pipeline Flow
1. **Document Ingestion** → Text extraction and preprocessing
2. **Vectorization** → TF-IDF feature extraction
3. **Summarization** → Sentence scoring and selection
4. **Question Answering** → Similarity-based retrieval
5. **Challenge Generation** → Pattern-based question creation
6. **Answer Evaluation** → Semantic matching and scoring

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd genai-research-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

5. **Start the FastAPI server**
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend
npm install
```

2. **Start the React development server**
```bash
npm start
```
The application will be available at `http://localhost:3000`

## 📁 Project Structure

```
genai-research-assistant/
├── backend/
│   ├── main.py                 # FastAPI application
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styling
│   │   └── index.js           # React entry point
│   ├── public/
│   │   └── index.html         # HTML template
│   └── package.json           # Node.js dependencies
└── README.md                  # This file
```

## 🎯 Usage Guide

### 1. Upload Document
- Click on the upload area or drag & drop your PDF/TXT file
- Click "Upload & Analyze" to process the document
- View the auto-generated summary

### 2. Ask Anything Mode
- Navigate to the "Ask Anything" tab
- Type your question about the document
- Get AI-powered answers with source references

### 3. Challenge Mode
- Click "Generate Challenges" to create comprehension questions
- Answer the questions in the text area
- Receive scored feedback with justifications
- Navigate through multiple challenges

## 🧠 ML/AI Technologies Used

### Natural Language Processing
- **NLTK**: Tokenization, POS tagging, stopword removal
- **Sentence Tokenization**: Breaking documents into analyzable units
- **TF-IDF Vectorization**: Converting text to numerical features
- **Cosine Similarity**: Finding relevant content for questions

### Machine Learning Techniques
- **Extractive Summarization**: Selecting important sentences
- **Semantic Search**: Finding relevant document sections
- **Feature Engineering**: Creating meaningful text representations
- **Similarity Scoring**: Ranking content relevance

### Deep Learning Concepts
- **Vector Space Models**: Representing text in high-dimensional space
- **Information Retrieval**: Efficient document search and ranking
- **Contextual Understanding**: Maintaining document context

## 🔧 Configuration

### Backend Configuration
- **CORS**: Configured to allow frontend requests
- **File Upload**: Supports PDF and TXT formats
- **Processing Limits**: Optimized for research documents

### Frontend Configuration
- **API Base URL**: Set to `http://localhost:8000`
- **File Types**: Restricted to `.pdf` and `.txt`
- **Response Timeouts**: Configured for large document processing

## 🎨 Design Features

- **Glassmorphism UI**: Modern translucent design elements
- **Gradient Backgrounds**: Eye-catching color schemes
- **Responsive Layout**: Mobile-first design approach
- **Interactive Elements**: Hover effects and smooth transitions
- **Loading States**: User feedback during processing
- **Error Handling**: Graceful error messages and recovery

## 🚫 No External APIs

This application is designed to work completely offline using local ML libraries:
- No OpenAI API required
- No external ML services
- All processing done locally
- Privacy-focused approach

## 📊 Evaluation Metrics

- **Response Quality**: Accuracy and relevance of answers (30%)
- **Reasoning Functionality**: Quality of generated challenges (20%)
- **UI/UX Experience**: Interface design and flow (15%)
- **Code Quality**: Structure and documentation (15%)
- **Innovation**: Creative features and implementations (10%)
- **Context Accuracy**: Minimal hallucination and good references (10%)

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

2. **CORS Errors**
   - Ensure backend is running on port 8000
   - Check frontend proxy configuration

3. **File Upload Issues**
   - Verify file format (PDF/TXT only)
   - Check file size limits

4. **Processing Errors**
   - Ensure document has readable text
   - Check Python dependencies are installed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is created for educational and research purposes.

## 🎯 Future Enhancements

- [ ] Support for more file formats (DOCX, RTF)
- [ ] Advanced summarization techniques
- [ ] Multi-language support
- [ ] Document comparison features
- [ ] Export functionality for summaries and Q&A
- [ ] User session management
- [ ] Advanced visualization of document insights

---

**Built with ❤️ using FastAPI, React, and advanced ML/NLP techniques**