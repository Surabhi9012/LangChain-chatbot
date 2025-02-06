# Technical Course Chatbot with Langchain

## Overview
A sophisticated Flask-based chatbot that extracts and retrieves information from technical course websites using advanced NLP techniques.

## Features
- Web scraping of course information
- AI-powered conversational retrieval
- Persistent vector store
- Robust error handling
- Configurable conversational memory

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/technical-course-chatbot.git
cd technical-course-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Run Application
```bash
python app.py
```

### API Endpoint
- **URL:** `/api/chat`
- **Method:** POST
- **Request Body:** 
  ```json
  {
    "question": "What Python courses are available?"
  }
  ```

## Technologies
- Flask
- Langchain
- OpenAI
- ChromaDB
- Unstructured

## Architecture
1. Document Loading
2. Text Splitting
3. Embedding Generation
4. Vector Store Creation
5. Conversational Retrieval

## Logging
Comprehensive logging implemented to track application performance and errors.

## Error Handling
- 400 Bad Request
- 404 Not Found
- 500 Internal Server Error

## Contributing
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request
