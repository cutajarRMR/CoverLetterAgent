# Cover Letter Agent

This project implements a Cover Letter Agent using Langchain, ChromaDB, and OpenAI's language models. The agent can read and embed a user's resume and cover letters, and generate tailored cover letters based on job descriptions. It also allows for editing generated cover letters based on user feedback.

## Features
- RAG: Retrieve relevant information from the user's resume and past cover letters to inform the generation of new cover letters.
- Text Embedding: Convert resume and cover letter text into vector embeddings for efficient retrieval.
- Custom Tools: The agent has access to tools for querying embedded data and generating cover letters.
- Memory: The agent can remember past interactions and use that information to improve future responses.
- User Feedback Loop: Users can provide feedback on generated cover letters, and the agent can edit the cover letter accordingly.

## Usage
```python
from cover_letter_agent import CoverLetterAgent
cv = CoverLetterAgent()
cv.enter_resume("path/to/resume.pdf")
cv.enter_cover_letter_files(["path/to/coverletter1.docx", "path/to/coverletter2.docx"])
cv.build_cover_letter("Job description text here")
print(cv.cover_letter)
print(cv.message)
cv.edit_cover_letter("Make it more concise and engaging, and highlight my passion for learning and problem-solving.", cover_letter)
```

## GUI

A simple Flask web interface is included for users who prefer a graphical interface to interact with the Cover Letter Agent. The GUI allows users to upload their resume and cover letters, input job descriptions, and view generated cover letters.

```python
python app.py
```

Then navigate to `http://localhost:5000` in your web browser to access the interface.


## API Keys
You will require an OpenAI API key
You will require a Tavily API key