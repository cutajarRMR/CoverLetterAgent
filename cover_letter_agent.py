from docx import Document
import tiktoken
import re
import chroma_memory as cm
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from typing import List, Dict
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()
# Create embeddings object
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def extract_text_from_docx(file_path: str) -> str:
    """Read all paragraph text from a .docx file."""
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_text_from_pdf(file_path: str) -> str:
    """Read all text from a PDF file."""
    import PyPDF2
    
    text_parts = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(page_text)
    
    return "\n".join(text_parts)

def _token_count(_tokenizer, text: str) -> int:
    return len(_tokenizer.encode(text))

def clear_collection(collection_name: str, path: str = './chroma_db'):
    """Delete and recreate a ChromaDB collection."""
    client = __import__('chromadb').PersistentClient(path=path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection doesn't exist yet

def chunk_characters(text: str, chunk_size: int, overlap: int, MAX_TOKENS: int, _tokenizer) -> list[str]:
    """_summary_

    Args:
        text (str): document text to chunk
        chunk_size (int): maximum number of characters in each chunk
        overlap (int): number of characters to overlap between chunks
        MAX_TOKENS (int): maximum number of tokens allowed in each chunk

    Returns:
        list[str]: chunks of text that are within the token limit
    """
    text = " ".join(text.split())

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Safety: if a chunk exceeds the token limit, shrink it
        while _token_count(_tokenizer, chunk) > MAX_TOKENS and len(chunk) > 1:
            chunk = chunk[: len(chunk) - 100]

        chunks.append(chunk.strip())
        start += chunk_size - overlap
        
    return chunks
        
def chunk_text(text: str, chunk_size: int, overlap: int, MAX_TOKENS: int, _tokenizer) -> list[str]:
    """
    First attempts to chunk text by paragraphs and sentences while respecting the token limit. If a single sentence exceeds the token limit, it falls back to character-based chunking for that sentence.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of characters in each chunk.
        overlap (int): The number of characters to overlap between chunks.
        MAX_TOKENS (int): The maximum number of tokens allowed in each chunk.
        _tokenizer: A tokenizer object with an encode method to count tokens.
        
    Returns:
        list[str]: A list of text chunks that are within the token limit.
    """   
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    para = paragraphs[0]

    chunks = []
    for para in paragraphs:
        token_count = _token_count(_tokenizer, para)
        if token_count <= MAX_TOKENS:
            # Paragraph fits add as-is
            chunks.append(para)
            #print("fits")
        else:
            print("no fit")
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
                
            for sentence in sentences:
                test_chunk = f"{current_chunk} {sentence}".strip()
                
                if _token_count(_tokenizer, test_chunk) <= MAX_TOKENS:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                    
                    # Edge case: single sentence exceeds max_tokens
                    if _token_count(_tokenizer, current_chunk) > MAX_TOKENS:
                        # Fall back to character splitting for this sentence only
                        chunks.extend(chunk_characters(current_chunk, chunk_size, overlap, MAX_TOKENS, _tokenizer))
                        current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk)
    return chunks



def define_agent():
    """Defines agent with tools, system prompt, response format, and memory using langchain.

    Returns:
        llm: llm agent with tools and memory
    """

    @tool
    def query_embeddings(
        query: str, 
        collection_name: str, 
        n_results: int = 4, 
        distance_threshold: float = 1  # adjust based on your embedding distance metric
    ) -> List[Dict[str, float]]:
        """
        Queries the embeddings collection and returns IDs and distances,
        only if they are below a distance threshold.

        Args:
            query (str): The text query to search.
            collection_name (str): Name of the collection to query.
            n_results (int, optional): Number of results to return. Defaults to 2.
            distance_threshold (float, optional): Max distance to consider a match. Defaults to 1.
            
        Returns:
            List[Dict[str, float]]: Each dict contains 'text_id' (ID), 'text', and 'distance'.
        """
        try:
            print(f"Querying collection '{collection_name}' with query: '{query}', n_results: {n_results}, distance_threshold: {distance_threshold}")
            embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            result =  cm.query_bank_embeddings(embeddings, query, collection_name, './chroma_db', n_results, distance_threshold,verbose = True)
            print(result)
        except Exception as e:
            print(f"Error querying embeddings: {e}")
            result = []
        return result

    search_tool = TavilySearch(max_results=2,topic="general")



    class OutputCoverLetter(BaseModel):
        """How to return output from the agent"""
        coverLetter: str = Field(description="Enter the cover letter here")
        communication: str = Field(description="Ask any questions or include any comments")
        
    system_prompt = """You are an expert cover letter writing assistant with access to the candidate's resume and previous cover letters.

    ## Your Capabilities
    You have access to the `query_embeddings` tool which can search two collections:
    - **'resume'**: Contains the candidate's resume with their experience, skills, and achievements
    - **'cover_letters'**: Contains the candidate's previously written cover letters for reference

    You have access to the `search_tool` which can search the web for additional information about the company, role, or industry trends to personalize the cover letter.
    ## Your Task
    When the user provides a job description:

    1. **Analyze the job requirements**
    - Extract key skills, qualifications, and responsibilities
    - Identify the most important requirements

    2. **Research the candidate's background**
    - Query the 'resume' collection to find relevant experience matching the job requirements
    - Query the 'cover_letters' collection to find similar past applications and tone/style

    3. **Ask clarifying questions if needed**
    - If critical information is missing, ask the user (e.g., "I see this role requires team leadership. Can you tell me about a specific leadership experience?")
    - Ask about the company if more context would help personalize the letter
    - Indentify any gaps in experience and ask how the candidate would like to address them (e.g., "This role requires Python experience, but I don't see that in your resume. Do you have any related experience or projects you'd like to highlight?")

    4. **Write a compelling cover letter** that:
    - Opens with enthusiasm and mentions the specific position
    - Highlights 2-3 most relevant experiences from their resume that match the job requirements
    - Uses specific examples and achievements (with metrics when available)
    - Demonstrates knowledge of the company/role
    - Maintains a professional yet personable tone similar to their previous cover letters
    - Keeps the length to 3-4 paragraphs (~250-400 words)
    - Closes with a strong call to action

    ## Query Strategy
    - **First**: Query 'resume' with specific skills/requirements from the job description
    - **Then**: Query 'cover_letters' to understand the candidate's writing style and how they've positioned similar experience
    - **Use distance_threshold=1.2** for broader matches, adjust if results are too narrow/broad

    ## Writing Guidelines
    - Be specific and quantifiable (e.g., "reduced pipeline runtime by 40%" not "improved efficiency")
    - Match the job description's language (if they say "Python pipelines," use that exact term)
    - Show enthusiasm but remain professional
    - Never fabricate experience—only use what's in the resume
    - If the candidate lacks required experience, focus on transferable skills and eagerness to learn

    ## Example Interaction Flow
    User: "Write a cover letter for [job description]"
    You: *Query resume for relevant skills → Query cover letters for style → Draft letter OR Ask clarifying questions*
    You: "I found your experience with X, Y, and Z. Here's a draft cover letter..."

    Remember: You have conversation memory, so you can iterate and refine the letter based on user feedback."""

    # Update your agent creation
    agent = create_agent(
        ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"]), 
        tools=[query_embeddings, search_tool],
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt,
        response_format=OutputCoverLetter
    )
    return agent


def embed_resume(resume, chunk_size, overlap, MAX_TOKENS, _tokenizer, clear_first=False):
    """Read the users resume (expects .pdf file), chunk it, and add it to the ChromaDB collection. Optionally clear the collection first if you want to replace old resume data with new.

    Args:
        resume (str): path to the resume file (PDF)
        chunk_size (int): number of characters in each chunk
        overlap (int): number of characters to overlap between chunks
        MAX_TOKENS (int): maximum number of tokens allowed in each chunk
        _tokenizer : tokenizer object to count tokens in chunks
        clear_first (bool, optional): Option to wipe chomaDB. Defaults to False.
    """
    if clear_first:
        clear_collection("resume")
    resume_text = extract_text_from_pdf(resume)
    chunked_res = chunk_text(resume_text, chunk_size, overlap, MAX_TOKENS, _tokenizer)

    # Add documents
    cm.add_to_memery(
        embeddings_ai=embeddings,
        collection_name="resume",
        path="./chroma_db",
        docs=chunked_res,
        ids=[f"chunk_{i}" for i in range(len(chunked_res))]
    )
    print("Resume embedded successfully.")
    

def embed_cover_letters(folder, chunk_size, overlap, MAX_TOKENS, _tokenizer, clear_first=False):
    """Read the users cover letters from a folder. Expects .docx files.
    Chunk it, and add it to the ChromaDB collection. Optionally clear the collection first if you want to replace old resume data with new.

    Args:
        folder (str): Folder path containing cover letter files (.docx)
        chunk_size (int): number of characters in each chunk
        overlap (int): number of characters to overlap between chunks
        MAX_TOKENS (int): maximum number of tokens allowed in each chunk
        _tokenizer : tokenizer object to count tokens in chunks
        clear_first (bool, optional): Option to wipe chomaDB. Defaults to False.
    """
    if clear_first:
        clear_collection("cover_letters")

    for cover in os.listdir(folder):
        if not cover.endswith('.docx'):
            continue
        text = extract_text_from_docx(os.path.join(folder, cover))
        chunked_cover = chunk_text(text, chunk_size, overlap, MAX_TOKENS, _tokenizer)
        cm.add_to_memery(
            embeddings_ai=embeddings,
            collection_name="cover_letters",
            path="./chroma_db",
            docs=chunked_cover,
            ids=[f"{cover}_chunk_{i}" for i in range(len(chunked_cover))]
        )


def embed_cover_letter_files(file_paths, chunk_size, overlap, MAX_TOKENS, _tokenizer, clear_first=False):
    """Embed individual cover letter files (list of file paths) instead of a folder.
    
    Args:
        file_paths (list): List of file paths to cover letters (.docx)
        chunk_size (int): number of characters in each chunk
        overlap (int): number of characters to overlap between chunks
        MAX_TOKENS (int): maximum number of tokens allowed in each chunk
        _tokenizer : tokenizer object to count tokens in chunks
        clear_first (bool, optional): Option to wipe chomaDB. Defaults to False.
    """
    if clear_first:
        clear_collection("cover_letters")

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        text = extract_text_from_docx(file_path)
        chunked_cover = chunk_text(text, chunk_size, overlap, MAX_TOKENS, _tokenizer)
        cm.add_to_memery(
            embeddings_ai=embeddings,
            collection_name="cover_letters",
            path="./chroma_db",
            docs=chunked_cover,
            ids=[f"{filename}_chunk_{i}" for i in range(len(chunked_cover))]
        )


def create_cover_letter(agent, jd):
    """
    Generates a cover letter based on the provided job description (jd) using the defined agent.
    
    Args:
    agent: The language model agent with access to tools and memory.
    jd (str): The job description for which to generate the cover letter.
    
    Returns:
    A tuple containing:
    - communication (str): Any questions or comments from the agent regarding the cover letter.
    - coverLetter (str): The generated cover letter text.
    """
    user_input = f"""Write a cover letter for this job:

    {jd}
    """
    response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"}},
        )
    return response['structured_response'].communication , response['structured_response'].coverLetter

class CoverLetterAgent():
    """
    Cover Letter Agent class to manage resume and cover letter embedding, and cover letter generation based on job descriptions using a language model agent with access to tools and memory.
    
    Methods:
        - enter_resume(resume): Embed the user's resume into the ChromaDB collection.
        - enter_cover_letters(coverlettersfolder): Embed multiple cover letters from a specified folder into the ChromaDB collection.
        - enter_cover_letter_files(file_paths): Embed multiple cover letters from a list of file paths intothe ChromaDB collection.
        - build_cover_letter(jd): Generate a cover letter based on the provided job description (jd) using the defined agent.
        - edit_cover_letter(user_input, coverletter): Edit the existing cover letter based on user feedback 
    """
    def __init__(self, chunk_size=1000, overlap=200, MAX_TOKENS=8192):
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.MAX_TOKENS = MAX_TOKENS
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.agent = define_agent()
        
    def enter_resume(self, resume):
        embed_resume(resume, self.chunk_size, self.overlap, self.MAX_TOKENS, self._tokenizer, clear_first=True)
        
    def enter_cover_letters(self, coverlettersfolder):
        embed_cover_letters(coverlettersfolder, self.chunk_size, self.overlap, self.MAX_TOKENS, self._tokenizer, clear_first=True)
    
    def enter_cover_letter_files(self, file_paths):
        embed_cover_letter_files(file_paths, self.chunk_size, self.overlap, self.MAX_TOKENS, self._tokenizer, clear_first=True)
        
    def build_cover_letter(self, jd):
        self.message, self.cover_letter =  create_cover_letter(self.agent, jd)
        
    def edit_cover_letter(self, user_input, coverletter = None):
        if coverletter is None:
            coverletter = self.cover_letter
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": f"Here is the current cover letter: {coverletter}\n\nUser feedback: {user_input}\n\nPlease edit the cover letter based on the user's feedback."}]},
            {"configurable": {"thread_id": "1"}},
        )
        self.message = response['structured_response'].communication
        self.cover_letter = response['structured_response'].coverLetter