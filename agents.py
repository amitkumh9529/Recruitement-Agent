import os
import io
import json
import re
import pyPDF2
import tempfile

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor


class ResumeAnalysisAgent:
    def __init__(self, api_key, cutoff_score=75):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analysis_results = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weakness = []
        self.resume_strength = []
        self.improvement_suggestions = {}


    def extract_text_from_pdf(self, pdf_file):

        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = pyPDF2.PdfReader(pdf_file_like)
            else:
                reader = pyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return "" 

    def extract_text_from_txt(self, txt_file):
        try:
            if hasattr(txt_file, 'getvalue'):
                txt_data = txt_file.getvalue()
                return txt_data.decode('utf-8')
            else:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
                
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            return ""

    def extract_text_from_file(self, file):  

        if hasattr(file, 'name'):
            file_extention = file.name.split('.')[-1].lower()
        else:
            file_extention = file.filename.split('.')[-1].lower()

        if file_extention == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extention == 'txt':
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file type: {file_extention}")
            return ""  


    def create_rag_vector_store(self, text):
        # create an Vector store for RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def create_vector_store(self, text):

        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        vectorstore = FAISS.from_texts([text], embeddings)
        return vectorstore       

    def analyze_skill(self, qa_chain, skill):

        query = f"On scale of 0-100, how proficient is the resume in {skill}? Answer with a number and provide a brief explanation."
        response = qa_chain.run(query)
        match = re.search(r'^(\d{1,2})', response)
        score = int(match.group(1)) if match else 0

        reasoning = response.split('.', 1)[1].strip() if '.' in response and len(response.split('.', 1)) > 1 else "No explanation provided."
        return skill, min(score, 10), reasoning
    

    def analyze_resume_weakness(self):

        if not self.resume_text or not self.extracted_skills or not self.analysis_results:
            return []
        weaknesses = []
        for skill in self.analysis_results.get('missing_skills', []):
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            prompt = f"""Analyze Why the resume is weak in demonstrating proficiency in {skill} and provide suggestions for improvement."

            for your analysis, consider the following:
            1. What's missing in the resume that would demonstrate proficiency in {skill}?
            2. How could be it improved with specific examples or details?
            3. What specific action Item would make this skill stand out more in the resume?

            Resume Context:
            {self.resume_text[:3000]}...

            provide your response in the following JSON format:
            {{ 
                "weakness": "A concise description of what's missing or problematic (1-2 sentences).",
                "improvement_suggestions": [
                    "Specific suggestion 1",  
                    "Specific suggestion 2",
                    "Specific suggestion 3"
                ],
                "example_additions": "A specific bullet point or sentence that could be added to the resume to demonstrate this skill more effectively."
            }}   

            Return only valid JSON, no other text or explanations. 
            """

            response = llm.invoke(prompt)
            weakness_context = response.content.strip()

            try:
                weakness_data = json.loads(weakness_context)
                weakness_detail = {
                    "skill": skill,
                    "score": self.analysis_results.get("skill_scores", {}).get(skill, 0),
                    "details": weakness_data.get("weakness", "No weakness description provided."),
                    "improvement_suggestions": weakness_data.get("improvement_suggestions", []),
                    "example_additions": weakness_data.get("example_additions", "No example provided.")
                }

                weaknesses.append(weakness_detail)

                self.improvement_suggestions[skill] = {
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example_additions": weakness_data.get("example_additions", "No example provided.")
                }

            except json.JSONDecodeError:
                weaknesses.append({
                    "skill": skill,
                    "score": self.analysis_results.get("skill_scores", {}).get(skill, 0),
                    "details": weakness_context[:200] # Truncate to avoid excessively long text in case of parsing failure
                })

            self.resume_weakness = weaknesses
            return weaknesses

    def extract_skills_from_jd(self, jd_text):
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            prompt = f"""Extract a comprehensive list of technical skills, technologies and competencies required from this job description. Format the output as a python list of strings. Only include the list, nothing else.

            Job Description:
            {jd_text}

            """
            response = llm.invoke(prompt)
            skills_text = response.content.strip()

            match = re.search(r'\[.*?\]', skills_text, re.DOTALL)
            if match:
               skills_text = match.group(0)
            try:
                skills_list = eval(skills_text)
                if isinstance(skills_list, list):
                    return skills_list
            except:
                pass

            skills = []
            for line in skills_text.splitlines('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    skill = line[2:].strip()
                    if skill:
                        skills.append(skill)
                elif line.startswith('"') and line.endswith('"'):
                    skill = line.strip('"')
                    if skill:
                        skills.append(skill)
            return skills
        except Exception as e:
            print(f"Error extracting skills from JD: {e}")
            return []
        
    def semantic_skill_analysis(self, resume_text, skills):

        vectorstore = self.create_vector_store(resume_text)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o", api_key=self.api_key),
            retriever=retriever,
            return_source_documents=False
        )

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            

         

        

        

       
