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
            results = list(executor.map(lambda skill: self.analyze_skill(qa_chain, skill), skills))

        for skill, score, reasoning in results:
            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning
            total_score += score
            if score <= 5:
                missing_skills.append(skill)

        overall_score = int((total_score / (len(skills) * 10)) * 100)
        selected = overall_score >= self.cutoff_score

        reasoning = "Candidate evaluated based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        improvement_areas = missing_skills if not selected else []

        self.resume_strength = strengths

        return {
            "overall_score": overall_score,
            "selected": selected,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "missing_skills": missing_skills,
            "reasoning": reasoning,
            "strengths": strengths,
            "improvement_areas": improvement_areas
        }

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):

        self.resume_text = self.extract_text_from_file(resume_file)


        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name
        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extracted_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)

            self.analysis_results = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)
        elif role_requirements:
            self.extracted_skills = role_requirements
            self.analysis_results = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)

        if self.analysis_results and "missing_skills" in self.analysis_results and self.analysis_results["missing_skills"]:
            self.analyze_resume_weakness()

            self.analysis_results["detailed_weaknesses"] = self.resume_weakness
        return self.analysis_results


# Ask questions Feature         

    def ask_questions(self, question):

        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first to enable question answering."

        retriever = self.rag_vectorstore.as_retriever( search_kwargs={"k": 3} )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o", api_key=self.api_key),
            chain_type = "stuff",
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain.run(question)
        return response
    

    # Interview Preparation Feature

    def generate_interview_questions(self, question_types, difficulty, num_questions):

        if not self.resume_text or not self.extracted_skills:
            return []
        
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            context = f"""
            Resume Content:
            {self.resume_text[:3000]}...

            skills:
            {', '.join(self.extracted_skills)}
            Strengths: {', '.join(self.analysis_results.get('strengths', []))}
            Areas for Improvement: {', '.join(self.analysis_results.get('missing_skills', []))}
            """

            prompt = f"""
            Generate {num_questions} personalized {difficulty.lower()}-difficulty interview questions based on the candidate's resume and skills. Focus on the following question types: {', '.join(question_types)}.

            For each question, provide:
            1. Clearly label the question type
            2. Make the question relevant to the candidate's experience and skills
            3. For coding questions, include a specific problem statement and expected input/output format.

            {context}
            
            Format the response as a list pf tuples with the question type and the question itself. 
            Each tuple should be in the format: ("Question Type", "Full Question Text")
            """

            response = llm.invoke(prompt)
            questions_text = response.content.strip()
            questions = []
            pattern = r'[("]([^"]+)[",)\s]+[",\s]+([^"]+)[")\s]+'
            matches = re.findall(pattern, questions_text, re.DOTALL)

            for match in matches:
                if len(match) >= 2:
                    question_type = match[0].strip()
                    question = match[1].strip()

                    for requested_type in question_types:
                        if requested_type.lower() in question_type.lower():
                            questions.append((requested_type, question))
                            break
            if not questions:
                lines = questions_text.split('\n')
                current_type = None
                current_question = ""
                for line in lines:
                    line = line.strip()
                    if any(t.lower() in line.lower() for t in question_types) and not current_question:
                        current_type = next((t for t in question_types if t.lower() in line.lower()), None)
                        if ":" in line:
                            current_type = line.split(":", 1)[1].strip()
                    elif current_type and line:
                        current_question += " " + line        
                    elif current_type and current_question:
                        questions.append((current_type, current_question.strip()))
                        current_type = None
                        current_question = ""
            questions = questions[:num_questions]
            return questions
        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []            
    def improve_resume(self, improvement_areas, target_role=""):

        if not self.resume_text:
            return {}
        
        try:
            improvements = {}

            for area in improvement_areas:

                if area == "Skills Highlighting" and self.resume_weaknessess:
                    skill_improvements = {
                        "description": "Your resume needs to better highlight your proficiency in certain skills that are important for the target role.",
                        "specific": []
                    }

                    before_after_examples = {}

                    for weakness in self.resume_weaknessess:
                        skill_name = weakness.get("skill", "")
                        if "suggestions" in weakness and weakness["suggestions"]:
                            for suggestion in weakness["suggestions"]:
                                skill_improvements["specific"].append(f"**{skill_name}**: {suggestion}")
                        if "example" in weakness and weakness["example"]:

                            resume_chunks = self.resume_text.split('\n')
                            relevant_chunk = ""

                            for chunk in resume_chunks:
                                if skill_name.lower() in chunk.lower():
                                    relevant_chunk = chunk
                                    break        
                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip() + "\n " + weakness["example"]
                                }
                    if before_after_examples:
                        skill_improvements["before_after_examples"] = before_after_examples
                    improvements["Skills Highlighting"] = skill_improvements         
            remaining_areas = [area for area in improvement_areas if area not in improvements]

            if remaining_areas:
                llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)

                weaknessess_text = ""
                if self.resume_weaknessess:
                    weaknessess_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.resume_weaknessess):
                        weaknessess_text += f"{i+1}. {weakness['skill']}: {weakness['detail']}\n"
                        if "suggestions" in weakness:
                            for j, sugg in enumerate(weakness["suggestions"]):
                                weaknessess_text += f"   - {sugg}\n"               
                context = f"""
                Resume Content:
                {self.resume_text}

                Skills to focus on: {', '.join(self.extracted_skills)}
                Strengths: {', '.join(self.analysis_results.get('strengths', []))}

                Areas for improvement: {', '.join(self.analysis_results.get('missing_skills', []))}

                {weaknessess_text}

                Target Role: {target_role if target_role else "Not specified"} 
                """

                prompt = f"""Provide specific suggestions to improve the resume in the area of {', '.join(remaining_areas)} for a candidate targeting the role of {target_role}. Base your suggestions on the resume content and any identified weaknesses.

                {context}

                For each improvement area, provide:
                1. A clear description of what needs improvement.
                2. 3-5 specific actionable suggestions to enhance that area of the resume.
                3. Where relevant, provide a before/after example

                Format the response as a JSON object with each improvement area as a key and its corresponding suggestions and examples as values:
                - " description" : general description
                - "specific_suggestions": list of specific suggestions
                - "before_after": (where applicable) a dict with "before" and "after" examples
                
                Only include the requested improvement areas that aren't already covered.
                Focus particularly on addressing the resume weaknesses identified.
                """

                response = llm.invoke(prompt)
                

                # Try to parse the response as JSON
                ai_improvements = {}

                # Extract JSON from the response
                json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response.content)
                if json_match:
                    try:
                        ai_improvements = json.loads(json_match.group(1))

                        improvements.update(ai_improvements)
                    except json.JSONDecodeError:
                        pass
                if not ai_improvements:
                    sections = response.content.split('##')

                    for section in sections:
                        if not section.strip():
                            continue


                        lines = section.strip().split('\n')
                        area = None

                        for line in lines:
                            if not area and line.strip():
                                area = line.strip()
                                improvements[area] = {
                                    "description": "",
                                    "specific": []
                                }
                            elif area and "specific" in improvements[area]:
                                if line.strip().startswith('-'):
                                    improvements[area]["specific"].append(line.strip()[2:].strip())        
                                elif not improvements[area]["description"]:
                                    improvements[area]["description"] += line.strip()
            for area in improvement_areas:
                if area not in improvements:
                    improvements[area] = {
                        "description": "No specific suggestions provided.",
                        "specific": ["Review and Enhance this section"]
                    }
            return improvements

        except Exception as e:
            print(f"Error generating resume improvement suggestions: {e}")
            return {area: {"description": "Error generating suggestions.", "specific": []} for area in improvement_areas}





    def get_improved_resume(self, target_role="", highlight_skills=""):

        if not self.resume_text:
            return "Please upload and analyze a resume first."
        
            def get_improved_resume(self, target_role="", highlight_skills=""):

        if not self.resume_text:
            return "Please upload and analyze a resume first."

        
        try:

            skills_to_highlight = []
            if highlight_skills:

                if len(highlight_skills) > 100:
                    self.jd_text = highlight_skills

                    try:
                        parsed_skills = self.extract_skills_from_jd(highlight_skills)
                        if parsed_skills:
                            skills_to_highlight = parsed_skills
                        else:
                            skills_to_highlight = [s.strip() for s in highlight_skills.split(',') if s.strip()]
                    except:

                        skills_to_highlight = [s.strip() for s in highlight_skills.split(',') if s.strip()]
                else:
                    skills_to_highlight = [s.strip() for s in highlight_skills.split(',') if s.strip()]

            if not skills_to_highlight and self.analysis_results:

                skills_to_highlight = self.analysis_results.get('missing_skills', [])
                skills_to_highlight.extend([
                    skill for skill in self.analysis_results.get('strengths', []) if skill not in skills_to_highlight
                ])

                if self.extracted_skills:
                    skills_to_highlight.extend([skill for skill in self.extracted_skills if skill not in skills_to_highlight])

            weaknesses_context = ""
            improvement_suggestions = ""                                 
            
            
            if self.resume_weakness:
                weaknesses_context = "Address this specific weakness:\n"
                for weakness in self.resume_weaknessess:
                    skill_name = weakness.get("skill", "")
                    weaknesses_context += f"- {skill_name}: {weakness.get('details', '')}\n"


                    if 'suggestions' in weakness and weakness['suggestions']:
                        weaknesses_context += "  Suggestions:\n"
                        for suggestion in weakness['suggestions']:
                            weaknesses_context += f"   * {suggestion}\n"

                    if 'example' in weakness and weakness['example']:
                        improvement_suggestions += f"For {skill_name} : {weakness['example']}\n\n"

            skills_to_highlight = highlight_skills if highlight_skills else ", ".join(strengths)

            prompt = f"""
You are an expert technical recruiter and resume strategist.

Rewrite the following resume to make it stronger, more impactful, and tailored for the role of {target_role if target_role else "the desired role"}.

Objectives:
1. Strengthen weak skill areas.
2. Emphasize measurable achievements.
3. Improve clarity and ATS optimization.
4. Highlight these skills prominently: {skills_to_highlight}
5. Make bullet points action-oriented.
6. Remove fluff and generic statements.

Original Resume:
-------------------
{self.resume_text}

Analysis Insights:
-------------------
Strengths: {', '.join(strengths)}
Missing Skills / Improvement Areas: {', '.join(improvement_areas)}

{weaknesses_context}

Instructions:
- Rewrite the entire resume professionally.
- Keep it realistic — do NOT fabricate fake companies or fake experience.
- Enhance phrasing and add quantified impact where reasonable.
- Optimize formatting for ATS readability.
- Return ONLY the improved resume text.
"""

            response = llm.invoke(prompt)
            improved_resume = response.content.strip()

            return {
                "target_role": target_role,
                "highlighted_skills": skills_to_highlight,
                "improved_resume": improved_resume
            }

        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return {"error": "Failed to generate improved resume."}