import os
import io
import json
import re
import PyPDF2
import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # updated — llama3-8b-8192 is decommissioned

print("agents.py loaded — version GROQ-FIXED")


def _get_llm():
    """Return a ChatGroq instance using the key from .env"""
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


def _get_embeddings():
    """Return local HuggingFace embeddings (no API key needed)"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class ResumeAnalysisAgent:
    def __init__(self, api_key=None, cutoff_score=75):
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


    # ─────────────────────────────────────────────
    # TEXT EXTRACTION
    # ─────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)
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
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.filename.split('.')[-1].lower()

        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file type: {file_extension}")
            return ""


    # ─────────────────────────────────────────────
    # VECTOR STORES
    # ─────────────────────────────────────────────

    def create_rag_vector_store(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embeddings = _get_embeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def create_vector_store(self, text):
        embeddings = _get_embeddings()
        vectorstore = FAISS.from_texts([text], embeddings)
        return vectorstore


    # ─────────────────────────────────────────────
    # SKILL ANALYSIS
    # ─────────────────────────────────────────────

    def _build_qa_chain(self, vectorstore):
    
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the provided context.\n\n"
            "Context: {context}\n\n"
            "Question: {input}"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def analyze_skill(self, qa_chain, skill):
        query = (
            f"On a scale of 0-100, how proficient is the candidate in {skill}? "
            f"Start your answer with just the number, then a period, then a brief explanation."
        )
    # New chain returns string directly, not a dict
        answer = qa_chain.invoke(query)

        match = re.search(r'(\d{1,3})', answer)
        score = int(match.group(1)) if match else 0
        score = min(score, 100)
        score_10 = min(round(score / 10), 10)

        reasoning = answer.split('.', 1)[1].strip() if '.' in answer else "No explanation provided."
        return skill, score_10, reasoning

    def analyze_resume_weakness(self):
        if not self.resume_text or not self.extracted_skills or not self.analysis_results:
            return []

        weaknesses = []
        for skill in self.analysis_results.get('missing_skills', []):
            llm = _get_llm()
            prompt = f"""Analyze why the resume is weak in demonstrating proficiency in {skill} and provide suggestions for improvement.

            For your analysis, consider the following:
            1. What's missing in the resume that would demonstrate proficiency in {skill}?
            2. How could it be improved with specific examples or details?
            3. What specific action item would make this skill stand out more in the resume?

            Resume Context:
            {self.resume_text[:3000]}...

            Provide your response in the following JSON format:
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

            weakness_context = re.sub(r'^```(?:json)?\s*', '', weakness_context)
            weakness_context = re.sub(r'\s*```$', '', weakness_context)

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
                    "details": weakness_context[:200]
                })

        self.resume_weakness = weaknesses
        return weaknesses


    def extract_skills_from_jd(self, jd_text):
        try:
            llm = _get_llm()
            prompt = f"""Extract a comprehensive list of technical skills, technologies and competencies from this text.
            Format the output as a Python list of strings. Only include the list, nothing else.

            Text:
            {jd_text[:3000]}
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
            for line in skills_text.splitlines():
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
        qa_chain = self._build_qa_chain(vectorstore)

        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        for skill in skills:
            s, score, reasoning = self.analyze_skill(qa_chain, skill)
            skill_scores[s] = score
            skill_reasoning[s] = reasoning
            total_score += score
            if score <= 5:
                missing_skills.append(s)

        overall_score = int((total_score / (len(skills) * 10)) * 100) if skills else 0
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
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)
            self.analysis_results = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)

        elif role_requirements:
            self.extracted_skills = role_requirements
            self.analysis_results = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)

        else:
            # ── No JD or skills provided — auto extract from resume itself
            print("No JD or skills provided — auto-extracting skills from resume...")
            self.extracted_skills = self.extract_skills_from_jd(self.resume_text)

            if not self.extracted_skills:
                print("Skill extraction returned empty — using fallback skills")
                self.extracted_skills = [
                    "Communication", "Problem Solving", "Teamwork",
                    "Technical Skills", "Time Management"
                ]

            self.analysis_results = self.semantic_skill_analysis(
                self.resume_text, self.extracted_skills
            )

        if self.analysis_results and self.analysis_results.get("missing_skills"):
            self.analyze_resume_weakness()
            self.analysis_results["detailed_weaknesses"] = self.resume_weakness

        return self.analysis_results


    # ─────────────────────────────────────────────
    # Q&A
    # ─────────────────────────────────────────────

    def ask_questions(self, question):
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first to enable question answering."

        qa_chain = self._build_qa_chain(self.rag_vectorstore)
    # New chain returns string directly
        answer = qa_chain.invoke(question)
        return answer


    # ─────────────────────────────────────────────
    # INTERVIEW QUESTIONS
    # ─────────────────────────────────────────────

    def generate_interview_questions(self, question_types, difficulty, num_questions):
        if not self.resume_text or not self.extracted_skills:
            return []

        try:
            llm = _get_llm()
            context = f"""
            Resume Content:
            {self.resume_text[:3000]}...

            Skills: {', '.join(self.extracted_skills)}
            Strengths: {', '.join(self.analysis_results.get('strengths', []))}
            Areas for Improvement: {', '.join(self.analysis_results.get('missing_skills', []))}
            """
            prompt = f"""Generate {num_questions} personalized {difficulty.lower()}-difficulty interview questions based on the candidate's resume and skills.
            Focus on the following question types: {', '.join(question_types)}.

            {context}

            Format the response as a numbered list. Each item should be:
            [Question Type]: Full question text

            Example:
            1. [Technical]: Explain how you used Python in your previous role.
            2. [Behavioral]: Tell me about a time you had to learn a new technology quickly.
            """
            response = llm.invoke(prompt)
            questions_text = response.content.strip()
            questions = []

            lines = questions_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^\d+\.\s*\[([^\]]+)\]:\s*(.+)', line)
                if match:
                    q_type = match.group(1).strip()
                    q_text = match.group(2).strip()
                    matched_type = next(
                        (t for t in question_types if t.lower() in q_type.lower()),
                        q_type
                    )
                    questions.append((matched_type, q_text))

            if not questions:
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        text = re.sub(r'^\d+\.\s*', '', line)
                        matched_type = next(
                            (t for t in question_types if t.lower() in text.lower()),
                            question_types[0] if question_types else "General"
                        )
                        questions.append((matched_type, text))

            return questions[:num_questions]

        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []


    # ─────────────────────────────────────────────
    # RESUME IMPROVEMENT
    # ─────────────────────────────────────────────

    def improve_resume(self, improvement_areas, target_role=""):
        if not self.resume_text:
            return {}

        try:
            improvements = {}

            for area in improvement_areas:
                if area == "Skills Highlighting" and self.resume_weakness:
                    skill_improvements = {
                        "description": "Your resume needs to better highlight your proficiency in certain skills that are important for the target role.",
                        "specific": []
                    }
                    before_after_examples = {}

                    for weakness in self.resume_weakness:
                        skill_name = weakness.get("skill", "")
                        if weakness.get("improvement_suggestions"):
                            for suggestion in weakness["improvement_suggestions"]:
                                skill_improvements["specific"].append(f"{skill_name}: {suggestion}")
                        if weakness.get("example_additions"):
                            resume_chunks = self.resume_text.split('\n')
                            relevant_chunk = ""
                            for chunk in resume_chunks:
                                if skill_name.lower() in chunk.lower():
                                    relevant_chunk = chunk
                                    break
                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip() + "\n " + weakness["example_additions"]
                                }

                    if before_after_examples:
                        skill_improvements["before_after_examples"] = before_after_examples
                    improvements["Skills Highlighting"] = skill_improvements

            remaining_areas = [area for area in improvement_areas if area not in improvements]

            if remaining_areas:
                llm = _get_llm()

                weaknesses_text = ""
                if self.resume_weakness:
                    weaknesses_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.resume_weakness):
                        weaknesses_text += f"{i+1}. {weakness['skill']}: {weakness.get('details', '')}\n"
                        for sugg in weakness.get("improvement_suggestions", []):
                            weaknesses_text += f"   - {sugg}\n"

                context = f"""
                Resume Content:
                {self.resume_text}

                Skills to focus on: {', '.join(self.extracted_skills or [])}
                Strengths: {', '.join(self.analysis_results.get('strengths', []))}
                Areas for improvement: {', '.join(self.analysis_results.get('missing_skills', []))}

                {weaknesses_text}

                Target Role: {target_role if target_role else "Not specified"}
                """
                prompt = f"""Provide specific suggestions to improve the resume in these areas: {', '.join(remaining_areas)}
                for a candidate targeting the role of {target_role if target_role else "a relevant position"}.

                {context}

                Format the response as a JSON object:
                {{
                    "Area Name": {{
                        "description": "general description",
                        "specific_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
                    }}
                }}

                Return only valid JSON, no other text.
                """
                response = llm.invoke(prompt)
                ai_response = response.content.strip()
                ai_response = re.sub(r'^```(?:json)?\s*', '', ai_response)
                ai_response = re.sub(r'\s*```$', '', ai_response)

                try:
                    ai_improvements = json.loads(ai_response)
                    improvements.update(ai_improvements)
                except json.JSONDecodeError:
                    for area in remaining_areas:
                        improvements[area] = {
                            "description": "Review and enhance this section.",
                            "specific_suggestions": ["Tailor content to the target role", "Use action verbs", "Add quantifiable achievements"]
                        }

            for area in improvement_areas:
                if area not in improvements:
                    improvements[area] = {
                        "description": "No specific suggestions provided.",
                        "specific_suggestions": ["Review and enhance this section"]
                    }

            return improvements

        except Exception as e:
            print(f"Error generating resume improvement suggestions: {e}")
            return {area: {"description": "Error generating suggestions.", "specific_suggestions": []} for area in improvement_areas}


    # ─────────────────────────────────────────────
    # RESUME REWRITE
    # ─────────────────────────────────────────────

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
                        skills_to_highlight = parsed_skills if parsed_skills else [s.strip() for s in highlight_skills.split(',') if s.strip()]
                    except:
                        skills_to_highlight = [s.strip() for s in highlight_skills.split(',') if s.strip()]
                else:
                    skills_to_highlight = [s.strip() for s in highlight_skills.split(',') if s.strip()]

            if not skills_to_highlight and self.analysis_results:
                skills_to_highlight = self.analysis_results.get('missing_skills', [])
                skills_to_highlight.extend([
                    skill for skill in self.analysis_results.get('strengths', [])
                    if skill not in skills_to_highlight
                ])
                if self.extracted_skills:
                    skills_to_highlight.extend([
                        skill for skill in self.extracted_skills
                        if skill not in skills_to_highlight
                    ])

            weaknesses_context = ""
            improvement_suggestions = ""

            if self.resume_weakness:
                weaknesses_context = "Address these specific weaknesses:\n"
                for weakness in self.resume_weakness:
                    skill_name = weakness.get("skill", "")
                    weaknesses_context += f"- {skill_name}: {weakness.get('details', '')}\n"
                    for suggestion in weakness.get("improvement_suggestions", []):
                        weaknesses_context += f"   * {suggestion}\n"
                    if weakness.get("example_additions"):
                        improvement_suggestions += f"For {skill_name}: {weakness['example_additions']}\n\n"

            llm = _get_llm()

            jd_context = ""
            if self.jd_text:
                jd_context = f"Job Description:\n{self.jd_text}\n\n"
            elif target_role:
                jd_context = f"Target role: {target_role}\n\n"

            prompt = f"""You are an expert technical recruiter and resume strategist.

                Rewrite the following resume to make it stronger, more impactful, and tailored for the role of {target_role if target_role else "the desired role"}.

                {jd_context}

                Original Resume:
                -------------------
                {self.resume_text}

                Skills to highlight (in order of priority): {', '.join(skills_to_highlight)}

                Objectives:
                1. Strengthen weak skill areas.
                2. Emphasize measurable achievements.
                3. Improve clarity and ATS optimization.
                4. Highlight these skills prominently: {', '.join(skills_to_highlight)}
                5. Make bullet points action-oriented.
                6. Remove fluff and generic statements.

                Analysis Insights:
-------------------
                Strengths: {', '.join(self.analysis_results.get('strengths', []) if self.analysis_results else [])}
                Missing Skills: {', '.join(self.analysis_results.get('missing_skills', []) if self.analysis_results else [])}

                {weaknesses_context}

                Instructions:
                - Rewrite the entire resume professionally.
                - Do NOT fabricate fake companies or fake experience.
                - Enhance phrasing and add quantified impact where reasonable.
                - Optimize formatting for ATS readability.
                - Return ONLY the improved resume text, no commentary.
                """
            response = llm.invoke(prompt)
            improved_resume = response.content.strip()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
                tmp.write(improved_resume)
                self.improved_resume_path = tmp.name

            return improved_resume

        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return "Error: Failed to generate improved resume."


    # ─────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────

    def cleanup(self):
        try:
            if hasattr(self, 'resume_file_path') and os.path.exists(self.resume_file_path):
                os.unlink(self.resume_file_path)
            if hasattr(self, 'improved_resume_path') and os.path.exists(self.improved_resume_path):
                os.unlink(self.improved_resume_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")