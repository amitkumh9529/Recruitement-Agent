import os
import json
import tempfile
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Import the ResumeAnalysisAgent from agents.py
from agents import ResumeAnalysisAgent

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# In-memory agent sessions keyed by session_id
agent_sessions: dict = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_agent(session_id: str, cutoff_score: int = 75) -> ResumeAnalysisAgent:
    """Get or create an agent for the given session."""
    if session_id not in agent_sessions:
        agent_sessions[session_id] = ResumeAnalysisAgent(
            api_key=GROQ_API_KEY,
            cutoff_score=cutoff_score
        )
    return agent_sessions[session_id]


# ─────────────────────────────────────────────
# 1. ANALYZE RESUME
# ─────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    session_id = request.form.get('session_id', '').strip()
    cutoff_raw = request.form.get('cutoff_score', '75')
    cutoff_score = int(cutoff_raw) if cutoff_raw and cutoff_raw.isdigit() else 75

    if not session_id:
        return jsonify({'error': 'session_id is required.'}), 400

    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided.'}), 400

    resume_file = request.files['resume']
    if not allowed_file(resume_file.filename):
        return jsonify({'error': 'Only PDF and TXT files are supported.'}), 400

    # Save resume to temp file
    suffix = '.' + resume_file.filename.rsplit('.', 1)[1].lower()
    tmp_resume = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    resume_file.save(tmp_resume.name)
    tmp_resume.close()
    tmp_resume_path = tmp_resume.name

    # Optional JD file
    jd_tmp_path = None
    if 'job_description' in request.files:
        jd_file = request.files['job_description']
        if jd_file and jd_file.filename and allowed_file(jd_file.filename):
            jd_suffix = '.' + jd_file.filename.rsplit('.', 1)[1].lower()
            tmp_jd = tempfile.NamedTemporaryFile(delete=False, suffix=jd_suffix)
            jd_file.save(tmp_jd.name)
            tmp_jd.close()
            jd_tmp_path = tmp_jd.name

    # Role requirements (comma-separated skills)
    role_requirements_raw = request.form.get('role_requirements', '').strip()
    role_requirements = [s.strip() for s in role_requirements_raw.split(',') if s.strip()] \
        if role_requirements_raw else None

    resume_named = None
    jd_named = None

    try:
        agent = get_agent(session_id, cutoff_score)

        class NamedFile:
            def __init__(self, path):
                self.name = path
                self._f = open(path, 'rb')
            def getvalue(self):
                self._f.seek(0)
                return self._f.read()
            def close(self):
                try:
                    self._f.close()
                except Exception:
                    pass

        resume_named = NamedFile(tmp_resume_path)
        jd_named = NamedFile(jd_tmp_path) if jd_tmp_path else None

        results = agent.analyze_resume(
            resume_file=resume_named,
            role_requirements=role_requirements,
            custom_jd=jd_named
        )

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Always close and delete temp files — fixes Windows WinError 32
        if resume_named:
            resume_named.close()
        if jd_named:
            jd_named.close()
        try:
            if os.path.exists(tmp_resume_path):
                os.unlink(tmp_resume_path)
        except Exception:
            pass
        if jd_tmp_path:
            try:
                if os.path.exists(jd_tmp_path):
                    os.unlink(jd_tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────
# 2. ASK A QUESTION ABOUT THE RESUME
# ─────────────────────────────────────────────
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json(force=True)
    session_id = data.get('session_id', '').strip()
    question = data.get('question', '').strip()

    if not session_id:
        return jsonify({'error': 'session_id is required.'}), 400
    if not question:
        return jsonify({'error': 'question is required.'}), 400
    if session_id not in agent_sessions:
        return jsonify({'error': 'Session not found. Please analyse a resume first.'}), 404

    try:
        agent = get_agent(session_id)
        answer = agent.ask_questions(question)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# 3. GENERATE INTERVIEW QUESTIONS
# ─────────────────────────────────────────────
@app.route('/api/interview-questions', methods=['POST'])
def interview_questions():
    data = request.get_json(force=True)
    session_id = data.get('session_id', '').strip()
    question_types = data.get('question_types', ['Technical', 'Behavioral'])
    difficulty = data.get('difficulty', 'Medium')
    num_questions = int(data.get('num_questions', 5))

    if not session_id:
        return jsonify({'error': 'session_id is required.'}), 400
    if session_id not in agent_sessions:
        return jsonify({'error': 'Session not found. Please analyse a resume first.'}), 404

    try:
        agent = get_agent(session_id)
        questions = agent.generate_interview_questions(
            question_types=question_types,
            difficulty=difficulty,
            num_questions=num_questions
        )
        formatted = [{'type': q[0], 'question': q[1]} for q in questions]
        return jsonify({'success': True, 'questions': formatted})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# 4. IMPROVE RESUME (SUGGESTIONS)
# ─────────────────────────────────────────────
@app.route('/api/improve', methods=['POST'])
def improve_resume():
    data = request.get_json(force=True)
    session_id = data.get('session_id', '').strip()
    improvement_areas = data.get('improvement_areas', [])
    target_role = data.get('target_role', '')

    if not session_id:
        return jsonify({'error': 'session_id is required.'}), 400
    if session_id not in agent_sessions:
        return jsonify({'error': 'Session not found. Please analyse a resume first.'}), 404

    try:
        agent = get_agent(session_id)
        improvements = agent.improve_resume(
            improvement_areas=improvement_areas,
            target_role=target_role
        )
        return jsonify({'success': True, 'improvements': improvements})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# 5. GET IMPROVED RESUME (REWRITE)
# ─────────────────────────────────────────────
@app.route('/api/rewrite', methods=['POST'])
def get_improved_resume():
    data = request.get_json(force=True)
    session_id = data.get('session_id', '').strip()
    target_role = data.get('target_role', '')
    highlight_skills = data.get('highlight_skills', '')

    if not session_id:
        return jsonify({'error': 'session_id is required.'}), 400
    if session_id not in agent_sessions:
        return jsonify({'error': 'Session not found. Please analyse a resume first.'}), 404

    try:
        agent = get_agent(session_id)
        improved = agent.get_improved_resume(
            target_role=target_role,
            highlight_skills=highlight_skills
        )
        return jsonify({'success': True, 'improved_resume': improved})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# 6. DOWNLOAD IMPROVED RESUME
# ─────────────────────────────────────────────
@app.route('/api/download-resume', methods=['GET'])
def download_resume():
    session_id = request.args.get('session_id', '').strip()
    if not session_id or session_id not in agent_sessions:
        return jsonify({'error': 'Session not found.'}), 404

    agent = agent_sessions[session_id]
    if not hasattr(agent, 'improved_resume_path') or not os.path.exists(agent.improved_resume_path):
        return jsonify({'error': 'No improved resume available. Call /api/rewrite first.'}), 404

    return send_file(
        agent.improved_resume_path,
        as_attachment=True,
        download_name='improved_resume.txt',
        mimetype='text/plain'
    )


# ─────────────────────────────────────────────
# 7. CLEANUP SESSION
# ─────────────────────────────────────────────
@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    data = request.get_json(force=True)
    session_id = data.get('session_id', '').strip()
    if session_id in agent_sessions:
        agent_sessions[session_id].cleanup()
        del agent_sessions[session_id]
    return jsonify({'success': True, 'message': 'Session cleaned up.'})


if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found in .env file!")
    else:
        print(f"Groq API key loaded: {GROQ_API_KEY[:8]}...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)