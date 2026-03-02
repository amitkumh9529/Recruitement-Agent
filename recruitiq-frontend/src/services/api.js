/**
 * RecruitIQ API Service
 * All communication with the Flask backend lives here.
 */

import axios from 'axios'

// When running via Vite dev server the proxy in vite.config.js
// forwards /api → http://localhost:5000/api  (no CORS issues)
const BASE = '/api'

const client = axios.create({ baseURL: BASE })

// ── 1. Analyse Resume ─────────────────────────────────────
export async function analyzeResume({ sessionId, cutoffScore, resumeFile, jdFile, roleRequirements }) {
  const fd = new FormData()
  fd.append('session_id', sessionId)
  fd.append('cutoff_score', cutoffScore ?? 75)
  fd.append('resume', resumeFile)
  if (jdFile) fd.append('job_description', jdFile)
  if (roleRequirements?.trim()) fd.append('role_requirements', roleRequirements.trim())

  const { data } = await client.post('/analyze', fd)
  if (data.error) throw new Error(data.error)
  return data.results
}

// ── 2. Q&A ────────────────────────────────────────────────
export async function askQuestion({ sessionId, question }) {
  const { data } = await client.post('/ask', { session_id: sessionId, question })
  if (data.error) throw new Error(data.error)
  return data.answer
}

// ── 3. Interview Questions ────────────────────────────────
export async function generateInterviewQuestions({ sessionId, questionTypes, difficulty, numQuestions }) {
  const { data } = await client.post('/interview-questions', {
    session_id: sessionId,
    question_types: questionTypes,
    difficulty,
    num_questions: numQuestions,
  })
  if (data.error) throw new Error(data.error)
  return data.questions
}

// ── 4. Resume Improvement Suggestions ────────────────────
export async function getImprovements({ sessionId, improvementAreas, targetRole }) {
  const { data } = await client.post('/improve', {
    session_id: sessionId,
    improvement_areas: improvementAreas,
    target_role: targetRole,
  })
  if (data.error) throw new Error(data.error)
  return data.improvements
}

// ── 5. Rewrite Resume ─────────────────────────────────────
export async function rewriteResume({ sessionId, targetRole, highlightSkills }) {
  const { data } = await client.post('/rewrite', {
    session_id: sessionId,
    target_role: targetRole,
    highlight_skills: highlightSkills,
  })
  if (data.error) throw new Error(data.error)
  return data.improved_resume
}

// ── 6. Download Resume ────────────────────────────────────
export function getDownloadUrl(sessionId) {
  return `${BASE}/download-resume?session_id=${sessionId}`
}

// ── 7. Cleanup Session ────────────────────────────────────
export async function cleanupSession(sessionId) {
  try {
    await client.post('/cleanup', { session_id: sessionId })
  } catch (_) {
    // best-effort
  }
}
