import { useState, useCallback } from 'react'
import { analyzeResume } from '../services/api'

export function useAnalysis(sessionId, showAlert) {
  const [results, setResults]   = useState(null)
  const [analysed, setAnalysed] = useState(false)
  const [loading, setLoading]   = useState(false)

  const runAnalysis = useCallback(async (formData) => {
    if (!formData.resumeFile) {
      showAlert('Please upload a resume file.')
      return
    }

    setLoading(true)
    try {
      const res = await analyzeResume({ ...formData, sessionId })
      console.log('Analysis result:', res)   // ← add this to debug
      if (res) {
        setResults(res)
        setAnalysed(true)
        showAlert('Analysis complete!', 'success')
      } else {
        showAlert('No results returned from server.')
      }
    } catch (err) {
      console.error('Analysis error:', err)
      showAlert(err.message || 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }, [sessionId, showAlert])

  return { results, analysed, loading, runAnalysis }
}