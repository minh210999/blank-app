import React, { useState } from 'react'

export default function ArtAncestry() {
  const [uploadedImage, setUploadedImage] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [predecessorImage, setPredecessorImage] = useState(null)
  const [apiKey, setApiKey] = useState(process.env.REACT_APP_OPENAI_KEY || '')

  const SYSTEM_PROMPT = `You are an expert Art Historian specializing in Comparative Historical Analysis. 
Your task is to analyze an uploaded artwork and compare it to its specific historical predecessor from the same country or region.

### THE PROTOCOL (5-STEP FRAMEWORK)
1. **Gather Evidence (Observation):** Observe the raw visual facts (Medium, Line, Color, Subject). Identify the specific *predecessor work*.
2. **Group Categories:** Organize observations into themes.
3. **Identify Importance:** Select the ONE most significant category.
4. **Compare (The Chart):** Contrast the two works within that category.
5. **Synthesize (The Argument):** Write a concise analytical statement.

### RESPONSE FORMAT (JSON ONLY)
Return valid JSON with no markdown:
{
  "current_work": {"title": "Title", "artist": "Artist", "date": "Year", "style": "Style", "description": "Brief summary"},
  "predecessor_work": {"title": "Title", "artist": "Artist", "date": "Period", "style": "Style"},
  "analysis": {
    "step_3_category": "Primary Theme",
    "step_4_comparison": {"similarity": "Key element kept", "difference": "Innovation or departure"},
    "step_5_synthesis": "3-4 sentence comparative essay"
  }
}`

  const analyzeArt = async (imageBase64) => {
    if (!apiKey) {
      throw new Error('Please enter your OpenAI API key.')
    }

    const url = 'https://api.openai.com/v1/chat/completions'
    const imageDataUrl = `data:image/png;base64,${imageBase64}`

    const payload = {
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analyze this artwork following the 5-step framework. Return valid JSON with the structure shown.'
            },
            { type: 'image_url', image_url: { url: imageDataUrl } }
          ]
        }
      ],
      temperature: 0.2,
      max_tokens: 1200
    }

    let lastError = null
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })

        if (!response.ok) {
          if (response.status === 429 && attempt < 2) {
            const waitTime = Math.pow(2, attempt)
            await new Promise(resolve => setTimeout(resolve, waitTime * 1000))
            continue
          }
          throw new Error(`OpenAI API error: ${response.status}`)
        }

        const data = await response.json()
        let assistantText = data.choices?.[0]?.message?.content

        if (!assistantText) throw new Error('No response from OpenAI')

        // Parse JSON
        let result
        try {
          result = JSON.parse(assistantText)
        } catch {
          const startIdx = assistantText.indexOf('{')
          const endIdx = assistantText.lastIndexOf('}')
          if (startIdx >= 0 && endIdx > startIdx) {
            result = JSON.parse(assistantText.substring(startIdx, endIdx + 1))
          } else {
            throw new Error('Could not parse JSON response')
          }
        }

        return result
      } catch (err) {
        lastError = err
      }
    }
    throw lastError || new Error('Analysis failed after retries')
  }

  const getWikimediaImage = async (query) => {
    if (!query) return null
    try {
      const response = await fetch('https://commons.wikimedia.org/w/api.php?' +
        new URLSearchParams({
          action: 'query',
          format: 'json',
          generator: 'search',
          gsrnamespace: 6,
          gsrsearch: query,
          gsrlimit: 10,
          prop: 'imageinfo',
          iiprop: 'url|size|mime|thumburl',
          iiurlwidth: 2048,
          origin: '*'
        })
      )
      const data = await response.json()
      const pages = data.query?.pages || {}
      
      for (const pageId in pages) {
        const imageInfo = pages[pageId].imageinfo?.[0]
        if (imageInfo?.thumburl) return imageInfo.thumburl
        if (imageInfo?.url) return imageInfo.url
      }
    } catch (err) {
      console.error('Wikimedia search error:', err)
    }
    return null
  }

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const reader = new FileReader()
      reader.onload = async (event) => {
        const base64 = event.target?.result?.split(',')[1]
        if (!base64) throw new Error('Failed to encode image')

        setUploadedImage(event.target?.result)
        
        const analysisResult = await analyzeArt(base64)
        
        // Extract data
        const current = analysisResult.current_work || {}
        const predecessor = analysisResult.predecessor_work || {}
        const analysisData = analysisResult.analysis || {}

        const analysisText = [
          analysisData.step_3_category ? `**Primary Theme:** ${analysisData.step_3_category}` : '',
          analysisData.step_4_comparison?.similarity ? `**Similarity:** ${analysisData.step_4_comparison.similarity}` : '',
          analysisData.step_4_comparison?.difference ? `**Difference:** ${analysisData.step_4_comparison.difference}` : '',
          analysisData.step_5_synthesis ? `**Analysis:** ${analysisData.step_5_synthesis}` : ''
        ].filter(Boolean).join('\n\n')

        setAnalysis({
          currentTitle: current.title || 'Unknown',
          currentDate: current.date || '',
          currentStyle: current.style || '',
          predecessorTitle: predecessor.title || '',
          predecessorArtist: predecessor.artist || '',
          predecessorDate: predecessor.date || '',
          predecessorStyle: predecessor.style || '',
          analysisText
        })

        // Fetch predecessor image
        if (predecessor.title && predecessor.artist) {
          const imgUrl = await getWikimediaImage(`${predecessor.title} ${predecessor.artist}`)
          setPredecessorImage(imgUrl)
        }

        setLoading(false)
      }
      reader.readAsDataURL(file)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui' }}>
      <h1>Art Ancestry — Comparative Historical Analysis</h1>
      <p>Upload a JPG or PNG image of an artwork to analyze its probable predecessor.</p>

      <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
          OpenAI API Key:
        </label>
        <input
          type="password"
          placeholder="sk-..."
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid #ccc',
            fontSize: '14px',
            boxSizing: 'border-box'
          }}
        />
        <p style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
          Your API key is only used in this browser session and is never stored or shared.
        </p>
      </div>

      <input
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleImageUpload}
        disabled={loading || !apiKey}
        style={{ marginBottom: '20px' }}
      />

      {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}
      {loading && <div style={{ marginBottom: '20px' }}>Analyzing provenance...</div>}

      {analysis && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
            <div>
              <h2>Your Upload</h2>
              {uploadedImage && <img src={uploadedImage} alt="Uploaded" style={{ maxWidth: '100%' }} />}
              <p>
                <strong>Identified:</strong> {analysis.currentTitle}
                {analysis.currentStyle && ` (${analysis.currentStyle})`}
                {analysis.currentDate && ` — ${analysis.currentDate}`}
              </p>
            </div>

            <div>
              <h2>The Predecessor</h2>
              {predecessorImage && <img src={predecessorImage} alt="Predecessor" style={{ maxWidth: '100%' }} />}
              {!predecessorImage && <p style={{ color: '#999' }}>Predecessor image not found</p>}
              <p>
                <strong>Predecessor:</strong> {analysis.predecessorTitle} — {analysis.predecessorArtist}
                {analysis.predecessorStyle && ` (${analysis.predecessorStyle})`}
                {analysis.predecessorDate && ` — ${analysis.predecessorDate}`}
              </p>
            </div>
          </div>

          <hr />
          <h2>Comparative Historical Analysis</h2>
          <div style={{ whiteSpace: 'pre-wrap' }}>{analysis.analysisText}</div>
        </>
      )}
    </div>
  )
}
