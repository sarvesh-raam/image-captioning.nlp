import React, { useState, useEffect, useRef } from 'react'
import './App.css'
import AttentionOverlay from './AttentionOverlay.jsx'

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8001").replace(/\/$/, "");
const MAX_UPLOAD_MB = 10;
const EXAMPLES = ["/examples/example1.jpg", "/examples/example2.jpg", "/examples/example3.jpg"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [activeWord, setActiveWord] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [statusText, setStatusText] = useState("")
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  // The free Hugging Face Space sleeps when idle; ping it on page load so it starts waking up
  useEffect(() => {
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);

  // Release the previous preview's memory when the image changes
  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleImageSelection(files[0]);
    }
  };

  const handleImageSelection = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError("Please upload a valid image file.");
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`Image is too large (max ${MAX_UPLOAD_MB} MB).`);
      return;
    }
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setActiveWord(null);
    setError(null);
  };

  const loadExample = async (url) => {
    try {
      const blob = await (await fetch(url)).blob();
      handleImageSelection(new File([blob], url.split('/').pop(), { type: blob.type || 'image/jpeg' }));
    } catch {
      setError("Could not load the example image.");
    }
  };

  const generateCaption = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setResult(null);
    setActiveWord(null);
    setError(null);
    setStatusText("Our AI is looking at your photo...");

    const formData = new FormData();
    formData.append('file', selectedImage);

    // Retry while the server is cold-starting (network error or 502/503 from the Space proxy)
    const MAX_ATTEMPTS = 8;
    try {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        let response;
        try {
          response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
        } catch {
          response = null;
        }

        if (response && response.ok) {
          setResult(await response.json());
          return;
        }

        const retryable = !response || [502, 503, 504].includes(response.status);
        if (!retryable || attempt === MAX_ATTEMPTS) {
          let message = "Failed to generate caption. The server may be offline.";
          if (response) {
            const data = await response.json().catch(() => null);
            message = data?.detail || data?.error || `Server error (${response.status}).`;
          }
          throw new Error(message);
        }

        setStatusText("Waking up the server (this can take a minute on first use)...");
        await sleep(8000);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
      setStatusText("");
    }
  };

  const attention = result?.attention;
  const captions = result?.captions || [];
  const activeWeights = attention && activeWord !== null ? attention.maps[activeWord] : null;

  return (
    <div className="App">
      <nav className="navbar">
        <div className="logo-dot"></div>
        <h1>Magic Image Captioner</h1>
      </nav>

      <div className="container">
        <header className="hero-section">
          <h2>Discover what's in<br/>your photos</h2>
          <p>Simply upload any image and our AI will magically describe it for you in plain English.</p>
        </header>

        <section className="upload-card">
          <div
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current?.click(); } }}
            role="button"
            tabIndex={0}
            aria-label="Upload an image"
          >
            <input
              type="file"
              ref={inputRef}
              className="hidden"
              accept="image/*"
              onChange={(e) => { handleImageSelection(e.target.files[0]); e.target.value = ""; }}
            />

            {previewUrl ? (
              <div className="preview-container">
                <AttentionOverlay
                  src={previewUrl}
                  grid={attention?.grid}
                  crop={attention?.crop}
                  weights={activeWeights}
                />
                <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Click or drop to choose a different photo</p>
              </div>
            ) : (
              <>
                <div className="upload-icon">
                   <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                     <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                   </svg>
                </div>
                <p style={{ fontSize: '1.2rem', fontWeight: '600' }}>Drop a photo here</p>
                <p style={{ color: 'var(--text-muted)' }}>or click to browse from your device</p>
              </>
            )}
          </div>

          <div className="examples">
            <span>Or try an example:</span>
            {EXAMPLES.map((url, i) => (
              <button key={url} className="example-thumb" onClick={() => loadExample(url)} aria-label={`Example image ${i + 1}`}>
                <img src={url} alt="" />
              </button>
            ))}
          </div>

          {selectedImage && (
            <button
              className="btn"
              onClick={generateCaption}
              disabled={isLoading}
            >
              {isLoading ? 'Thinking...' : 'Generate Magic Caption'}
            </button>
          )}

          {isLoading && (
            <div className="loader-container">
              <div className="spinner"></div>
              <span style={{ marginLeft: '12px', color: 'var(--accent)' }}>{statusText}</span>
            </div>
          )}

          <div className={`result-section ${result ? 'visible' : ''}`} aria-live="polite">
            <p className="result-label">Here's what we see:</p>
            <p className="caption-text">{result?.caption}</p>

            {result?.warnings?.map((w) => (
              <p key={w} className="warning">{w}</p>
            ))}

            {attention?.words?.length > 0 && (
              <>
                <p className="result-label" style={{ marginTop: '24px' }}>Where the model looked (hover or tap a word)</p>
                <div className="word-chips">
                  {attention.words.map((word, i) => (
                    <button
                      key={i}
                      className={`word-chip ${activeWord === i ? 'active' : ''}`}
                      onMouseEnter={() => setActiveWord(i)}
                      onMouseLeave={() => setActiveWord(null)}
                      onFocus={() => setActiveWord(i)}
                      onBlur={() => setActiveWord(null)}
                      onClick={() => setActiveWord(i)}
                      title={`Probability ${(attention.word_probs[i] * 100).toFixed(0)}%`}
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </>
            )}

            {captions.length > 1 && (
              <>
                <p className="result-label" style={{ marginTop: '24px' }}>Other candidates</p>
                <ul className="alt-captions">
                  {captions.slice(1).map((c) => (
                    <li key={c.text}>{c.text}</li>
                  ))}
                </ul>
              </>
            )}

            {typeof result?.confidence === 'number' && (
              <p className="result-meta">
                Confidence {(result.confidence * 100).toFixed(0)}% · generated in {(result.inference_ms / 1000).toFixed(1)}s
              </p>
            )}
          </div>

          {error && (
            <div role="alert" style={{ marginTop: '20px', color: '#ef4444' }}>
              {error}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
