import React, { useState, useCallback, useEffect } from 'react'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [caption, setCaption] = useState("")
  const [displayedCaption, setDisplayedCaption] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)

  // Typewriter effect for the caption
  useEffect(() => {
    if (caption && !isLoading) {
      setDisplayedCaption(caption);
    }
  }, [caption, isLoading]);

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
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setCaption("");
      setDisplayedCaption("");
      setError(null);
    } else {
      setError("Please upload a valid image file.");
    }
  };

  const generateCaption = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setCaption("");
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to generate caption. Is the backend running?');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      // Nuclear fix for string artifacts
      const cleanCaption = (data.caption || "")
        .replace(/undefined/g, '')
        .replace(/\.+$/, '.') // Ensure only one dot at the end
        .trim();

      setCaption(cleanCaption);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="logo-dot"></div>
        <h1>Magic Image Captioner</h1>
      </nav>

      <div className="container">
        <header className="hero-section">
          <h2>Discover what's in<br/>your photos ✨</h2>
          <p>Simply upload any image and our AI will magically describe it for you in plain English.</p>
        </header>

        <section className="upload-card">
          <div 
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => document.getElementById('imageInput').click()}
          >
            <input 
              type="file" 
              id="imageInput" 
              className="hidden" 
              accept="image/*"
              onChange={(e) => handleImageSelection(e.target.files[0])}
            />
            
            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Switch file for analysis</p>
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

          {selectedImage && (
            <button 
              className="btn" 
              onClick={generateCaption} 
              disabled={isLoading}
            >
              {isLoading ? 'Thinking...' : 'Generate Magic Caption ✨'}
            </button>
          )}

          {isLoading && (
            <div className="loader-container">
              <div className="spinner"></div>
              <span style={{ marginLeft: '12px', color: 'var(--accent)' }}>Our AI is looking at your photo...</span>
            </div>
          )}

          <div className={`result-section ${displayedCaption ? 'visible' : ''}`}>
            <p className="result-label">Here's what we see:</p>
            <p className="caption-text">{displayedCaption}</p>
          </div>

          {error && (
            <div style={{ marginTop: '20px', color: '#ef4444' }}>
              System Error: {error}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
