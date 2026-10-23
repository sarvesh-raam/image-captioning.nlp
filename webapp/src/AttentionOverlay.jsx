import React, { useEffect, useRef, useState } from 'react'

// Draws one word's attention grid as a heatmap over the photo. The grid covers either the whole
// image (ResNet stretches it to a square) or a centred square crop (CLIP), depending on `crop`.
function AttentionOverlay({ src, grid, crop, weights }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)
  const [size, setSize] = useState(null)

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const update = () => setSize({ w: img.clientWidth, h: img.clientHeight })
    const observer = new ResizeObserver(update)
    observer.observe(img)
    return () => observer.disconnect()
  }, [src])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!weights) return

    // Render the coarse grid into a tiny canvas, then let the browser upscale it smoothly
    const small = document.createElement('canvas')
    small.width = grid
    small.height = grid
    const sctx = small.getContext('2d')
    const max = Math.max(...weights) || 1
    const pixels = sctx.createImageData(grid, grid)
    weights.forEach((w, i) => {
      const v = w / max
      pixels.data.set([255, Math.round(80 + 140 * v), 40, Math.round(200 * v)], i * 4)
    })
    sctx.putImageData(pixels, 0, 0)

    let x = 0, y = 0, w = size.w, h = size.h
    if (crop === 'center') {
      const side = Math.min(size.w, size.h)
      x = (size.w - side) / 2
      y = (size.h - side) / 2
      w = h = side
    }
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(small, x, y, w, h)
  }, [weights, grid, crop, size])

  return (
    <div className="attention-wrap">
      <img ref={imgRef} src={src} alt="Uploaded" className="preview-image" />
      {size && (
        <canvas
          ref={canvasRef}
          width={size.w}
          height={size.h}
          className={`attention-canvas ${weights ? 'visible' : ''}`}
          aria-hidden="true"
        />
      )}
    </div>
  )
}

export default AttentionOverlay
