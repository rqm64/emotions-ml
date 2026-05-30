import { useEffect, useRef, useState } from 'react'
import './App.css'

const CAPTURE_INTERVAL_MS = 1800
const TOP_EMOJIS = {
  angry: '😠',
  disgust: '🤢',
  fear: '😨',
  happy: '😄',
  neutral: '🙂',
  sad: '😢',
  surprise: '😲',
}

function emojiForEmotion(emotion) {
  return TOP_EMOJIS[emotion?.toLowerCase?.()] ?? '🙂'
}

export default function App() {
  const [stream, setStream] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [sending, setSending] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const inFlightRef = useRef(false)

  const start = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      })
      setStream(mediaStream)
    } catch {}
  }

  useEffect(() => {
    const video = videoRef.current
    if (!video || !stream) return

    video.srcObject = stream
    return () => {
      stream.getTracks().forEach((track) => track.stop())
    }
  }, [stream])

  useEffect(() => {
    if (!stream) return

    let timerId = null
    let disposed = false

    const sendFrame = async () => {
      if (disposed || inFlightRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) return

      const width = video.videoWidth
      const height = video.videoHeight
      if (!width || !height) return

      const ctx = canvas.getContext('2d')
      canvas.width = width
      canvas.height = height
      ctx.drawImage(video, 0, 0, width, height)

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85))
      if (!blob) return

      const formData = new FormData()
      formData.append('image', blob, 'frame.jpg')

      try {
        inFlightRef.current = true
        setSending(true)
        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        })
        if (!response.ok) return
        const data = await response.json().catch(() => null)
        if (!data || disposed) return
        setPrediction(data)
      } finally {
        inFlightRef.current = false
        if (!disposed) {
          setSending(false)
        }
      }
    }

    sendFrame()
    timerId = window.setInterval(sendFrame, CAPTURE_INTERVAL_MS)

    return () => {
      disposed = true
      if (timerId) window.clearInterval(timerId)
    }
  }, [stream])

  if (stream) {
    const mainEmoji = emojiForEmotion(prediction?.emotion)
    const topPredictions = prediction?.top_predictions ?? []

    return (
      <div className="camera-stage">
        <video
          ref={videoRef}
          className="camera"
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="hidden-canvas" />

        <div className="overlay">
          <div className="main-emoji">{mainEmoji}</div>
          <div className="status-card">
            <p className="status-title">
              Эмоция: <strong>{prediction?.emotion ?? '...'}</strong>
            </p>
            <p className="status-subtitle">
              Вероятность: {prediction?.probability != null ? `${(prediction.probability * 100).toFixed(1)}%` : '...'}
            </p>
            {topPredictions.length > 0 && (
              <div className="top-list">
                {topPredictions.map((item) => (
                  <span key={item.emotion} className="top-item">
                    {emojiForEmotion(item.emotion)} {item.emotion} {(item.probability * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <main className="start-screen">
      <button type="button" className="start-btn" onClick={start}>
        Старт
      </button>
    </main>
  )
}
