import { useState, useRef, useEffect } from 'react';
import './index.css';

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Xử lý khi người dùng upload ảnh
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target?.result as string);
      setPrediction(null);
      setProcessedImage(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!image) return;
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setPrediction(data.prediction);
        if (data.processed_image) {
          setProcessedImage(data.processed_image);
        }
      } else {
        setError(data.error || 'Có lỗi xảy ra khi dự đoán.');
      }
    } catch (err) {
      console.error(err);
      setError('Không thể kết nối đến server backend. Hãy chắc chắn app.py đang chạy.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    setProcessedImage(null);
    setPrediction(null);
    setError(null);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  // Khởi tạo canvas màu đen
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && image) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const img = new Image();
        img.onload = () => {
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          const imgAspect = img.width / img.height;
          let drawWidth = canvas.width;
          let drawHeight = canvas.height;
          if (imgAspect > 1) {
            drawHeight = canvas.width / imgAspect;
          } else {
            drawWidth = canvas.height * imgAspect;
          }
          const x = (canvas.width - drawWidth) / 2;
          const y = (canvas.height - drawHeight) / 2;
          ctx.drawImage(img, x, y, drawWidth, drawHeight);
        };
        img.src = image;
      }
    }
  }, [image]);

  return (
    <div className="app-container">
      <header>
        <h1>AI Digit Recognizer</h1>
        <p className="subtitle">Upload ảnh số của bạn, mô hình tự code bằng NumPy sẽ dự đoán!</p>
      </header>

      <main className="main-card">
        {!image ? (
          <div className="upload-area">
            <div className="upload-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <h3>Click để tải ảnh lên</h3>
            <p style={{marginTop: '0.5rem', color: 'var(--text-muted)'}}>Hỗ trợ định dạng JPG, PNG</p>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
          </div>
        ) : (
          <div className="preview-container">
            <h3 style={{color: 'var(--text-muted)'}}>Ảnh tải lên:</h3>
            <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Bản Gốc</span>
                <div className="canvas-wrapper">
                  <canvas 
                    ref={canvasRef} 
                    width={280} 
                    height={280} 
                  />
                </div>
              </div>
              
              {processedImage && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Mô hình AI nhìn thấy</span>
                  <div className="canvas-wrapper">
                    <img 
                      src={processedImage} 
                      alt="Processed" 
                      width={280} 
                      height={280} 
                      style={{ imageRendering: 'pixelated', borderRadius: '8px', backgroundColor: '#000' }}
                    />
                  </div>
                </div>
              )}
            </div>
            
            <div className="actions">
              <button className="secondary" onClick={handleClear} disabled={loading}>
                Tải ảnh khác
              </button>
              <button onClick={handlePredict} disabled={loading}>
                {loading ? <div className="loader"></div> : 'Nhận diện Số'}
              </button>
            </div>

            {error && (
              <div style={{color: '#ef4444', marginTop: '1rem', textAlign: 'center'}}>
                {error}
              </div>
            )}

            {prediction !== null && (
              <div className="result-box">
                <div className="result-title">Kết quả dự đoán</div>
                <div className="result-number">{prediction}</div>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
