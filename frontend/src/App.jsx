import React, { useState, useRef, forwardRef } from 'react';
import { Search, Upload, Clock, Play, Loader2, ShieldAlert, Eye, Activity, Database, Server, ChevronRight } from 'lucide-react';

const API_URL = "http://localhost:8005";

// --- Monolith Dark Theme CSS (Refined) ---
const styles = `
  :root {
    /* Base Palette - Monolith */
    --bg-root: #191919;
    --bg-surface: #616161;
    --bg-panel: #7b7b7b;
    --bg-element: #2f2f2f;
    --bg-element-hover: #5d5d5d;
    
    --text-primary: #ffffff;
    --text-secondary: #a3a3a3; /* Neutral 400 */
    --text-tertiary: #525252;  /* Neutral 600 */
    
    /* Accents - Titanium & Silver */
    --accent-primary: #ffffff;
    --accent-secondary: #e5e5e5;
    --accent-gradient: linear-gradient(135deg, #ffffff 0%, #737373 100%);
    --accent-glow: 0 0 25px rgb(57, 57, 57);
    
    /* Borders & Depth */
    --border: #4f4f4f;
    --border-hover: #545454;
    --shadow-sm: 0 1px 2px 0 rgba(44, 44, 44, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(43, 43, 43, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
    
    /* Status Colors (Muted) */
    --success: #d4d4d4;
    --warning: #a3a3a3;
    --danger: #ef4444;
    
    /* Metrics */
    --radius-xl: 12px;
    --radius-lg: 8px;
    --radius-md: 6px;
    --nav-height: 64px;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background-color: radial-gradient(circle at 50% 0%, #343434 0%, var(--bg-root) 60%);
    background-image: radial-gradient(circle at 50% 0%, #343434 0%, var(--bg-root) 60%);
    background-attachment: radial-gradient(circle at 50% 0%, #343434 0%, var(--bg-root) 60%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  .app-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* --- Header --- */
  .app-header {
    height: var(--nav-height);
    background-color: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    align-items: center;
  }

  .header-content {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .logo-box {
    width: 32px;
    height: 32px;
    background: #fff;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 12px rgb(71, 71, 71);
  }

  .logo-icon {
    color: #000; 
  }

  .app-title {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
  }

  .app-tagline {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    padding-left: 1rem;
    border-left: 1px solid var(--border);
    display: none;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  @media (min-width: 640px) { .app-tagline { display: block; } }

  /* --- Layout --- */
  .main-content {
    flex: 1;
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  @media (min-width: 1024px) {
    .main-content {
      grid-template-columns: 2.5fr 1fr;
      gap: 3rem;
    }
  }

  /* --- Panels --- */
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-sm);
  }

  /* --- Video Player --- */
  .player-container {
    padding: 0;
    overflow: hidden;
    position: relative;
    box-shadow: var(--shadow-lg);
    background: #000;
    border-radius: var(--radius-xl);
  }

  .video-frame {
    position: relative;
    aspect-ratio: 16 / 9;
    background: #000;
    width: 100%;
    height: 100%;
  }

  .video-element {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-tertiary);
    background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-root) 100%);
  }

  /* --- Search Bar --- */
  .search-container {
    position: relative;
    margin-top: 2rem;
  }

  .search-input {
    width: 100%;
    background: var(--bg-element);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 1rem 1rem 1rem 3rem; /* Adjusted padding */
    border-radius: var(--radius-lg);
    font-size: 0.95rem;
    transition: all 0.2s ease;
    height: 3.5rem; /* Fixed height for consistency */
  }

  .search-input::placeholder {
    color: var(--text-tertiary);
  }

  .search-input:focus {
    outline: none;
    background: var(--bg-element-hover);
    border-color: var(--text-secondary);
    box-shadow: 0 0 0 2px rgba(255,255,255,0.05);
  }

  .search-icon {
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-tertiary);
  }

  .search-btn {
    position: absolute;
    right: 0.5rem;
    top: 0.5rem;
    bottom: 0.5rem;
    padding: 0 1.25rem;
    background: var(--accent-primary);
    color: #000;
    border: none;
    border-radius: var(--radius-md);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.9rem;
  }
  .search-btn:hover:not(:disabled) {
    background: #e5e5e5;
    box-shadow: 0 0 10px rgba(255,255,255,0.2);
  }
  .search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* --- Sidebar --- */
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .card-content {
    padding: 1.5rem;
  }

  .card-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgb(0, 0, 0);
  }

  .upload-area {
    border: 1px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 2.5rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    background: var(--bg-root);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .upload-area:hover {
    border-color: var(--text-secondary);
    background: var(--bg-element);
  }
  
  .file-name {
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    font-family: monospace;
    word-break: break-all;
    background: var(--bg-element);
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    display: inline-block;
    max-width: 100%;
  }

  .btn-primary {
    width: 100%;
    margin-top: 1rem;
    padding: 0.875rem;
    background: var(--accent-primary);
    color: #000;
    border: none;
    border-radius: var(--radius-lg);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9rem;
  }
  .btn-primary:hover:not(:disabled) {
    background: #e5e5e5;
  }
  .btn-primary:disabled { opacity: 0.6; cursor: wait; background: var(--text-tertiary); }

  /* Status List */
  .status-list {
    display: flex;
    flex-direction: column;
  }
  
  .status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.875rem 0; /* Reduced padding */
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
  }
  .status-item:last-child { border-bottom: none; }
  .status-label { rgb(0, 0, 0); display: flex; align-items: center; gap: 0.75rem; }
  .status-val { font-family: monospace; color: rgb(0, 0, 0); font-weight: 500; font-size: 0.8rem; }

  /* --- Results Grid --- */
  .results-area {
    margin-top: 2.5rem;
  }
  
  .results-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    color: var(--text-primary);
  }
  
  .count-badge {
    background: var(--bg-element);
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    border: 1px solid var(--border);
    font-weight: 500;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1.5rem;
  }

  .result-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s ease-out;
    opacity: 0;
    animation: fadeIn 0.4s ease-out forwards;
    display: flex;
    flex-direction: column;
  }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .result-card:hover {
    transform: translateY(-4px);
    border-color: var(--text-secondary);
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.8);
  }

  .card-thumb {
    aspect-ratio: 16/9;
    background: #000;
    position: relative;
    overflow: hidden;
  }

  .thumb-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.7;
    transition: transform 0.4s, opacity 0.3s;
    filter: grayscale(0.2);
  }
  .result-card:hover .thumb-img { opacity: 1; transform: scale(1.03); filter: grayscale(0); }

  .time-badge {
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(0,0,0,0.8);
    backdrop-filter: blur(4px);
    color: #fff;
    padding: 3px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: monospace;
    display: flex;
    align-items: center;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.1);
  }

  .card-body { padding: 1rem; }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .score-badge {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    background: var(--bg-element);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }
  .score-badge.high { border-color: var(--text-primary); background: rgba(255,255,255,0.1); }
  .score-badge.med { border-color: var(--text-tertiary); color: var(--text-secondary); }

  .metric-detail {
    font-size: 0.75rem;
    color: rgb(255, 255, 255);
    font-family: monospace;
    display: flex;
    justify-content: space-between;
  }

  /* Abstain UI */
  .abstain-box {
    background: rgba(20, 0, 0, 0.5);
    border: 1px solid rgba(239, 68, 68, 0.2); /* Red tint border */
    padding: 1.5rem;
    border-radius: var(--radius-xl);
    display: flex;
    gap: 1.25rem;
    align-items: flex-start;
    animation: fadeIn 0.4s ease-out;
  }
  
  .abstain-text h4 { margin: 0 0 0.5rem 0; color: #fff; font-size: 0.95rem; }
  .abstain-text p { margin: 0; color: var(--text-secondary); line-height: 1.5; font-size: 0.85rem; }
`;

// --- Video Component ---
const VideoPlayer = forwardRef(({ videoUrl }, ref) => {
  if (!videoUrl) return (
    <div className="panel player-container">
      <div className="video-frame">
        <div className="placeholder">
          <Upload size={40} style={{ opacity: 0.5, marginBottom: '1rem', color: '#fff' }} />
          <p style={{fontSize: '0.9rem', fontWeight: 500, color: '#fff'}}>Upload footage to initialize</p>
        </div>
      </div>
    </div>
  );
  return (
    <div className="panel player-container">
      <div className="video-frame">
        <video ref={ref} src={videoUrl} controls className="video-element" />
      </div>
    </div>
  );
});
VideoPlayer.displayName = 'VideoPlayer';

export default function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const videoRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) setVideoFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", videoFile);

    try {
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("Upload failed");
      const data = await response.json();
      setVideoUrl(`${API_URL}${data.video_url}`);
      
      if (data.status === "abstained_no_motion") {
        alert("System Abstained: No significant motion events detected.");
      } else {
        alert(`Verification Complete! Index contains ${data.events_verified} verified events.`);
      }
    } catch (error) {
      console.error(error);
      alert("Error uploading. Check backend connection.");
    } finally {
      setUploading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery) return;
    setStatus("searching");
    try {
      const response = await fetch(`${API_URL}/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      
      if (data.status === "abstained" || (data.results && data.results.length === 0)) {
        setResults([]);
        setStatus("abstained");
      } else {
        setResults(data.results || []);
        setStatus("success");
      }
    } catch (error) {
      console.error(error);
      setStatus("error");
    }
  };

  const jumpToTimestamp = (timestamp) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.play();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div className="app-container">
      <style>{styles}</style>
      
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-box">
            <Eye className="logo-icon" size={18} />
          </div>
          <div>
            <div className="app-title">Semantic Eye</div>
          </div>
          <span className="app-tagline">Event Analysis Engine</span>
        </div>
      </header>

      <main className="main-content">
        
        {/* Left Column */}
        <div>
          <VideoPlayer ref={videoRef} videoUrl={videoUrl} />
          
          <form onSubmit={handleSearch} className="search-container">
            <Search className="search-icon" size={18} />
            <input
              type="text"
              placeholder="Search verified events..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <button type="submit" disabled={status === "searching"} className="search-btn">
              {status === "searching" ? <Loader2 className="animate-spin" size={18} /> : "Search"}
            </button>
          </form>

          {/* Results Area */}
          <div className="results-area">
            {status === "abstained" && (
              <div className="abstain-box">
                <ShieldAlert size={24} style={{color: 'var(--danger)', opacity: 0.9, flexShrink: 0}} />
                <div className="abstain-text">
                  <h4>System Abstained</h4>
                  <p>Insufficient evidence found. Candidates rejected due to low semantic confidence or lack of verified motion.</p>
                </div>
              </div>
            )}

            {status === "success" && results.length > 0 && (
              <>
                <div className="results-header">
                  <span>Matches Found</span>
                  <span className="count-badge">{results.length}</span>
                </div>
                
                <div className="grid">
                  {results.map((result, idx) => (
                    <div 
                      key={idx} 
                      onClick={() => jumpToTimestamp(result.timestamp)} 
                      className="result-card"
                      style={{animationDelay: `${idx * 0.1}s`}}
                    >
                      <div className="card-thumb">
                        <img src={`${API_URL}${result.url}`} alt="Event Frame" className="thumb-img" />
                        <div className="time-badge">
                          <Clock size={10} /> {result.display_time}
                        </div>
                      </div>
                      <div className="card-body">
                        <div className="metric-row">
                          <span style={{color: 'rgb(0, 0, 0)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.05em'}}>CONFIDENCE</span>
                          <span className={`score-badge ${result.score > 28 ? 'high' : 'med'}`}>
                            {result.score}%
                          </span>
                        </div>
                        <div className="metric-detail">
                          <span>Motion Energy</span>
                          <span style={{color: 'rgb(255, 255, 255)'}}>{result.motion_conf}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Right Sidebar */}
        <aside className="sidebar">
          <div className="panel card-content">
            <div className="card-title">
              <Upload size={14} /> Import Footage
            </div>
            
            <label className="upload-area">
              <span style={{color: 'var(--text-primary)', fontSize: '0.85rem', fontWeight: 500}}>
                {videoFile ? "File Selected" : "Select Video File"}
              </span>
              {videoFile && <div className="file-name">{videoFile.name}</div>}
              <input type="file" hidden accept="video/*" onChange={handleFileChange} />
            </label>

            {videoFile && (
              <button onClick={handleUpload} disabled={uploading} className="btn-primary">
                {uploading ? (
                  <span style={{display:'flex', alignItems:'center', justifyContent:'center', gap:'0.5rem'}}>
                    <Loader2 className="animate-spin" size={16}/> Processing...
                  </span>
                ) : (
                   "Run Pipeline"
                )}
              </button>
            )}
          </div>

          <div className="panel card-content">
            <div className="card-title">
              <Activity size={14} /> Pipeline Status
            </div>
            <div className="status-list">
              <div className="status-item">
                <span className="status-label"><Clock size={14}/> Temporal Unit</span>
                <span className="status-val">16 Frames</span>
              </div>
              <div className="status-item">
                <span className="status-label"><Activity size={14}/> Motion Gate</span>
                <span className="status-val" style={{color: '#000000'}}>Optical Flow</span>
              </div>
              <div className="status-item">
                <span className="status-label"><Server size={14}/> Backend</span>
                <span className="status-val" style={{color: '#000000'}}>FastAPI</span>
              </div>
              <div className="status-item">
                <span className="status-label"><Database size={14}/> Vector DB</span>
                <span className="status-val" style={{color: '#000000'}}>LanceDB</span>
              </div>
            </div>
          </div>
        </aside>

      </main>
    </div>
  );
}