# transcript retrieval using Whisper
import os
import tempfile
from urllib.parse import urlparse, parse_qs
from faster_whisper import WhisperModel
import yt_dlp


# --------------------------
# 🔹 Utility Functions
# --------------------------
def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from standard, short, or embed URLs."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    # Case 1: Standard YouTube link
    if "v" in query:
        return query["v"][0]

    # Case 2: Shortened youtu.be link
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    # Case 3: Embed or shorts link
    path_parts = parsed.path.split("/")
    for part in path_parts[::-1]:
        if len(part) == 11:  # typical video ID length
            return part

    return None


# --------------------------
# 🔹 Audio Download with yt-dlp
# --------------------------
def download_audio_ytdlp(url: str, output_path: str):
    """Download audio using yt-dlp Python library (no FFmpeg needed for basic audio)."""
    print("⬇️ Downloading audio using yt-dlp...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if has_ffmpeg() else [],
    }
    
    # If no FFmpeg, just download the audio stream directly
    if not has_ffmpeg():
        print("ℹ️  FFmpeg not detected - downloading audio in original format")
        ydl_opts['format'] = 'bestaudio'
        # Remove .mp3 extension if present, yt-dlp will add correct extension
        if output_path.endswith('.mp3'):
            output_path = output_path[:-4]
            ydl_opts['outtmpl'] = output_path
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            print(f"📺 Title: {info.get('title', 'Unknown')}")
            duration = info.get('duration', 0)
            print(f"⏱️  Duration: {duration // 60}:{duration % 60:02d}")
            print("✅ Audio downloaded successfully.")
            
            # Find the actual downloaded file
            if has_ffmpeg():
                downloaded_file = output_path
            else:
                # yt-dlp adds extension automatically
                base = output_path
                for ext in ['.webm', '.m4a', '.mp4', '.opus']:
                    if os.path.exists(base + ext):
                        downloaded_file = base + ext
                        break
                else:
                    downloaded_file = output_path
            
            return downloaded_file
            
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        raise


def has_ffmpeg():
    """Check if FFmpeg is available."""
    import shutil
    return shutil.which('ffmpeg') is not None


# --------------------------
# 🔹 Speech-to-Text with Whisper
# --------------------------
def transcribe_audio(audio_path: str, model_size: str = "tiny"):
    """
    Run faster-whisper to transcribe locally.
    
    Model sizes: tiny, base, small, medium, large
    - tiny: fastest, least accurate (~1GB RAM)
    - base: good balance (~1GB RAM) - default
    - small: better accuracy (~2GB RAM)
    - medium: high accuracy (~5GB RAM)
    - large: best accuracy (~10GB RAM)
    """
    print(f"\n🎙 Starting Whisper transcription with '{model_size}' model...")
    print("   (This may take a few minutes depending on audio length)")
    
    # Use GPU if available, otherwise CPU
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Check if onnxruntime is available for VAD
    use_vad = False
    try:
        import onnxruntime
        use_vad = True
        print("   ✓ VAD (Voice Activity Detection) enabled")
    except ImportError:
        print("   ℹ️  VAD disabled (install onnxruntime for better accuracy)")
    
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",  # Remove this line if you want auto-detection
        vad_filter=use_vad,  # Voice activity detection - improves accuracy
    )
    
    print(f"📝 Detected language: {info.language} (confidence: {info.language_probability:.2%})")
    
    # Collect all segments
    transcript_parts = []
    segment_count = 0
    
    for segment in segments:
        transcript_parts.append(segment.text.strip())
        segment_count += 1
        
        # Progress indicator for long videos
        if segment_count % 50 == 0:
            print(f"   ... processed {segment_count} segments")
    
    text = " ".join(transcript_parts)
    print(f"✅ Transcription complete! ({segment_count} segments)")
    
    return text


# --------------------------
# 🔹 Main Orchestration
# --------------------------
def get_transcript(url: str, model_size: str = "base"):
    """
    Generate transcript from YouTube video using Whisper.
    
    Args:
        url: YouTube video URL
        model_size: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        Transcript text
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("❌ Invalid YouTube URL — could not extract video ID.")

    print(f"🎯 Video ID: {video_id}\n")

    # Download audio and transcribe
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_path = tmp.name
    
    downloaded_file = None
    
    try:
        downloaded_file = download_audio_ytdlp(url, temp_path)
        transcript = transcribe_audio(downloaded_file, model_size)
        return transcript
    finally:
        # Clean up temp files
        for file_path in [temp_path, downloaded_file]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        print("\n🧹 Cleaned up temporary files.")


# --------------------------
# 🔹 Example Run
# --------------------------
if __name__ == "__main__":
    video_url = "https://youtu.be/fNk_zzaMoSs?si=E8RezC_6w2kvcJ7H"
    
    try:
        print("="*60)
        print("🎬 YouTube Transcript Generator with Whisper")
        print("="*60 + "\n")
        
        # You can change model_size to: tiny, base, small, medium, large
        # Larger models are more accurate but slower
        text = get_transcript(video_url, model_size="base")
        
        print(f"\n{'='*60}")
        print(f"🧾 TRANSCRIPT SUMMARY")
        print(f"{'='*60}")
        print(f"Length: {len(text)} characters")
        print(f"Words: ~{len(text.split())} words")
        print(f"\n{'='*60}")
        print(f"PREVIEW (first 500 chars):")
        print(f"{'='*60}")
        print(text[:500] + "...\n")
        
        # Optionally save to file
        output_file = "transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Full transcript saved to '{output_file}'")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(1)