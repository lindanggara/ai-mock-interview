import streamlit as st
import sys
import plotly.graph_objects as go
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from text_mining import TextMiningAnalyzer
from scoring import ScoringEngine
from visualizations import VisualizationGenerator
from data_loader import DataLoader
from cv_analyzer import CVAnalyzer
from voice_handler import VoiceHandler

# Konfigurasi Halaman
st.set_page_config(
    page_title="Simulator Interview Data Science",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk UI/UX Modern
st.markdown("""
<style>
    /* Tema Utama */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Header */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: #333;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s;
    }
    
    .subtitle {
        text-align: center;
        color: #333;
        font-size: 3.0rem;
        font-weight: 800;
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4);
    }
    
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .correct-answer {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .feedback-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .improvement-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .strength-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Skor Cards */
    .score-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.3);
    }
    
    .score-good {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    .score-fair {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(240, 147, 251, 0.3);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'cv_uploaded' not in st.session_state:
    st.session_state.cv_uploaded = False
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = None
if 'interview_mode' not in st.session_state:
    st.session_state.interview_mode = 'text'
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Load Data
@st.cache_resource
def load_application_data():
    data_loader = DataLoader()
    return {
        'questions': data_loader.load_questions(),
        'keywords': data_loader.load_keywords(),
        'best_answers': data_loader.load_best_answers(),
        'stopwords': data_loader.load_stopwords()
    }

try:
    data = load_application_data()
    questions_data = data['questions']
    keywords_data = data['keywords']
    best_answers_data = data['best_answers']
    stopwords = data['stopwords']
except Exception as e:
    st.error(f"❌ Gagal memuat data: {str(e)}")
    st.info("💡 Pastikan semua file data ada di folder 'data/'")
    st.stop()

# Inisialisasi Komponen
text_analyzer = TextMiningAnalyzer(stopwords)
scoring_engine = ScoringEngine()
viz_generator = VisualizationGenerator()
cv_analyzer = CVAnalyzer()
voice_handler = VoiceHandler()

# Header
st.markdown('<h1 class="main-title">🎯 Simulator Interview Data Science</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Latihan Interview Data Science dengan Analisis AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Profil Anda")
    
    # Bagian Upload CV
    with st.expander("📄 Upload CV", expanded=not st.session_state.cv_uploaded):
        st.markdown("Upload CV Anda untuk mendapat rekomendasi personal")
        uploaded_file = st.file_uploader(
            "Pilih file CV (PDF/DOCX)",
            type=['pdf', 'docx', 'doc'],
            help="Upload CV untuk analisis skill dan rekomendasi pertanyaan"
        )
        
        if uploaded_file:
            with st.spinner("Menganalisis CV Anda..."):
                cv_data = cv_analyzer.analyze_cv(uploaded_file)
                st.session_state.cv_uploaded = True
                st.session_state.cv_data = cv_data
            
            if cv_data and not cv_data.get('error'):
                st.success("✅ CV Berhasil Dianalisis!")
                
                st.markdown("**Skill yang Terdeteksi:**")
                if cv_data.get('skills'):
                    for skill in cv_data['skills'][:8]:
                        st.markdown(f"• {skill}")
                
                st.markdown("**Level Pengalaman:**")
                level_map = {
                    'Junior': '🔰 Junior',
                    'Mid-level': '⭐ Mid-level',
                    'Senior': '🌟 Senior'
                }
                level = cv_data.get('experience_level', 'Mid-level')
                st.info(f"{level_map.get(level, level)}")
            else:
                st.error("❌ Gagal membaca CV. Coba file lain.")
    
    st.markdown("---")
    
    # Statistik Interview
    st.markdown("### 📊 Progress Latihan")
    if st.session_state.question_count > 0:
        avg_score = st.session_state.total_score / st.session_state.question_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pertanyaan", st.session_state.question_count)
        with col2:
            st.metric("Rata-rata", f"{avg_score:.1f}")
        
        # Klasifikasi skor
        if avg_score >= 4.0:
            st.success("🌟 Sangat Bagus!")
        elif avg_score >= 3.5:
            st.info("👍 Bagus!")
        else:
            st.warning("💪 Terus Berlatih!")
        
        if st.button("🔄 Reset Progress", use_container_width=True):
            st.session_state.interview_history = []
            st.session_state.total_score = 0
            st.session_state.question_count = 0
            st.session_state.current_analysis = None
            st.rerun()
    else:
        st.info("Mulai latihan untuk melihat progress Anda!")
    
    st.markdown("---")
    
    # Pilihan Mode Interview
    st.markdown("### 🎙️ Mode Interview")
    mode = st.radio(
        "Pilih mode latihan:",
        ["💬 Mode Teks", "🎤 Mode Suara"],
        index=0 if st.session_state.interview_mode == 'text' else 1,
        help="Mode teks: ketik jawaban | Mode suara: bicara jawaban"
    )
    st.session_state.interview_mode = 'text' if '💬' in mode else 'voice'
    
    if st.session_state.interview_mode == 'voice':
        st.info("🎤 Mode suara: Bicara jawaban Anda dan sistem akan mentranskripsikannya!")

# Konten Utama
tab1, tab2, tab3 = st.tabs(["🎯 Latihan Interview", "📊 Analitik", "💡 Tips & Panduan"])

with tab1:
    # Pemilihan Pertanyaan
    st.markdown("### 📝 Pilih Topik Pertanyaan")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        categories = list(questions_data.keys())
        category = st.selectbox(
            "Kategori:",
            categories,
            help="Pilih kategori sesuai fokus latihan Anda"
        )
    
    with col2:
        difficulty = st.select_slider(
            "Level:",
            options=["Junior", "Mid-level", "Senior"],
            value="Mid-level"
        )
    
    with col3:
        if st.button("🎲 Acak", use_container_width=True):
            import random
            category = random.choice(categories)
            st.rerun()
    
    # Tampilkan Pertanyaan
    st.markdown("---")
    current_question = questions_data[category]
    
    st.markdown(f"### 💬 Pertanyaan: {category}")
    st.markdown(f'<div class="feature-card"><h4>{current_question["question"]}</h4></div>', unsafe_allow_html=True)
    
    # Tips Berdasarkan CV
    if st.session_state.cv_uploaded and st.session_state.cv_data:
        with st.expander("🎯 Tips Personal Berdasarkan CV Anda"):
            cv_data = st.session_state.cv_data
            st.markdown(f"**Level Pengalaman Anda:** {cv_data.get('experience_level', 'N/A')}")
            
            st.markdown("**Skill Relevan yang Bisa Disebutkan:**")
            relevant_skills = [s for s in cv_data.get('skills', []) 
                             if any(kw in s.lower() for kw in current_question['keywords'])]
            if relevant_skills:
                for skill in relevant_skills[:5]:
                    st.markdown(f"• {skill}")
            else:
                st.info("💡 Pertimbangkan untuk belajar skill yang relevan dengan pertanyaan ini")
    
    # Petunjuk Jawaban
    with st.expander("💡 Lihat Petunjuk"):
        ideal_range = current_question['ideal_length']
        st.markdown(f"""
        **Panjang Ideal:** {ideal_range[0]}-{ideal_range[1]} kata
        
        **Topik yang Harus Dibahas:** {', '.join(current_question['keywords'][:8])}
        
        **Fokus Penilaian:**
        - Teknis: {current_question['weight']['technical']*100:.0f}%
        - Kedalaman: {current_question['weight']['depth']*100:.0f}%
        - Komunikasi: {current_question['weight']['structure']*100:.0f}%
        
        **Gunakan Metode STAR:**
        - **S**ituasi: Jelaskan konteksnya
        - **T**ugas: Apa tantangannya
        - **A**ksi: Apa yang Anda lakukan
        - **R**esult: Hasilnya apa (dengan angka!)
        """)
    
    # Input Jawaban
    st.markdown("---")
    st.markdown("### ✍️ Tulis Jawaban Anda")
    
    answer = ""
    
    if st.session_state.interview_mode == 'text':
        answer = st.text_area(
            "Ketik jawaban Anda di sini:",
            height=250,
            placeholder="""Contoh jawaban yang baik:

"Saya punya pengalaman 3 tahun menggunakan Python untuk data science. Di proyek terakhir saya menganalisis churn pelanggan untuk perusahaan e-commerce, saya pakai pandas untuk manipulasi 2 juta data transaksi dengan 15 fitur. Saya implementasi feature engineering pakai numpy array, buat rolling windows dan agregasi berbasis waktu. Untuk modeling, saya gunakan RandomForestClassifier dan XGBoost dari scikit-learn, mencapai akurasi 87% dengan F1-score 0.82. Model ini berhasil identifikasi 15 ribu pelanggan berisiko, dan kampanye retensi kami menyelamatkan pendapatan sekitar Rp 7 miliar per tahun. Saya deploy model pakai Flask API dengan Docker, handling 1000+ prediksi per detik."

Ingat: Sertakan angka, tools spesifik, dan dampak bisnis!""",
            key="answer_input"
        )
        
        # Penghitung kata
        word_count = len(answer.split()) if answer else 0
        ideal_range = current_question['ideal_length']
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.caption(f"📝 Jumlah kata: {word_count}")
        with col_c2:
            st.caption(f"🎯 Target: {ideal_range[0]}-{ideal_range[1]} kata")
        with col_c3:
            status = "✅ Pas" if ideal_range[0] <= word_count <= ideal_range[1] else "⚠️ Perlu disesuaikan"
            st.caption(f"{status}")
        
    else:  # Mode suara
        # Check if mic recorder available
        try:
            from streamlit_mic_recorder import mic_recorder
            mic_available = True
        except ImportError:
            mic_available = False
        
        if not mic_available:
            st.error("""
            ❌ **Library untuk Mode Suara belum terinstall!**
            
            Install dengan 1 command:
            """)
            st.code("pip install streamlit-mic-recorder SpeechRecognition", language="bash")
            st.info("""
            Setelah install, **restart aplikasi**:
            """)
            st.code("streamlit run app.py", language="bash")
            
            st.markdown("---")
            st.success("""
            💡 **Kelebihan Mode Suara Baru:**
            - ✅ Tidak perlu PyAudio (no hassle!)
            - ✅ Works di WSL/Windows/Mac/Linux
            - ✅ Gunakan browser microphone langsung
            - ✅ Simple & reliable
            """)
            
        else:
            # Mode suara tersedia!
            st.success("✅ Mode Suara siap digunakan!")
            
            st.markdown("""
            <div class="info-box">
            <h4>🎤 Cara Menggunakan:</h4>
            <ol>
            <li><strong>Klik "🔴 Rekam"</strong> di bawah</li>
            <li>Browser akan minta izin microphone → <strong>Allow/Izinkan</strong></li>
            <li><strong>Bicara dengan jelas</strong> dalam Bahasa Indonesia</li>
            <li><strong>Klik "⏹️ Stop"</strong> jika selesai (atau otomatis stop 3 menit)</li>
            <li><strong>Dengar playback</strong> untuk cek kualitas</li>
            <li><strong>Klik "📝 Transkripsi"</strong> untuk convert ke teks</li>
            <li><strong>Edit</strong> jika perlu, lalu <strong>Analisis</strong>!</li>
            </ol>
            
            <p><strong>💡 Tips:</strong></p>
            <ul>
            <li>🔇 Cari tempat tenang</li>
            <li>🎤 Gunakan headset untuk kualitas lebih baik</li>
            <li>📏 Target 2-3 menit</li>
            <li>⭐ Gunakan metode STAR</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Microphone recorder
            st.markdown("### 🎙️ Rekam Jawaban Anda")
            
            audio_data = mic_recorder(
                start_prompt="🔴 Rekam",
                stop_prompt="⏹️ Stop",
                just_once=False,
                use_container_width=True,
                format="wav",
                callback=None,
                args=(),
                kwargs={},
                key='voice_recorder'
            )
            
            # Jika ada audio yang direkam
            if audio_data:
                st.success("✅ Audio berhasil direkam!")
                
                # Tampilkan audio player
                st.audio(audio_data['bytes'], format='audio/wav')
                st.caption(f"Durasi: ~{len(audio_data['bytes']) / 16000:.1f} detik")
                
                # Tombol transkripsi
                col_t1, col_t2 = st.columns([1, 3])
                
                with col_t1:
                    transcribe_btn = st.button("📝 Transkripsi", use_container_width=True, type="primary")
                
                with col_t2:
                    st.caption("Klik untuk convert audio ke teks (memerlukan internet)")
                
                if transcribe_btn:
                    with st.spinner("🔄 Sedang mentranskripsikan audio... (10-30 detik)"):
                        transcribed_text = voice_handler.transcribe_from_audio_bytes(audio_data['bytes'])
                        st.session_state.transcribed_answer = transcribed_text
                    
                    if "❌" in transcribed_text:
                        st.error(transcribed_text)
                    else:
                        st.success("✅ Transkripsi berhasil!")
            
            # Tampilkan hasil transkripsi
            if st.session_state.get('transcribed_answer'):
                transcribed = st.session_state.transcribed_answer
                
                # Jika bukan error message
                if "❌" not in transcribed:
                    st.markdown("---")
                    st.markdown("### ✍️ Hasil Transkripsi")
                    st.info("💡 Anda bisa edit hasil transkripsi di bawah jika ada yang salah")
                    
                    answer = st.text_area(
                        "Edit jika perlu:",
                        transcribed,
                        height=200,
                        key="transcribed_text_area",
                        help="Hasil transkripsi bisa diedit untuk perbaiki kesalahan"
                    )
                    
                    # Word counter
                    word_count = len(answer.split()) if answer else 0
                    ideal_range = current_question['ideal_length']
                    
                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        st.caption(f"📝 Jumlah kata: {word_count}")
                    with col_c2:
                        st.caption(f"🎯 Target: {ideal_range[0]}-{ideal_range[1]} kata")
                    with col_c3:
                        status = "✅ Pas" if ideal_range[0] <= word_count <= ideal_range[1] else "⚠️ Perlu disesuaikan"
                        st.caption(f"{status}")
    
    # Tombol Aksi
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        analyze_btn = st.button("🔬 Analisis Jawaban", type="primary", use_container_width=True)
    with col_btn2:
        quick_btn = st.button("⚡ Statistik Cepat", use_container_width=True)
    with col_btn3:
        clear_btn = st.button("🗑️ Hapus", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    # Statistik Cepat
    if quick_btn and answer.strip():
        quick_analysis = text_analyzer.quick_analysis(answer, current_question['keywords'])
        
        st.markdown("### ⚡ Statistik Cepat")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        
        with col_q1:
            st.metric("Kata", quick_analysis['word_count'])
        with col_q2:
            st.metric("Kalimat", quick_analysis['sentence_count'])
        with col_q3:
            st.metric("Kata/Kalimat", f"{quick_analysis['avg_sentence_length']:.1f}")
        with col_q4:
            st.metric("Keyword", quick_analysis['keywords_found'])
        
        progress = quick_analysis['keyword_coverage'] / 100
        st.progress(progress)
        st.caption(f"Cakupan Keyword: {quick_analysis['keyword_coverage']:.0f}%")
    
    # Analisis Lengkap
    if analyze_btn:
        if not answer.strip():
            st.error("❌ Silakan tulis jawaban terlebih dahulu.")
        elif len(answer.split()) < 20:
            st.warning("⚠️ Jawaban terlalu singkat. Minimal 20 kata untuk analisis bermakna.")
        else:
            with st.spinner("🔬 Sedang menganalisis jawaban Anda..."):
                # Ambil jawaban terbaik
                best_answer = best_answers_data.get(category, {}).get('answer', '')
                
                # Jalankan analisis
                analysis_result = text_analyzer.comprehensive_analysis(
                    answer=answer,
                    question_data=current_question,
                    best_answer=best_answer,
                    category_keywords=keywords_data.get(category, [])
                )
                
                # Hitung skor
                scores = scoring_engine.calculate_scores(
                    analysis_result=analysis_result,
                    question_weights=current_question['weight'],
                    difficulty=difficulty
                )
                
                # Generate feedback
                feedback = scoring_engine.generate_detailed_feedback(
                    answer=answer,
                    best_answer=best_answer,
                    analysis_result=analysis_result,
                    scores=scores
                )
                
                # Simpan ke session
                st.session_state.current_analysis = {
                    'category': category,
                    'question': current_question['question'],
                    'answer': answer,
                    'best_answer': best_answer,
                    'analysis': analysis_result,
                    'scores': scores,
                    'feedback': feedback
                }
                
                # Update history
                st.session_state.question_count += 1
                st.session_state.total_score += scores['overall']
                st.session_state.interview_history.append({
                    'category': category,
                    'score': scores['overall'],
                    'difficulty': difficulty
                })
            
            # Tampilkan Hasil
            st.markdown("---")
            st.success("✅ Analisis Selesai!")
            
            # Skor Keseluruhan
            overall = scores['overall']
            if overall >= 4.5:
                score_class = "score-excellent"
                emoji = "🌟"
                label = "Luar Biasa!"
            elif overall >= 3.5:
                score_class = "score-good"
                emoji = "👍"
                label = "Bagus!"
            else:
                score_class = "score-fair"
                emoji = "💪"
                label = "Terus Tingkatkan!"
            
            st.markdown(f'<div class="{score_class}">{emoji} Skor Keseluruhan: {overall:.1f}/5.0 - {label}</div>', 
                       unsafe_allow_html=True)
            
            # Breakdown Skor
            st.markdown("### 📊 Rincian Skor")
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.metric("🎯 Akurasi Teknis", f"{scores['technical_accuracy']:.1f}/5.0")
            with col_s2:
                st.metric("📚 Kedalaman", f"{scores['depth_of_knowledge']:.1f}/5.0")
            with col_s3:
                st.metric("💬 Komunikasi", f"{scores['communication_clarity']:.1f}/5.0")
            
            # Bagian Feedback
            st.markdown("---")
            st.markdown("### 📝 Feedback Detail")
            
            # Jawaban Anda
            st.markdown("#### 📄 Jawaban Anda")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Jawaban Terbaik
            st.markdown("#### ✅ Contoh Jawaban Terbaik")
            best_preview = best_answer[:400] + "..." if len(best_answer) > 400 else best_answer
            st.markdown(f'<div class="correct-answer">{best_preview}</div>', unsafe_allow_html=True)
            
            with st.expander("📖 Lihat Jawaban Lengkap"):
                st.markdown(best_answer)
            
            # Perbandingan
            st.markdown("#### 🔄 Analisis Perbandingan")
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown("**✅ Yang Sudah Bagus:**")
                if feedback['strengths']:
                    for strength in feedback['strengths']:
                        st.markdown(f'<div class="strength-box">✅ {strength}</div>', unsafe_allow_html=True)
                else:
                    st.info("Belum ada kekuatan yang teridentifikasi")
            
            with col_comp2:
                st.markdown("**⚠️ Yang Masih Kurang:**")
                if feedback['gaps']:
                    for gap in feedback['gaps']:
                        st.markdown(f'<div class="improvement-box">⚠️ {gap}</div>', unsafe_allow_html=True)
                else:
                    st.success("Jawaban sudah cukup lengkap!")
            
            # Feedback Spesifik
            st.markdown("#### 💡 Feedback Spesifik")
            st.markdown(f'<div class="feedback-box">{feedback["specific_feedback"]}</div>', 
                       unsafe_allow_html=True)
            
            # Area Perbaikan
            st.markdown("#### 🎯 Area yang Perlu Diperbaiki")
            if feedback['improvements']:
                for i, improvement in enumerate(feedback['improvements'], 1):
                    st.markdown(f"**{i}.** {improvement}")
            else:
                st.success("Jawaban Anda sudah sangat baik!")
            
            # Ringkasan
            st.markdown("#### 📋 Ringkasan")
            st.info(feedback['summary'])

            # Rekomendasi
            with st.expander("📚 Rekomendasi Belajar"):
                st.markdown("**Sumber Belajar yang Direkomendasikan:**")
                for rec in feedback['recommendations']:
                    st.markdown(f"• {rec}")

with tab2:
    st.markdown("### 📊 Dashboard Analitik Anda")

    if st.session_state.question_count == 0:
        st.info("📝 Mulai latihan untuk melihat analitik Anda!")
    
    else:
        # ============ HITUNG METRIK ============
        avg_score = st.session_state.total_score / st.session_state.question_count
        best_score = max([item['score'] for item in st.session_state.interview_history])

        improvement = 0
        if len(st.session_state.interview_history) >= 2:
            recent_avg = sum([item['score'] for item in st.session_state.interview_history[-3:]]) / min(3, len(st.session_state.interview_history))
            first_avg = sum([item['score'] for item in st.session_state.interview_history[:3]]) / min(3, len(st.session_state.interview_history))
            improvement = recent_avg - first_avg

        # ============ HERO CARDS ============
        st.markdown("#### 🎯 Performance Overview")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

        with col_sum1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 15px; text-align: center;">
                <div style="color:white; opacity:0.9; font-size:0.9rem;">Total Latihan</div>
                <div style="color:white; font-size:2.5rem; font-weight:800;">{st.session_state.question_count}</div>
                <div style="color:white; opacity:0.8; font-size:0.85rem;">pertanyaan dijawab</div>
            </div>
            """, unsafe_allow_html=True)

        with col_sum2:
            avg_color = "#11998e" if avg_score >= 4.0 else "#667eea" if avg_score >= 3.5 else "#f093fb"
            avg_color_end = "#38ef7d" if avg_score >= 4.0 else "#764ba2" if avg_score >= 3.5 else "#f5576c"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {avg_color}, {avg_color_end});
                        padding: 1.5rem; border-radius: 15px; text-align: center;">
                <div style="color:white; opacity:0.9; font-size:0.9rem;">Rata-rata Skor</div>
                <div style="color:white; font-size:2.5rem; font-weight:800;">
                    {avg_score:.1f}<span style="font-size:1.2rem; opacity:0.8;">/5.0</span>
                </div>
                <div style="color:white; opacity:0.8; font-size:0.85rem;">
                    {'🎯 Excellent!' if avg_score >= 4.0 else '👍 Good!' if avg_score >= 3.5 else '💪 Keep Going!'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sum3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb, #f5576c);
                        padding: 1.5rem; border-radius: 15px; text-align: center;">
                <div style="color:white; opacity:0.9; font-size:0.9rem;">Skor Terbaik</div>
                <div style="color:white; font-size:2.5rem; font-weight:800;">
                    {best_score:.1f}<span style="font-size:1.2rem; opacity:0.8;">/5.0</span>
                </div>
                <div style="color:white; opacity:0.8; font-size:0.85rem;">🏆 personal best</div>
            </div>
            """, unsafe_allow_html=True)

        with col_sum4:
            imp_color = "#11998e" if improvement > 0 else "#667eea" if improvement == 0 else "#dc3545"
            imp_color_end = "#38ef7d" if improvement > 0 else "#764ba2" if improvement == 0 else "#f5576c"
            imp_icon = "📈" if improvement > 0 else "➡️" if improvement == 0 else "📉"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {imp_color}, {imp_color_end});
                        padding: 1.5rem; border-radius: 15px; text-align: center;">
                <div style="color:white; opacity:0.9; font-size:0.9rem;">Peningkatan</div>
                <div style="color:white; font-size:2.5rem; font-weight:800;">{improvement:+.1f}</div>
                <div style="color:white; opacity:0.8; font-size:0.85rem;">{imp_icon} dari awal</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ============ KATEGORI CHART ============
        st.markdown("#### 📊 Category Performance Ranking")

        # buat data kategori
        category_data = {}
        for item in st.session_state.interview_history:
            cat = item['category']
            category_data.setdefault(cat, []).append(item['score'])

        category_avgs = {cat: sum(v) / len(v) for cat, v in category_data.items()}

        sorted_cats = sorted(category_avgs.items(), key=lambda x: x[1], reverse=True)
        cats_sorted = [c[0] for c in sorted_cats]
        scores_sorted = [c[1] for c in sorted_cats]
        attempts_sorted = [len(category_data[c[0]]) for c in sorted_cats]

        colors_sorted = [
            '#11998e' if s >= 4 else '#667eea' if s >= 3.5 else '#f093fb' if s >= 3 else '#dc3545'
            for s in scores_sorted
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cats_sorted,
            x=scores_sorted,
            orientation='h',
            text=[f"{s:.1f} ({a}x)" for s, a in zip(scores_sorted, attempts_sorted)],
            textposition='outside',
            marker=dict(color=colors_sorted, line=dict(color='white', width=2))
        ))

        fig.update_layout(
            xaxis=dict(range=[0, 5.5]),
            height=max(300, len(cats_sorted) * 50),
            margin=dict(l=150)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Hijau ≥4.0 | Biru ≥3.5 | Pink ≥3.0 | Merah <3.0")

        st.markdown("---")

        # ============ PROGRESS CHART ============

        st.markdown("#### 📈 Perkembangan Skor dari Waktu ke Waktu")

        progress_fig = viz_generator.create_progress_chart(st.session_state.interview_history)
        st.plotly_chart(progress_fig, use_container_width=True)

        st.caption("💡 Biru: skor | Hijau: trend | Kuning: target 3.5")

        st.markdown("---")

        # ============ PERFORMA KATEGORI (BAR CHART) ============
        st.markdown("#### 🎯 Performa per Kategori")

        categories = list(category_avgs.keys())
        avgs = list(category_avgs.values())
        attempts = [len(category_data[c]) for c in categories]

        colors = ['#28a745' if a >= 4 else '#ffc107' if a >= 3.5 else '#dc3545' for a in avgs]

        fig = go.Figure([go.Bar(
            x=categories,
            y=avgs,
            text=[f"{a:.1f}<br>({t}x)" for a, t in zip(avgs, attempts)],
            textposition='outside',
            marker_color=colors
        )])

        fig.update_layout(
            yaxis=dict(range=[0, 5.5]),
            height=400
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="orange")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ============ DIFFICULTY BREAKDOWN ============
        st.markdown("#### 📊 Performa Berdasarkan Level Kesulitan")

        difficulty_data = {}
        for item in st.session_state.interview_history:
            difficulty_data.setdefault(item['difficulty'], []).append(item['score'])

        col_d1, col_d2, col_d3 = st.columns(3)
        for col, lvl in zip([col_d1, col_d2, col_d3], ['Junior', 'Mid-level', 'Senior']):
            if lvl in difficulty_data:
                scores = difficulty_data[lvl]
                col.metric(lvl, f"{sum(scores)/len(scores):.1f}/5.0", f"{len(scores)} percobaan")

        st.markdown("---")

        # ============ STRENGTH & WEAKNESS ============
        st.markdown("#### 💪 Kekuatan & Area Pengembangan")

        col_sw1, col_sw2 = st.columns(2)

        with col_sw1:
            st.markdown("**🌟 Kategori Terkuat:**")
            top_3 = sorted(category_avgs.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (cat, s) in enumerate(top_3, 1):
                st.markdown(f"{i}. **{cat}** — {s:.1f}/5.0")

        with col_sw2:
            st.markdown("**📈 Perlu Latihan:**")
            bottom_3 = sorted(category_avgs.items(), key=lambda x: x[1])[:3]
            for i, (cat, s) in enumerate(bottom_3, 1):
                st.markdown(f"{i}. **{cat}** — {s:.1f}/5.0")

        st.markdown("---")

        # ============ RECENT 5 ============
        st.markdown("#### 🕐 5 Latihan Terakhir")

        recent = st.session_state.interview_history[-5:]
        for i, item in enumerate(reversed(recent), 1):
            c1, c2, c3 = st.columns([3,1,1])
            with c1:
                st.markdown(f"**{len(st.session_state.interview_history)-i+1}. {item['category']}**")
            with c2:
                color = "🟢" if item['score']>=4 else "🟡" if item['score']>=3.5 else "🔴"
                st.markdown(f"{color} {item['score']:.1f}/5.0")
            with c3:
                st.caption(item['difficulty'])

with tab3:
    st.markdown("### 💡 Tips & Panduan Interview")
    
    col_tip1, col_tip2 = st.columns(2)
    
    with col_tip1:
        st.markdown("""
        #### 🎯 Metode STAR
        
        **S**ituasi - Jelaskan konteksnya
        - Di mana Anda bekerja?
        - Apa proyeknya?
        
        **T**ugas - Jelaskan tantangannya
        - Masalah apa yang diselesaikan?
        - Apa peran Anda?
        
        **A**ksi - Jelaskan yang Anda lakukan
        - Tools/metode apa yang digunakan?
        - Bagaimana pendekatannya?
        
        **R**esult - Bagikan hasilnya
        - Apa dampaknya?
        - Apakah ada angka/metrik?
        """)
    
    with col_tip2:
        st.markdown("""
        #### ✅ Hal yang Perlu Dilakukan
        
        **Lakukan:**
        - ✅ Gunakan contoh spesifik
        - ✅ Sebutkan tools & library
        - ✅ Kuantifikasi hasil (15% peningkatan)
        - ✅ Tunjukkan proses berpikir
        - ✅ Hubungkan ke nilai bisnis
        
        **Jangan:**
        - ❌ Terlalu teoritis
        - ❌ Abaikan konteks bisnis
        - ❌ Gunakan jargon tanpa penjelasan
        - ❌ Jawaban generik
        """)
    
    st.markdown("---")
    st.markdown("#### 📚 Sumber Belajar yang Direkomendasikan")
    
    st.info("💡 Klik link di bawah untuk langsung mengakses sumber belajar")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("""
        **📖 Buku & E-Book**
        
        - [Python for Data Analysis (O'Reilly)](https://wesmckinney.com/book/)
        - [Introduction to Statistical Learning](https://www.statlearning.com/)
        - [Hands-On Machine Learning](https://github.com/ageron/handson-ml2)
        - [Deep Learning Book](https://www.deeplearningbook.org/)
        """)
    
    with col_r2:
        st.markdown("""
        **🎥 Video & Course Gratis**
        
        - [Kaggle Learn](https://www.kaggle.com/learn)
        - [Fast.ai Courses](https://www.fast.ai/)
        - [StatQuest YouTube](https://www.youtube.com/@statquest)
        - [FreeCodeCamp DS](https://www.freecodecamp.org/learn/data-analysis-with-python/)
        - [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
        """)
    
    with col_r3:
        st.markdown("""
        **💻 Platform Latihan**
        
        - [LeetCode Database](https://leetcode.com/problemset/database/)
        - [HackerRank SQL](https://www.hackerrank.com/domains/sql)
        - [Kaggle Competitions](https://www.kaggle.com/competitions)
        - [StrataScratch](https://www.stratascratch.com/)
        - [DataCamp (Free tier)](https://www.datacamp.com/courses)
        """)
    
    st.markdown("---")
    
    st.markdown("#### 🇮🇩 Komunitas & Forum Indonesia")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("""
        **💬 Komunitas Online**
        
        - [Indonesia AI LinkedIn](https://www.linkedin.com/company/indonesia-ai/)
        - [Data Science Indonesia (Telegram)](https://t.me/datascienceindonesia)
        - [Python Indonesia (Discord)](https://discord.gg/python-indonesia)
        - [r/dataengineering](https://www.reddit.com/r/dataengineering/)
        """)
    
    with col_c2:
        st.markdown("""
        **🎓 Bootcamp & Course Indonesia**
        
        - [Dicoding (Machine Learning Path)](https://www.dicoding.com/learningpaths/30)
        - [Skilvul](https://skilvul.com/)
        - [MySkill.id](https://myskill.id/)
        - [BuildWithAngga](https://buildwithangga.com/)
        """)
    
    st.markdown("---")
    
    st.markdown("#### 📰 Blog & Newsletter")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("""
        **📝 Blog Teknis**
        
        - [Towards Data Science](https://towardsdatascience.com/)
        - [Analytics Vidhya](https://www.analyticsvidhya.com/blog/)
        - [KDnuggets](https://www.kdnuggets.com/)
        - [Machine Learning Mastery](https://machinelearningmastery.com/)
        """)
    
    with col_b2:
        st.markdown("""
        **📧 Newsletter**
        
        - [Data Science Weekly](https://www.datascienceweekly.org/)
        - [The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/)
        - [ImportAI](https://jack-clark.net/)
        - [TLDR AI](https://tldr.tech/ai)
        """)
    
    st.markdown("---")
    
    st.markdown("#### 🛠️ Tools & Praktik**")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("""
        **💻 Development Environment**
        
        - [Jupyter Lab](https://jupyter.org/)
        - [Google Colab](https://colab.research.google.com/)
        - [Kaggle Notebooks](https://www.kaggle.com/code)
        - [VS Code](https://code.visualstudio.com/)
        """)
    
    with col_t2:
        st.markdown("""
        **📊 Portfolio Projects**
        
        - [GitHub Portfolio Ideas](https://github.com/topics/data-science-portfolio)
        - [Kaggle Project Showcase](https://www.kaggle.com/datasets)
        - [Awesome Data Science](https://github.com/academic/awesome-datascience)
        """)
    
    st.markdown("---")
    
    st.success("""
    💡 **Tips Belajar Efektif:**
    1. Pilih 1-2 sumber dan fokus sampai selesai
    2. Praktik dengan project nyata, bukan hanya tutorial
    3. Join komunitas untuk networking & belajar bareng
    4. Konsisten 1-2 jam per hari lebih baik dari marathon weekend
    5. Build portfolio di GitHub untuk showcase ke recruiter
    """)
    
    st.markdown("---")
    st.markdown("#### 🎓 Contoh Jawaban: Bagus vs Kurang Bagus")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("##### ❌ Jawaban Kurang Bagus (Skor: 2.5)")
        st.markdown("""
        <div class="improvement-box">
        <p>"Saya tahu Python dan pernah pakai untuk analisis data. Saya sudah buat beberapa 
        proyek machine learning dan bisa kerja dengan data. Saya paham pandas dan numpy 
        dan pernah pakai sebelumnya."</p>
        
        <p><strong>Masalah:</strong></p>
        <ul>
        <li>Terlalu vague</li>
        <li>Tidak ada contoh spesifik</li>
        <li>Tidak ada hasil terukur</li>
        <li>Tidak ada konteks/dampak</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ex2:
        st.markdown("##### ✅ Jawaban Bagus (Skor: 4.8)")
        st.markdown("""
        <div class="strength-box">
        <p>"Saya punya pengalaman 3 tahun dengan Python untuk data science. Di proyek 
        terakhir menganalisis churn pelanggan untuk perusahaan e-commerce, saya gunakan 
        pandas untuk manipulasi 2 juta record transaksi dengan 15 fitur. Saya implementasi 
        feature engineering pakai numpy array, buat rolling windows dan agregasi time-based. 
        Untuk modeling, saya pakai RandomForestClassifier dan XGBoost dari scikit-learn, 
        mencapai akurasi 87% dengan F1-score 0.82. Model ini identifikasi 15 ribu pelanggan 
        berisiko, dan kampanye retensi kami selamatkan revenue Rp 7 miliar per tahun. 
        Saya deploy model pakai Flask API dengan Docker, handle 1000+ prediksi per detik."</p>
        
        <p><strong>Kenapa Bagus:</strong></p>
        <ul>
        <li>✅ Tools & library spesifik</li>
        <li>✅ Angka konkret (2M records, 15 fitur)</li>
        <li>✅ Detail teknis (RandomForest, XGBoost)</li>
        <li>✅ Hasil terukur (87% akurasi, Rp 7M saved)</li>
        <li>✅ Pipeline lengkap (development → deployment)</li>
        <li>✅ Dampak bisnis jelas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🎤 Panduan Mode Suara")
    
    st.markdown("""
    **Persiapan:**
    1. ✅ Cek microphone browser Anda
    2. ✅ Cari tempat yang tenang
    3. ✅ Gunakan headset untuk kualitas lebih baik
    4. ✅ Tes volume audio
    
    **Saat Merekam:**
    - 🎯 Bicara dengan kecepatan normal
    - 🎯 Ucapkan istilah teknis dengan jelas
    - 🎯 Jeda antar pikiran
    - 🎯 Gunakan bahasa natural
    - 🎯 Struktur dengan metode STAR
    
    **Tips untuk Istilah Teknis:**
    - Eja akronim: "S-Q-L" bukan "sequel"
    - Jeda sebelum istilah teknis
    - Gunakan konteks
    - Bicara sedikit lebih lambat untuk istilah kompleks
    """)

# Footer
st.markdown("---")
st.caption("🎯 Simulator Interview Data Science | Dibuat dengan ❤️ untuk Data Scientist Indonesia")
