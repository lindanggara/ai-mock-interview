import streamlit as st
import sys
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
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
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
        st.info("🎤 Mode suara: Bicara jawaban dan sistem akan mentranskripsikannya!")

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
            # ensure control flow re-runs so selection updates
            st.experimental_rerun()
    
    # Tampilkan Pertanyaan
    st.markdown("---")
    current_question = questions_data.get(category, {})
    
    st.markdown(f"### 💬 Pertanyaan: {category}")
    st.markdown(f'<div class="feature-card"><h4>{current_question.get("question","(Soal tidak ditemukan)")}</h4></div>', unsafe_allow_html=True)
    
    # Tips Berdasarkan CV
    if st.session_state.cv_uploaded and st.session_state.cv_data:
        with st.expander("🎯 Tips Personal Berdasarkan CV Anda"):
            cv_data = st.session_state.cv_data
            st.markdown(f"**Level Pengalaman Anda:** {cv_data.get('experience_level', 'N/A')}")
            
            st.markdown("**Skill Relevan yang Bisa Disebutkan:**")
            relevant_skills = []
            for s in cv_data.get('skills', []):
                s_lower = s.lower()
                if any(kw.lower() in s_lower for kw in current_question.get('keywords', [])):
                    relevant_skills.append(s)
            
            if relevant_skills:
                for skill in relevant_skills[:5]:
                    st.markdown(f"• {skill}")
            else:
                st.info("💡 Pertimbangkan untuk belajar skill yang relevan dengan pertanyaan ini")
    
    # Petunjuk Jawaban
    with st.expander("💡 Lihat Petunjuk"):
        ideal_range = current_question.get('ideal_length', (100, 250))
        st.markdown(f"""
        **Panjang Ideal:** {ideal_range[0]}-{ideal_range[1]} kata
        
        **Topik yang Harus Dibahas:** {', '.join(current_question.get('keywords', [])[:8])}
        
        **Fokus Penilaian:**
        - Teknis: {current_question.get('weight', {}).get('technical', 0.4)*100:.0f}%
        - Kedalaman: {current_question.get('weight', {}).get('depth', 0.3)*100:.0f}%
        - Komunikasi: {current_question.get('weight', {}).get('structure', 0.3)*100:.0f}%
        
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
        ideal_range = current_question.get('ideal_length', (100, 250))
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.caption(f"📝 Jumlah kata: {word_count}")
        with col_c2:
            st.caption(f"🎯 Target: {ideal_range[0]}-{ideal_range[1]} kata")
        with col_c3:
            status = "✅ Pas" if ideal_range[0] <= word_count <= ideal_range[1] else "⚠️ Perlu disesuaikan"
            st.caption(f"{status}")
        
    else:  # Mode suara
        st.markdown("""
        <div class="info-box">
        <h4>🎤 Cara Menggunakan Mode Suara:</h4>
        <ol>
        <li>Klik tombol "Mulai Rekaman"</li>
        <li>Bicara dengan jelas selama 2-3 menit</li>
        <li>Klik "Stop Rekaman" jika selesai</li>
        <li>Review transkripsi</li>
        <li>Klik "Analisis" untuk mendapat feedback</li>
        </ol>
        <p><strong>Tips:</strong> Bicara dengan kecepatan normal, jelas mengucapkan istilah teknis, dan gunakan metode STAR!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if st.button("🔴 Mulai Rekaman", use_container_width=True):
                st.info("🎤 Sedang merekam... Bicara sekarang!")
                st.session_state.recording = True
        with col_v2:
            if st.button("⏹️ Stop Rekaman", use_container_width=True):
                with st.spinner("Mentranskripsikan audio..."):
                    answer = voice_handler.transcribe_audio(category)
                    st.session_state.transcribed_answer = answer
                st.success("✅ Transkripsi selesai!")
        
        if st.session_state.get('transcribed_answer'):
            answer = st.text_area(
                "Hasil Transkripsi (bisa diedit):", 
                st.session_state.transcribed_answer, 
                height=200
            )
    
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
        quick_analysis = text_analyzer.quick_analysis(answer, current_question.get('keywords', []))
        
        st.markdown("### ⚡ Statistik Cepat")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        
        with col_q1:
            st.metric("Kata", quick_analysis.get('word_count', 0))
        with col_q2:
            st.metric("Kalimat", quick_analysis.get('sentence_count', 0))
        with col_q3:
            st.metric("Kata/Kalimat", f"{quick_analysis.get('avg_sentence_length', 0):.1f}")
        with col_q4:
            st.metric("Keyword", quick_analysis.get('keywords_found', 0))
        
        progress = quick_analysis.get('keyword_coverage', 0) / 100
        st.progress(progress)
        st.caption(f"Cakupan Keyword: {quick_analysis.get('keyword_coverage', 0):.0f}%")
    
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
                    question_weights=current_question.get('weight', {'technical':0.4,'depth':0.3,'structure':0.3}),
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
                    'question': current_question.get('question', ''),
                    'answer': answer,
                    'best_answer': best_answer,
                    'analysis': analysis_result,
                    'scores': scores,
                    'feedback': feedback
                }
                
                # Update history
                st.session_state.question_count += 1
                st.session_state.total_score += scores.get('overall', 0)
                st.session_state.interview_history.append({
                    'category': category,
                    'score': scores.get('overall', 0),
                    'difficulty': difficulty
                })
            
            # Tampilkan Hasil
            st.markdown("---")
            st.success("✅ Analisis Selesai!")
            
            # Skor Keseluruhan
            overall = scores.get('overall', 0)
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
                st.metric("🎯 Akurasi Teknis", f"{scores.get('technical_accuracy',0):.1f}/5.0")
            with col_s2:
                st.metric("📚 Kedalaman", f"{scores.get('depth_of_knowledge',0):.1f}/5.0")
            with col_s3:
                st.metric("💬 Komunikasi", f"{scores.get('communication_clarity',0):.1f}/5.0")
            
            # Bagian Feedback
            st.markdown("---")
            st.markdown("### 📝 Feedback Detail")
            
            # Jawaban Anda
            st.markdown("#### 📄 Jawaban Anda")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Jawaban Terbaik
            st.markdown("#### ✅ Contoh Jawaban Terbaik")
            best_preview = best_answer[:400] + "..." if isinstance(best_answer, str) and len(best_answer) > 400 else best_answer
            st.markdown(f'<div class="correct-answer">{best_preview}</div>', unsafe_allow_html=True)
            
            with st.expander("📖 Lihat Jawaban Lengkap"):
                st.markdown(best_answer)
            
            # Perbandingan
            st.markdown("#### 🔄 Analisis Perbandingan")
            col_comp1, col_comp2 = st.columns(2)
            
            strengths = feedback.get('strengths', [])
            gaps = feedback.get('gaps', [])
            improvements = feedback.get('improvements', [])
            recommendations = feedback.get('recommendations', [])
            specific_feedback = feedback.get('specific_feedback', None)
            summary = feedback.get('summary', None)
            
            with col_comp1:
                st.markdown("**✅ Yang Sudah Bagus:**")
                if strengths:
                    for strength in strengths:
                        st.markdown(f'<div class="strength-box">✅ {strength}</div>', unsafe_allow_html=True)
                else:
                    st.info("Belum ada kekuatan yang teridentifikasi")
            
            with col_comp2:
                st.markdown("**⚠️ Yang Masih Kurang:**")
                if gaps:
                    for gap in gaps:
                        st.markdown(f'<div class="improvement-box">⚠️ {gap}</div>', unsafe_allow_html=True)
                elif improvements:
                    for imp in improvements:
                        st.markdown(f'<div class="improvement-box">⚠️ {imp}</div>', unsafe_allow_html=True)
                else:
                    st.success("Jawaban sudah cukup lengkap!")
            
            # Feedback Spesifik
            st.markdown("#### 💡 Feedback Spesifik")
            # Use specific_feedback or join some recommendations as fallback
            specific_feedback_text = specific_feedback if specific_feedback else (" ".join(recommendations[:2]) if recommendations else "No specific feedback available.")
            st.markdown(f'<div class="feedback-box">{specific_feedback_text}</div>', unsafe_allow_html=True)
            
            # Area Perbaikan
            st.markdown("#### 🎯 Area yang Perlu Diperbaiki")
            if improvements:
                for i, improvement in enumerate(improvements, 1):
                    st.markdown(f"**{i}.** {improvement}")
            else:
                st.success("Jawaban Anda sudah sangat baik!")
            
            # Ringkasan
            st.markdown("#### 📋 Ringkasan")
            summary_text = summary if summary else (" ".join(recommendations[:2]) if recommendations else "Ringkasan tidak tersedia.")
            st.info(summary_text)
            
            # Rekomendasi
            with st.expander("📚 Rekomendasi Belajar"):
                st.markdown("**Sumber Belajar yang Direkomendasikan:**")
                for rec in recommendations:
                    st.markdown(f"• {rec}")

with tab2:
    st.markdown("### 📊 Dashboard Analitik Anda")
    
    if st.session_state.question_count == 0:
        st.info("📝 Mulai latihan untuk melihat analitik Anda!")
    else:
        # Grafik Progress
        if len(st.session_state.interview_history) > 0:
            st.markdown("#### 📈 Perkembangan Skor")
            try:
                progress_fig = viz_generator.create_progress_chart(st.session_state.interview_history)
                st.plotly_chart(progress_fig, use_container_width=True)
            except Exception as e:
                st.error("Gagal membuat grafik progress: " + str(e))
        
        # Performa per Kategori
        st.markdown("#### 🎯 Performa per Kategori")
        category_data = {}
        for item in st.session_state.interview_history:
            cat = item.get('category', 'Unknown')
            category_data.setdefault(cat, []).append(item.get('score', 0))
        
        if category_data:
            for cat, scores in category_data.items():
                avg = sum(scores) / len(scores) if scores else 0
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.metric(cat, f"{avg:.1f}/5.0")
                with col2:
                    st.caption(f"{len(scores)} percobaan")

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
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("""
        **📖 Buku**
        - Cracking the Data Science Interview
        - Data Science Handbook
        - Introduction to Statistical Learning
        - Python for Data Analysis
        """)
    
    with col_r2:
        st.markdown("""
        **🎥 Video & Course**
        - Kaggle Learn
        - DataCamp Career Tracks
        - StatQuest (YouTube)
        - Fast.ai Courses
        """)
    
    with col_r3:
        st.markdown("""
        **💻 Platform Latihan**
        - LeetCode (SQL & Python)
        - HackerRank Data Science
        - Kaggle Competitions
        - StrataScratch
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
