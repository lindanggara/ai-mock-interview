"""
Scoring Engine Module
Calculates final scores and generates feedback
"""


class ScoringEngine:
    """
    Engine for calculating interview scores based on text mining results
    """
    
    def __init__(self):
        self.difficulty_multipliers = {
            'Junior': 0.9,
            'Mid-level': 1.0,
            'Senior': 1.15
        }
    
    def calculate_scores(self, analysis_result, question_weights, difficulty='Mid-level'):
        """
        Calculate final scores from analysis results
        
        Args:
            analysis_result (dict): Results from TextMiningAnalyzer
            question_weights (dict): Weights for different aspects
            difficulty (str): Difficulty level
            
        Returns:
            dict: Calculated scores
        """
        # Extract component scores
        keyword_score = analysis_result['keyword_analysis']['score']
        tfidf_score = analysis_result['tfidf']['score']
        ner_score = analysis_result['ner']['score']
        sentiment_score = analysis_result['sentiment']['score']
        readability_score = analysis_result['readability']['score']
        structural_score = analysis_result['structural']['score']
        coherence_score = analysis_result['coherence']['score']
        
        # Add similarity if available
        similarity_score = analysis_result['similarity'].get('score', 0) if analysis_result.get('similarity') else 0
        
        # Calculate composite scores
        # Technical Accuracy: keyword coverage + NER + similarity
        technical_accuracy = (
            keyword_score * 0.35 +
            ner_score * 0.35 +
            similarity_score * 0.30
        )
        
        # Depth of Knowledge: TF-IDF + structure + ngrams
        ngram_score = analysis_result['ngrams'].get('score', 0)
        depth_of_knowledge = (
            tfidf_score * 0.40 +
            structural_score * 0.40 +
            ngram_score * 0.20
        )
        
        # Communication Clarity: readability + sentiment + coherence
        communication_clarity = (
            readability_score * 0.40 +
            coherence_score * 0.35 +
            sentiment_score * 0.25
        )
        
        # Apply question-specific weights
        weights = question_weights
        overall_score = (
            technical_accuracy * weights['technical'] +
            depth_of_knowledge * weights['depth'] +
            communication_clarity * weights['structure']
        )
        
        # Apply difficulty multiplier
        multiplier = self.difficulty_multipliers.get(difficulty, 1.0)
        overall_score = min(overall_score * multiplier, 5.0)
        
        # Ensure all scores are within bounds
        return {
            'technical_accuracy': round(min(technical_accuracy, 5.0), 2),
            'depth_of_knowledge': round(min(depth_of_knowledge, 5.0), 2),
            'communication_clarity': round(min(communication_clarity, 5.0), 2),
            'overall': round(min(overall_score, 5.0), 2),
            'components': {
                'keyword': keyword_score,
                'tfidf': tfidf_score,
                'ner': ner_score,
                'sentiment': sentiment_score,
                'readability': readability_score,
                'structural': structural_score,
                'coherence': coherence_score,
                'similarity': similarity_score
            }
        }
    
    def generate_detailed_feedback(self, answer, best_answer, analysis_result, scores):
        """
        Generate comprehensive feedback with comparison (INDONESIAN VERSION)
        
        Args:
            answer (str): User's answer
            best_answer (str): Reference answer
            analysis_result (dict): Analysis results
            scores (dict): Calculated scores
            
        Returns:
            dict: Detailed feedback with multiple sections
        """
        strengths = []
        improvements = []
        gaps = []
        recommendations = []
        
        # Compare answers
        answer_points = self._extract_key_points(answer)
        best_points = self._extract_key_points(best_answer)
        
        # Identify gaps (apa yang ada di best answer tapi tidak di jawaban user)
        for point in best_points[:5]:  # Ambil 5 point penting dari best answer
            if not any(self._similarity_check(point, ap) for ap in answer_points):
                # Ekstrak kata kunci dari point yang missing
                words = [w for w in point.lower().split() if len(w) > 4][:3]
                if words:
                    gaps.append(f"Tidak menyebutkan tentang {' '.join(words[:2])}")
        
        # Limit gaps
        gaps = gaps[:5]  # Maksimal 5 gaps
        
        # Analyze keyword coverage
        keyword_data = analysis_result['keyword_analysis']
        if keyword_data['coverage'] >= 60:
            strengths.append(
                f"Cakupan keyword sangat baik ({keyword_data['coverage']:.0f}%) - "
                "Anda menyebutkan sebagian besar istilah teknis yang penting"
            )
        elif keyword_data['coverage'] >= 40:
            strengths.append(
                f"Cakupan keyword cukup baik ({keyword_data['coverage']:.0f}%) - "
                "Pemahaman konsep kunci sudah solid"
            )
        elif keyword_data['coverage'] < 30:
            improvements.append(
                f"Cakupan keyword masih kurang ({keyword_data['coverage']:.0f}%) - "
                "Coba sertakan lebih banyak istilah teknis yang relevan"
            )
            recommendations.append(
                "Review kembali konsep kunci untuk topik ini dan pastikan jawaban Anda mencakup semuanya"
            )
        
        # Analyze technical entities
        ner_data = analysis_result['ner']
        if ner_data['total'] >= 5:
            strengths.append(
                f"Kedalaman teknis kuat - menyebutkan {ner_data['total']} tools/methods/metrics spesifik "
                f"dari {ner_data['diversity']} kategori berbeda"
            )
        elif ner_data['total'] >= 3:
            strengths.append(
                f"Referensi teknis bagus - menggunakan {ner_data['total']} istilah spesifik"
            )
        elif ner_data['total'] < 2:
            improvements.append(
                "Kurang spesifik secara teknis - sebutkan nama tools, library, atau metodologi konkret"
            )
            recommendations.append(
                "Sertakan nama spesifik tools/library/framework yang Anda gunakan dalam proyek"
            )
        
        # Analyze structure
        struct_data = analysis_result['structural']
        if struct_data['has_examples'] and struct_data['has_structure']:
            strengths.append(
                "Struktur jawaban rapi dengan contoh konkret - menunjukkan pengalaman praktis"
            )
        elif struct_data['has_examples']:
            strengths.append(
                "Memberikan contoh konkret - demonstrasi pengetahuan praktis yang baik"
            )
        elif not struct_data['has_examples']:
            improvements.append(
                "Tidak ada contoh spesifik - jawaban terasa terlalu teoritis"
            )
            recommendations.append(
                "Gunakan metode STAR: jelaskan Situasi, Tugas, Aksi, dan Result dari pengalaman Anda"
            )
        
        if not struct_data['length_appropriate']:
            if analysis_result['readability']['word_count'] < 50:
                improvements.append(
                    "Jawaban terlalu singkat - berikan penjelasan yang lebih detail"
                )
                recommendations.append(
                    "Target minimal 100 kata. Elaborasi proses berpikir dan reasoning Anda"
                )
            elif analysis_result['readability']['word_count'] > 400:
                improvements.append(
                    "Jawaban cukup panjang - fokus pada poin-poin paling penting"
                )
                recommendations.append(
                    "Latih untuk lebih concise. Target 150-250 kata untuk kebanyakan pertanyaan"
                )
        
        # Analyze readability
        read_data = analysis_result['readability']
        if read_data['assessment'] in ['Excellent', 'Good']:
            strengths.append(
                f"Komunikasi jelas dan terstruktur ({read_data['assessment'].lower()} readability)"
            )
        elif read_data['assessment'] in ['Poor structure', 'Needs improvement']:
            improvements.append(
                f"Readability bisa ditingkatkan - {read_data['assessment'].lower()}"
            )
            recommendations.append(
                f"Panjang kalimat rata-rata Anda {read_data['avg_sentence_length']:.1f} kata. "
                "Target 15-20 kata per kalimat untuk kejelasan optimal"
            )
        
        # Analyze sentiment
        sent_data = analysis_result['sentiment']
        if sent_data['polarity'] > 0.2:
            strengths.append(
                "Tone percaya diri dan positif sepanjang jawaban"
            )
        elif sent_data['polarity'] < -0.1:
            improvements.append(
                "Tone terkesan ragu atau negatif - proyeksikan lebih banyak kepercayaan diri"
            )
            recommendations.append(
                "Gunakan bahasa yang lebih positif. Daripada fokus pada tantangan, tekankan solusi"
            )
        
        # Analyze coherence
        coh_data = analysis_result['coherence']
        if coh_data['score'] >= 4.0:
            strengths.append(
                "Alur logika sangat baik dengan penggunaan kata transisi yang tepat"
            )
        elif coh_data['score'] < 2.5:
            improvements.append(
                "Jawaban kurang koheren - ide-ide terlihat terpisah"
            )
            recommendations.append(
                "Gunakan kata transisi seperti 'namun', 'selain itu', 'oleh karena itu' untuk menghubungkan ide"
            )
        
        # Analyze similarity (if available)
        if 'similarity' in analysis_result and analysis_result['similarity']:
            sim_data = analysis_result['similarity']
            if sim_data['cosine_similarity'] >= 0.5:
                strengths.append(
                    f"Alignment bagus dengan best practices ({sim_data['cosine_similarity']:.0%} kesamaan)"
                )
            elif sim_data['cosine_similarity'] < 0.3:
                improvements.append(
                    "Jawaban cukup berbeda dari pendekatan yang diharapkan"
                )
                recommendations.append(
                    "Pelajari contoh jawaban interview umum untuk topik ini untuk memahami ekspektasi"
                )
        
        # Analyze TF-IDF
        tfidf_data = analysis_result['tfidf']
        if tfidf_data['lexical_diversity'] >= 0.6:
            strengths.append(
                f"Kosakata kaya dengan {tfidf_data['lexical_diversity']:.0%} lexical diversity"
            )
        elif tfidf_data['lexical_diversity'] < 0.3:
            improvements.append(
                "Kosakata terbatas - jawaban terkesan repetitif"
            )
            recommendations.append(
                "Variasikan pilihan kata dan hindari mengulang istilah yang sama"
            )
        
        # Analyze numbers/metrics
        if not struct_data.get('has_numbers', False):
            improvements.append(
                "Tidak ada angka atau metrik - hasil kurang konkret"
            )
            recommendations.append(
                "Kuantifikasi hasil dengan angka spesifik: 'akurasi 87%', 'proses 2 juta records', dll"
            )
        
        # Overall recommendations based on score
        overall_score = scores['overall']
        if overall_score >= 4.0:
            recommendations.append(
                "Jawaban excellent! Terus latihan dengan pertanyaan yang lebih menantang"
            )
        elif overall_score >= 3.0:
            recommendations.append(
                "Fondasi solid. Fokus pada penambahan kedalaman teknis dan contoh spesifik"
            )
        else:
            recommendations.append(
                "Terus berlatih! Review konsep fundamental dan pelajari contoh jawaban"
            )
        
        # Generate specific actionable feedback
        specific_feedback = self._generate_specific_feedback(
            answer, best_answer, analysis_result, scores
        )
        
        # Generate summary
        summary = self._generate_summary(scores, strengths, improvements)
        
        return {
            'strengths': strengths[:4],
            'improvements': improvements[:4],
            'gaps': gaps if gaps else ["Jawaban Anda sudah cukup lengkap!"],
            'specific_feedback': specific_feedback,
            'summary': summary,
            'recommendations': recommendations[:5]
        }
    
    def _extract_key_points(self, text):
        """Extract key points from text"""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        # Return first 10 meaningful sentences
        return sentences[:10]
    
    def _similarity_check(self, text1, text2):
        """Check if two texts are similar"""
        # Simple word overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap > 3
    
    def _generate_specific_feedback(self, answer, best_answer, analysis, scores):
        """Generate specific actionable feedback (INDONESIAN)"""
        feedback_parts = []
        
        # Technical accuracy feedback
        if scores['technical_accuracy'] < 3.5:
            feedback_parts.append(
                "**Kedalaman Teknis:** Jawaban Anda kurang detail teknis. "
                "Sertakan tools, library, dan metodologi spesifik yang Anda gunakan. "
                "Misalnya, daripada bilang 'Saya analisis data', katakan 'Saya pakai pandas "
                "untuk analisis 2 juta records dengan 15 fitur, handle missing values pakai "
                "KNN imputation.'"
            )
        
        # Keyword feedback
        kw_data = analysis['keyword_analysis']
        if kw_data['coverage'] < 50:
            missing = set(kw_data['expected_keywords']) - set(kw_data['found_keywords'])
            feedback_parts.append(
                f"**Konsep Kunci yang Hilang:** Jawaban Anda belum cover konsep penting "
                f"seperti: {', '.join(list(missing)[:5])}. Pastikan address semua aspek "
                f"pertanyaan."
            )
        
        # Structure feedback
        struct_data = analysis['structural']
        if not struct_data['has_examples']:
            feedback_parts.append(
                "**Contoh Konkret:** Tambahkan contoh spesifik dari pengalaman Anda. "
                "Gunakan metode STAR: Situasi (konteks), Tugas (tantangan), "
                "Aksi (yang Anda lakukan), Result (hasil dengan metrik)."
            )
        
        # Quantification feedback
        if not struct_data['has_numbers']:
            feedback_parts.append(
                "**Kuantifikasi Hasil:** Sertakan metrik dan angka spesifik. "
                "Contoh: 'tingkatkan akurasi model 15%', 'kurangi waktu proses "
                "dari 2 jam jadi 15 menit', atau 'analisis dataset dengan 1 juta+ baris'."
            )
        
        # Communication feedback
        if scores['communication_clarity'] < 3.5:
            feedback_parts.append(
                "**Kejelasan:** Jawaban Anda bisa lebih jelas. Pecah ide kompleks "
                "jadi istilah lebih sederhana. Hindari jargon kecuali perlu, dan kalau pakai "
                "istilah teknis, jelaskan singkat dalam konteks bisnis."
            )
        
        # Best practice comparison
        if analysis.get('similarity', {}).get('cosine_similarity', 0) < 0.4:
            feedback_parts.append(
                "**Alignment dengan Best Practices:** Jawaban Anda cukup berbeda "
                "dari expected response. Review contoh jawaban untuk pahami apa yang "
                "interviewer biasanya cari. Fokus pada: dampak bisnis, implementasi teknis, "
                "pendekatan problem-solving, dan hasil terukur."
            )
        
        return "\n\n".join(feedback_parts) if feedback_parts else \
               "Jawaban Anda sudah terstruktur baik dan cover poin-poin kunci dengan efektif!"
    
    def _generate_summary(self, scores, strengths, improvements):
        """Generate overall summary (INDONESIAN)"""
        overall = scores['overall']
        
        if overall >= 4.5:
            return (
                "**Performa Luar Biasa!** Jawaban Anda menunjukkan pengetahuan teknis yang kuat, "
                "komunikasi jelas, dan pengalaman praktis. Anda sudah siap untuk interview "
                "data science. Terus latihan dengan pertanyaan lebih advanced dan fokus "
                "menjelaskan konsep kompleks dengan sederhana."
            )
        elif overall >= 4.0:
            return (
                "**Sangat Bagus!** Anda punya fondasi solid dan komunikasi ide dengan baik. "
                "Improvement kecil di kedalaman teknis atau contoh spesifik akan membuat "
                "jawaban lebih kuat. Fokus pada kuantifikasi achievement dan hubungkan "
                "pekerjaan teknis ke nilai bisnis."
            )
        elif overall >= 3.5:
            return (
                "**Progress Bagus!** Jawaban Anda cover basic dengan baik. Untuk improve, tambah "
                "detail teknis lebih spesifik, contoh konkret dari pengalaman, dan hasil "
                "terukur. Latih menjelaskan pendekatan step-by-step dan hubungkan ke "
                "dampak bisnis."
            )
        elif overall >= 3.0:
            return (
                "**Performa Cukup.** Anda paham konsep tapi perlu lebih detail dan struktur. "
                "Fokus pada: 1) Tambah tools/teknologi spesifik, 2) Berikan contoh konkret, "
                "3) Kuantifikasi hasil, 4) Gunakan metode STAR untuk struktur jawaban."
            )
        else:
            return (
                "**Perlu Improvement.** Jawaban Anda butuh perbaikan signifikan. Area kunci: "
                "1) Pelajari konsep core lebih dalam, 2) Latihan dengan contoh nyata, "
                "3) Belajar struktur jawaban pakai metode STAR, 4) Research pertanyaan "
                "interview umum dan model answer. Jangan berkecil hati - terus latihan "
                "setiap hari!"
            )
    
    def get_percentile_rank(self, score, historical_scores):
        """
        Calculate percentile rank compared to historical scores
        
        Args:
            score (float): Current score
            historical_scores (list): List of previous scores
            
        Returns:
            float: Percentile rank (0-100)
        """
        if not historical_scores:
            return 50.0
        
        below_count = sum(1 for s in historical_scores if s < score)
        percentile = (below_count / len(historical_scores)) * 100
        
        return round(percentile, 1)
    
    def generate_improvement_plan(self, scores, analysis_result):
        """
        Generate personalized improvement plan (INDONESIAN)
        
        Args:
            scores (dict): Score breakdown
            analysis_result (dict): Analysis results
            
        Returns:
            dict: Improvement plan with focus areas
        """
        focus_areas = []
        
        # Identify weakest areas
        component_scores = scores['components']
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
        
        weak_components = [comp for comp, score in sorted_components[:3] if score < 3.5]
        
        improvement_map = {
            'keyword': {
                'area': 'Terminologi Teknis',
                'action': 'Pelajari istilah dan konsep kunci terkait data science',
                'resources': ['Glossary online', 'Blog teknis', 'Dokumentasi']
            },
            'ner': {
                'area': 'Pengetahuan Tools & Teknologi',
                'action': 'Belajar dan praktik dengan tools dan library spesifik',
                'resources': ['Online courses', 'Hands-on projects', 'GitHub repositories']
            },
            'readability': {
                'area': 'Kejelasan Komunikasi',
                'action': 'Latih menulis penjelasan yang jelas dan concise',
                'resources': ['Workshop menulis', 'Panduan technical writing', 'Peer review']
            },
            'structural': {
                'area': 'Organisasi Jawaban',
                'action': 'Gunakan framework seperti metode STAR untuk respon terstruktur',
                'resources': ['Buku persiapan interview', 'Mock interviews', 'Feedback sessions']
            },
            'coherence': {
                'area': 'Alur Logika',
                'action': 'Latih menghubungkan ide dengan kata transisi',
                'resources': ['Latihan menulis', 'Praktik outlining', 'Analisis essay']
            },
            'similarity': {
                'area': 'Cakupan Konten yang Diharapkan',
                'action': 'Research pertanyaan interview umum dan model answer',
                'resources': ['Situs persiapan interview', 'Channel YouTube', 'Mentor feedback']
            }
        }
        
        for comp in weak_components:
            if comp in improvement_map:
                focus_areas.append(improvement_map[comp])
        
        return {
            'priority_areas': focus_areas,
            'overall_recommendation': self._get_overall_recommendation(scores['overall'])
        }
    
    def _get_overall_recommendation(self, overall_score):
        """Get overall recommendation based on score (INDONESIAN)"""
        if overall_score >= 4.5:
            return "Outstanding! Anda sudah interview-ready. Fokus pada topik advanced dan skenario leadership."
        elif overall_score >= 4.0:
            return "Sangat bagus! Polish jawaban dan latihan dengan time pressure."
        elif overall_score >= 3.5:
            return "Fondasi bagus. Kerjakan kedalaman teknis dan berikan contoh konkret."
        elif overall_score >= 3.0:
            return "Start yang decent. Fokus improve akurasi teknis dan kejelasan komunikasi."
        elif overall_score >= 2.5:
            return "Perlu improvement. Review konsep fundamental dan latihan reguler."
        else:
            return "Butuh kerja signifikan. Mulai dari basic dan bangun kompleksitas bertahap."