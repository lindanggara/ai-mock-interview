"""
Modul Analisis CV
Mengekstraksi keahlian, pengalaman, dan memberikan rekomendasi yang dipersonalisasi (Bahasa Indonesia)
"""

import re
from collections import Counter
import io

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


class CVAnalyzer:
    """
    Menganalisis CV untuk mengekstraksi informasi penting dan memberikan saran yang relevan
    """

    def __init__(self):
        # Daftar keahlian dalam bidang Data Science
        self.ds_skills = {
            'programming': ['python', 'r', 'sql', 'java', 'scala', 'julia', 'c++', 'javascript', 'bash', 'shell'],
            'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'h2o', 'mllib'],
            'data_tools': ['pandas', 'numpy', 'scipy', 'dask', 'polars', 'spark', 'hadoop', 'hive', 'pig', 'airflow'],
            'visualization': ['matplotlib', 'seaborn', 'plotly', 'bokeh', 'dash', 'tableau', 'power bi', 'looker', 'd3.js', 'ggplot'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'neo4j', 'dynamodb', 'snowflake', 'bigquery'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 's3', 'ec2', 'lambda', 'sagemaker', 'databricks'],
            'ml_techniques': ['regression', 'classification', 'clustering', 'deep learning', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'random forest', 'gradient boosting', 'ensemble', 'nlp', 'computer vision', 'time series', 'reinforcement learning'],
            'statistics': ['hypothesis testing', 'a/b testing', 'bayesian', 'regression analysis', 'statistical modeling', 'probability', 'inferential statistics'],
            'mlops': ['docker', 'kubernetes', 'mlflow', 'kubeflow', 'ci/cd', 'jenkins', 'git', 'github', 'gitlab', 'model deployment']
        }

        # Indikator level pengalaman
        self.experience_keywords = {
            'junior': ['intern', 'junior', 'entry', 'associate', 'assistant', '0-2 years', '1 year'],
            'mid': ['analyst', 'scientist', 'engineer', '2-5 years', '3 years', '4 years', 'mid-level'],
            'senior': ['senior', 'lead', 'principal', 'staff', 'architect', '5+ years', 'manager', 'head']
        }

    def analyze_cv(self, uploaded_file):
        """Fungsi utama untuk menganalisis CV"""
        text = self.extract_text(uploaded_file)

        if not text:
            return {
                'error': 'Tidak dapat mengekstrak teks dari CV',
                'skills': [],
                'experience_level': 'Tidak terdeteksi'
            }

        skills = self.extract_skills(text)
        experience_level = self.detect_experience_level(text)
        experience_years = self.extract_experience_years(text)
        education = self.extract_education(text)

        return {
            'skills': skills,
            'experience_level': experience_level,
            'experience_years': experience_years,
            'education': education,
            'skill_categories': self.categorize_skills(skills),
            'recommendations': self.generate_recommendations(skills, experience_level)
        }

    def extract_text(self, uploaded_file):
        """Ekstraksi teks dari file PDF atau DOCX"""
        file_name = getattr(uploaded_file, 'name', None) or 'temp.pdf'
        file_extension = file_name.split('.')[-1].lower()

        try:
            uploaded_file.seek(0)
            if file_extension == 'pdf':
                return self.extract_from_pdf(uploaded_file)
            elif file_extension in ['docx', 'doc']:
                return self.extract_from_docx(uploaded_file)
            else:
                return ""
        except Exception as e:
            print(f"Kesalahan saat ekstraksi teks: {e}")
            return ""

    def extract_from_pdf(self, uploaded_file):
        if PyPDF2 is None:
            return "Pembacaan PDF tidak tersedia. Install PyPDF2."

        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = "".join([page.extract_text() or '' for page in pdf_reader.pages])
            return text.lower()
        except Exception as e:
            return f"Kesalahan membaca PDF: {str(e)}"

    def extract_from_docx(self, uploaded_file):
        if docx is None:
            return "Pembacaan DOCX tidak tersedia. Install python-docx."

        try:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.lower()
        except Exception as e:
            return f"Kesalahan membaca DOCX: {str(e)}"

    def extract_skills(self, text):
        """Mengekstraksi keahlian yang disebutkan di CV"""
        found_skills = []
        text_lower = text.lower()
        for category, skills in self.ds_skills.items():
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill.title())
        return sorted(list(set(found_skills)))

    def categorize_skills(self, skills):
        """Mengelompokkan keahlian berdasarkan kategori"""
        categorized = {
            'Pemrograman': [],
            'ML/DL': [],
            'Alat Data': [],
            'Visualisasi': [],
            'Cloud/Infra': [],
            'Lainnya': []
        }

        for skill in skills:
            skill_l = skill.lower()
            if skill_l in self.ds_skills['programming']:
                categorized['Pemrograman'].append(skill)
            elif skill_l in self.ds_skills['ml_frameworks'] or skill_l in self.ds_skills['ml_techniques']:
                categorized['ML/DL'].append(skill)
            elif skill_l in self.ds_skills['data_tools']:
                categorized['Alat Data'].append(skill)
            elif skill_l in self.ds_skills['visualization']:
                categorized['Visualisasi'].append(skill)
            elif skill_l in self.ds_skills['cloud'] or skill_l in self.ds_skills['mlops']:
                categorized['Cloud/Infra'].append(skill)
            else:
                categorized['Lainnya'].append(skill)

        return {k: v for k, v in categorized.items() if v}

    def detect_experience_level(self, text):
        """Mendeteksi level pengalaman berdasarkan isi CV"""
        text_lower = text.lower()
        level_scores = {'Junior': 0, 'Mid-level': 0, 'Senior': 0}

        for level, keywords in self.experience_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'junior':
                        level_scores['Junior'] += 1
                    elif level == 'mid':
                        level_scores['Mid-level'] += 1
                    elif level == 'senior':
                        level_scores['Senior'] += 1

        if max(level_scores.values()) == 0:
            return 'Mid-level'
        return max(level_scores, key=level_scores.get)

    def extract_experience_years(self, text):
        """Mengekstraksi jumlah tahun pengalaman kerja"""
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?'
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                if len(match.groups()) == 1:
                    return f"{match.group(1)} tahun"
                else:
                    return f"{match.group(1)}-{match.group(2)} tahun"
        return "Tidak disebutkan"

    def extract_education(self, text):
        """Mengekstraksi tingkat pendidikan"""
        degrees = []
        degree_keywords = ['bachelor', 'b.s.', 'b.sc', 'b.tech', 'ba', 'bs', 'master', 'm.s.', 'm.sc', 'm.tech', 'ma', 'ms', 'mba', 'phd', 'ph.d', 'doctorate', 'doctoral']
        text_lower = text.lower()
        for degree in degree_keywords:
            if degree in text_lower:
                degrees.append(degree.upper())
        return list(set(degrees)) if degrees else ['Tidak disebutkan']

    def generate_recommendations(self, skills, experience_level):
        """Memberikan saran perbaikan CV berdasarkan hasil analisis"""
        rekomendasi = []

        has_python = any('python' in s.lower() for s in skills)
        has_ml = any(ml in s.lower() for s in skills for ml in ['tensorflow', 'pytorch', 'scikit', 'xgboost'])
        has_sql = any('sql' in s.lower() for s in skills)
        has_cloud = any(cloud in s.lower() for s in skills for cloud in ['aws', 'azure', 'gcp'])

        if not has_python:
            rekomendasi.append("Sertakan keahlian Python — sangat penting untuk peran Data Science.")
        if not has_ml:
            rekomendasi.append("Tambahkan pengalaman menggunakan framework ML seperti TensorFlow, PyTorch, atau scikit-learn.")
        if not has_sql:
            rekomendasi.append("Cantumkan pengalaman dengan SQL — kemampuan dasar untuk analisis data.")
        if not has_cloud and experience_level != 'Junior':
            rekomendasi.append("Pengalaman dengan cloud (AWS/Azure/GCP) sangat berharga untuk peran modern.")

        if experience_level == 'Junior':
            rekomendasi.append("Fokus pada proyek nyata dan pengalaman hands-on.")
            rekomendasi.append("Tunjukkan pengalaman magang atau proyek akademik.")
        elif experience_level == 'Mid-level':
            rekomendasi.append("Tunjukkan kontribusi dalam proyek end-to-end dan dampak bisnisnya.")
            rekomendasi.append("Gunakan data kuantitatif untuk menunjukkan hasil kerja.")
        else:
            rekomendasi.append("Soroti kemampuan kepemimpinan dan kontribusi strategis.")
            rekomendasi.append("Tampilkan pengalaman dalam desain sistem dan keputusan arsitektur.")

        return rekomendasi[:5]