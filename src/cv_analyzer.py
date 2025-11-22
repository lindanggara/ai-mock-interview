"""
CV Analyzer Module
Extracts skills, experience, and provides personalized recommendations
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
    Analyzes CV to extract relevant information for personalized interviews
    """
    
    def __init__(self):
        # Data Science Skills Dictionary
        self.ds_skills = {
            'programming': [
                'python', 'r', 'sql', 'java', 'scala', 'julia',
                'c++', 'javascript', 'bash', 'shell'
            ],
            'ml_frameworks': [
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
                'xgboost', 'lightgbm', 'catboost', 'h2o', 'mllib'
            ],
            'data_tools': [
                'pandas', 'numpy', 'scipy', 'dask', 'polars',
                'spark', 'hadoop', 'hive', 'pig', 'airflow'
            ],
            'visualization': [
                'matplotlib', 'seaborn', 'plotly', 'bokeh', 'dash',
                'tableau', 'power bi', 'looker', 'd3.js', 'ggplot'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                'elasticsearch', 'neo4j', 'dynamodb', 'snowflake', 'bigquery'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
                's3', 'ec2', 'lambda', 'sagemaker', 'databricks'
            ],
            'ml_techniques': [
                'regression', 'classification', 'clustering', 'deep learning',
                'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
                'random forest', 'gradient boosting', 'ensemble', 'nlp',
                'computer vision', 'time series', 'reinforcement learning'
            ],
            'statistics': [
                'hypothesis testing', 'a/b testing', 'bayesian', 'regression analysis',
                'statistical modeling', 'probability', 'inferential statistics'
            ],
            'mlops': [
                'docker', 'kubernetes', 'mlflow', 'kubeflow', 'ci/cd',
                'jenkins', 'git', 'github', 'gitlab', 'model deployment'
            ]
        }
        
        # Experience level indicators
        self.experience_keywords = {
            'junior': ['intern', 'junior', 'entry', 'associate', 'assistant', '0-2 years', '1 year'],
            'mid': ['analyst', 'scientist', 'engineer', '2-5 years', '3 years', '4 years', 'mid-level'],
            'senior': ['senior', 'lead', 'principal', 'staff', 'architect', '5+ years', 'manager', 'head']
        }
    
    def analyze_cv(self, uploaded_file):
        """
        Main CV analysis function
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Extracted CV data
        """
        # Extract text from file
        text = self.extract_text(uploaded_file)
        
        if not text:
            return {
                'error': 'Could not extract text from CV',
                'skills': [],
                'experience_level': 'Not detected'
            }
        
        # Analyze text
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
        """
        Extract text from PDF or DOCX file
        
        Args:
            uploaded_file: File object
            
        Returns:
            str: Extracted text
        """
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                return self.extract_from_pdf(uploaded_file)
            elif file_extension in ['docx', 'doc']:
                return self.extract_from_docx(uploaded_file)
            else:
                return ""
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def extract_from_pdf(self, uploaded_file):
        """Extract text from PDF"""
        if PyPDF2 is None:
            return "PDF reading not available. Install PyPDF2."
        
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.lower()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_from_docx(self, uploaded_file):
        """Extract text from DOCX"""
        if docx is None:
            return "DOCX reading not available. Install python-docx."
        
        try:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.lower()
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def extract_skills(self, text):
        """
        Extract data science skills from text
        
        Args:
            text (str): CV text
            
        Returns:
            list: Found skills
        """
        found_skills = []
        text_lower = text.lower()
        
        # Check all skill categories
        for category, skills in self.ds_skills.items():
            for skill in skills:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill.title())
        
        # Remove duplicates and sort
        return sorted(list(set(found_skills)))
    
    def categorize_skills(self, skills):
        """
        Categorize extracted skills
        
        Args:
            skills (list): List of skills
            
        Returns:
            dict: Skills by category
        """
        categorized = {
            'Programming': [],
            'ML/DL': [],
            'Data Tools': [],
            'Visualization': [],
            'Cloud/Infrastructure': [],
            'Other': []
        }
        
        skill_lower = [s.lower() for s in skills]
        
        for skill in skills:
            skill_l = skill.lower()
            categorized_flag = False
            
            if skill_l in self.ds_skills['programming']:
                categorized['Programming'].append(skill)
                categorized_flag = True
            
            if skill_l in self.ds_skills['ml_frameworks'] or skill_l in self.ds_skills['ml_techniques']:
                categorized['ML/DL'].append(skill)
                categorized_flag = True
            
            if skill_l in self.ds_skills['data_tools']:
                categorized['Data Tools'].append(skill)
                categorized_flag = True
            
            if skill_l in self.ds_skills['visualization']:
                categorized['Visualization'].append(skill)
                categorized_flag = True
            
            if skill_l in self.ds_skills['cloud'] or skill_l in self.ds_skills['mlops']:
                categorized['Cloud/Infrastructure'].append(skill)
                categorized_flag = True
            
            if not categorized_flag:
                categorized['Other'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def detect_experience_level(self, text):
        """
        Detect experience level from CV text
        
        Args:
            text (str): CV text
            
        Returns:
            str: Experience level
        """
        text_lower = text.lower()
        
        # Count matches for each level
        level_scores = {
            'Junior': 0,
            'Mid-level': 0,
            'Senior': 0
        }
        
        for keyword in self.experience_keywords['junior']:
            if keyword in text_lower:
                level_scores['Junior'] += 1
        
        for keyword in self.experience_keywords['mid']:
            if keyword in text_lower:
                level_scores['Mid-level'] += 1
        
        for keyword in self.experience_keywords['senior']:
            if keyword in text_lower:
                level_scores['Senior'] += 1
        
        # Return level with highest score
        if max(level_scores.values()) == 0:
            return "Mid-level"  # Default
        
        return max(level_scores, key=level_scores.get)
    
    def extract_experience_years(self, text):
        """
        Extract years of experience from text
        
        Args:
            text (str): CV text
            
        Returns:
            str: Years of experience
        """
        # Patterns for experience
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:\s*(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                if len(match.groups()) == 1:
                    return f"{match.group(1)} years"
                else:
                    return f"{match.group(1)}-{match.group(2)} years"
        
        return "Not specified"
    
    def extract_education(self, text):
        """
        Extract education information
        
        Args:
            text (str): CV text
            
        Returns:
            list: Education degrees
        """
        degrees = []
        degree_keywords = [
            'bachelor', 'b.s.', 'b.sc', 'b.tech', 'ba', 'bs',
            'master', 'm.s.', 'm.sc', 'm.tech', 'ma', 'ms', 'mba',
            'phd', 'ph.d', 'doctorate', 'doctoral'
        ]
        
        text_lower = text.lower()
        
        for degree in degree_keywords:
            if degree in text_lower:
                degrees.append(degree.upper())
        
        return list(set(degrees)) if degrees else ['Not specified']
    
    def generate_recommendations(self, skills, experience_level):
        """
        Generate personalized recommendations based on CV
        
        Args:
            skills (list): Detected skills
            experience_level (str): Experience level
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        # Check for essential skills
        has_python = any('python' in s.lower() for s in skills)
        has_ml = any(ml in s.lower() for s in skills for ml in ['tensorflow', 'pytorch', 'scikit', 'xgboost'])
        has_sql = any('sql' in s.lower() for s in skills)
        has_cloud = any(cloud in s.lower() for s in skills for cloud in ['aws', 'azure', 'gcp'])
        
        # Python recommendations
        if not has_python:
            recommendations.append("Consider highlighting Python skills - it's essential for DS roles")
        
        # ML framework recommendations
        if not has_ml:
            recommendations.append("Add ML frameworks experience (TensorFlow, PyTorch, scikit-learn)")
        
        # SQL recommendations
        if not has_sql:
            recommendations.append("SQL is crucial - make sure to mention database experience")
        
        # Cloud recommendations
        if not has_cloud and experience_level != 'Junior':
            recommendations.append("Cloud experience (AWS/Azure/GCP) is valuable for modern DS roles")
        
        # Level-specific recommendations
        if experience_level == 'Junior':
            recommendations.append("Focus on showcasing projects and hands-on experience")
            recommendations.append("Highlight any internships or academic projects")
        elif experience_level == 'Mid-level':
            recommendations.append("Emphasize end-to-end project delivery and business impact")
            recommendations.append("Quantify your achievements (X% improvement, $Y saved)")
        else:  # Senior
            recommendations.append("Highlight leadership, mentoring, and strategic contributions")
            recommendations.append("Focus on system design and architecture decisions")
        
        return recommendations[:5]  # Top 5 recommendations
