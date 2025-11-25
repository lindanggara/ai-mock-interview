"""
Advanced Text Mining Module
Implements comprehensive NLP algorithms for interview answer analysis
"""

import re
import string
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextMiningAnalyzer:
    """
    Comprehensive text mining analyzer for interview answers
    """
    
    def __init__(self, stopwords=None):
        """
        Initialize analyzer with stopwords
        
        Args:
            stopwords (set): Set of stopwords to filter
        """
        self.stopwords = stopwords if stopwords else set()
        
        # Data Science specific entities
        self.ds_entities = {
            'tools': ['python', 'r', 'sql', 'tableau', 'power bi', 'excel', 
                     'spark', 'hadoop', 'airflow', 'kafka', 'docker', 'kubernetes'],
            'libraries': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 
                         'pytorch', 'keras', 'matplotlib', 'seaborn', 'plotly',
                         'scipy', 'statsmodels', 'xgboost', 'lightgbm'],
            'methods': ['regression', 'classification', 'clustering', 'deep learning',
                       'random forest', 'xgboost', 'neural network', 'svm', 
                       'decision tree', 'ensemble', 'gradient boosting', 'kmeans',
                       'pca', 'dimensionality reduction', 'feature engineering'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'f1-score',
                       'rmse', 'mae', 'r2', 'r-squared', 'auc', 'roc', 
                       'confusion matrix', 'mse', 'cross-validation']
        }
    
    def preprocess_text(self, text, remove_stopwords=True):
        """
        Preprocess text: lowercase, remove punctuation, tokenize
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            list: List of cleaned tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def quick_analysis(self, answer, expected_keywords):
        """
        Quick statistical analysis
        
        Args:
            answer (str): User's answer
            expected_keywords (list): Expected keywords
            
        Returns:
            dict: Quick statistics
        """
        words = answer.split()
        sentences = sent_tokenize(answer)
        
        # Count keywords
        answer_lower = answer.lower()
        keywords_found = sum(1 for kw in expected_keywords if kw in answer_lower)
        keyword_coverage = (keywords_found / len(expected_keywords) * 100) if expected_keywords else 0
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'keywords_found': keywords_found,
            'keyword_coverage': keyword_coverage
        }
    
    def keyword_analysis(self, answer, expected_keywords):
        """
        Analyze keyword coverage and relevance
        
        Args:
            answer (str): User's answer
            expected_keywords (list): Expected keywords
            
        Returns:
            dict: Keyword analysis results
        """
        answer_lower = answer.lower()
        
        found_keywords = [kw for kw in expected_keywords if kw in answer_lower]
        coverage = (len(found_keywords) / len(expected_keywords) * 100) if expected_keywords else 0
        
        # Calculate keyword density
        total_words = len(answer.split())
        keyword_density = (len(found_keywords) / total_words * 100) if total_words > 0 else 0
        
        return {
            'expected_keywords': expected_keywords,
            'found_keywords': found_keywords,
            'coverage': coverage,
            'keyword_density': keyword_density,
            'score': min(coverage / 20, 5.0)
        }
    
    def tfidf_analysis(self, answer, reference_texts=None):
        """
        TF-IDF analysis to identify important terms
        
        Args:
            answer (str): User's answer
            reference_texts (list): Optional reference texts for comparison
            
        Returns:
            dict: TF-IDF analysis results
        """
        # Prepare corpus
        if reference_texts:
            corpus = [answer] + reference_texts
        else:
            corpus = [answer]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=list(self.stopwords) if self.stopwords else None,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores for the answer
            answer_scores = tfidf_matrix[0].toarray().flatten()
            
            # Get top terms
            top_indices = answer_scores.argsort()[-10:][::-1]
            top_terms = [(feature_names[i], answer_scores[i]) for i in top_indices if answer_scores[i] > 0]
            
            # Calculate statistics
            tokens = self.preprocess_text(answer, remove_stopwords=False)
            unique_tokens = set(tokens)
            
            lexical_diversity = len(unique_tokens) / len(tokens) if tokens else 0
            
            # Technical term density (terms with high TF-IDF)
            technical_terms = [term for term, score in top_terms if score > 0.1]
            technical_density = (len(technical_terms) / len(tokens) * 100) if tokens else 0
            
            return {
                'top_terms': top_terms,
                'total_words': len(tokens),
                'unique_words': len(unique_tokens),
                'lexical_diversity': lexical_diversity,
                'technical_density': technical_density,
                'score': min((lexical_diversity + technical_density / 20) * 2, 5.0)
            }
        except Exception as e:
            # Fallback if TF-IDF fails
            tokens = self.preprocess_text(answer)
            return {
                'top_terms': [],
                'total_words': len(tokens),
                'unique_words': len(set(tokens)),
                'lexical_diversity': 0,
                'technical_density': 0,
                'score': 0
            }
    
    def calculate_cosine_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            dict: Similarity results
        """
        if not text1 or not text2:
            return {
                'cosine_similarity': 0.0,
                'interpretation': 'No comparison available',
                'common_terms_count': 0
            }
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words=list(self.stopwords) if self.stopwords else None)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Find common terms
            tokens1 = set(self.preprocess_text(text1))
            tokens2 = set(self.preprocess_text(text2))
            common_terms = tokens1.intersection(tokens2)
            
            # Interpretation
            if similarity >= 0.7:
                interpretation = "Excellent alignment with best practices"
            elif similarity >= 0.5:
                interpretation = "Good alignment with expected answer"
            elif similarity >= 0.3:
                interpretation = "Moderate alignment, room for improvement"
            else:
                interpretation = "Low alignment, consider covering more key concepts"
            
            return {
                'cosine_similarity': float(similarity),
                'interpretation': interpretation,
                'common_terms_count': len(common_terms),
                'score': similarity * 5.0
            }
        except Exception as e:
            return {
                'cosine_similarity': 0.0,
                'interpretation': 'Similarity calculation failed',
                'common_terms_count': 0,
                'score': 0.0
            }
    
    def ngram_analysis(self, answer, n_range=(2, 3)):
        """
        Extract and analyze n-grams
        
        Args:
            answer (str): User's answer
            n_range (tuple): Range of n-gram sizes
            
        Returns:
            dict: N-gram analysis results
        """
        tokens = self.preprocess_text(answer)
        
        results = {}
        
        # Bigrams
        if 2 in range(n_range[0], n_range[1] + 1):
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = Counter(bigrams)
            results['bigrams'] = bigram_freq.most_common(10)
        
        # Trigrams
        if 3 in range(n_range[0], n_range[1] + 1):
            trigrams = list(ngrams(tokens, 3))
            trigram_freq = Counter(trigrams)
            results['trigrams'] = trigram_freq.most_common(10)
        
        # Calculate phrase richness
        total_bigrams = len(set(bigrams)) if 'bigrams' in results else 0
        total_trigrams = len(set(trigrams)) if 'trigrams' in results else 0
        phrase_richness = (total_bigrams + total_trigrams) / len(tokens) if tokens else 0
        
        results['phrase_richness'] = phrase_richness
        results['score'] = min(phrase_richness * 10, 5.0)
        
        return results
    
    def named_entity_recognition(self, answer):
        """
        Extract named entities (tools, libraries, methods, metrics)
        
        Args:
            answer (str): User's answer
            
        Returns:
            dict: Named entities found
        """
        answer_lower = answer.lower()
        
        found_entities = {
            'tools': [t for t in self.ds_entities['tools'] if t in answer_lower],
            'libraries': [l for l in self.ds_entities['libraries'] if l in answer_lower],
            'methods': [m for m in self.ds_entities['methods'] if m in answer_lower],
            'metrics': [m for m in self.ds_entities['metrics'] if m in answer_lower]
        }
        
        total_entities = sum(len(v) for v in found_entities.values())
        
        # Score based on diversity and quantity
        diversity_score = len([k for k, v in found_entities.items() if len(v) > 0])
        quantity_score = min(total_entities / 5, 3)
        
        ner_score = min((diversity_score * 0.8 + quantity_score * 0.2) * 1.2, 5.0)
        
        found_entities['total'] = total_entities
        found_entities['diversity'] = diversity_score
        found_entities['score'] = ner_score
        
        return found_entities
    
    def sentiment_analysis(self, answer):
        """
        Comprehensive keyword-based sentiment analysis for Indonesian + English mixed text
        Lebih reliable daripada TextBlob untuk mixed language
        """
        try:
            answer_lower = answer.lower()
            
            # ==================== POSITIVE KEYWORDS ====================
            positive_keywords = [
                # Indonesian - Achievement & Success
                'berhasil', 'sukses', 'meningkat', 'efektif', 'optimal', 'bagus',
                'baik', 'positif', 'untung', 'profit', 'naik', 'tinggi', 'unggul',
                'hebat', 'luar biasa', 'signifikan', 'peningkatan', 'mencapai',
                'membantu', 'solutif', 'inovatif', 'efisien', 'produktif', 'cemerlang',
                'sempurna', 'maksimal', 'terbaik', 'unggulan', 'andal', 'handal',
                'lancar', 'cepat', 'akurat', 'presisi', 'tepat', 'sesuai', 'cocok',
                
                # Indonesian - Growth & Improvement
                'bertumbuh', 'berkembang', 'maju', 'membaik', 'meningkatkan',
                'memperbaiki', 'mengoptimalkan', 'memaksimalkan', 'menyempurnakan',
                'menyelesaikan', 'mengatasi', 'solusi', 'teratasi', 'terselesaikan',
                
                # Indonesian - Business Impact
                'menghemat', 'hemat', 'efisiensi', 'revenue', 'pendapatan', 'omset',
                'keuntungan', 'laba', 'roi', 'return', 'valuable', 'bernilai',
                'berharga', 'menguntungkan', 'profitable', 'skalabel', 'scalable',
                
                # English - Achievement & Success
                'success', 'successful', 'achieve', 'achieved', 'accomplish',
                'accomplished', 'improve', 'improved', 'improvement', 'increase',
                'increased', 'effective', 'efficiently', 'optimal', 'optimized',
                'excellent', 'great', 'outstanding', 'exceptional', 'superior',
                'amazing', 'fantastic', 'wonderful', 'remarkable', 'impressive',
                
                # English - Growth & Performance
                'boost', 'boosted', 'enhance', 'enhanced', 'growth', 'grow',
                'gain', 'gained', 'advance', 'advanced', 'progress', 'progressed',
                'upgrade', 'upgraded', 'better', 'best', 'top', 'leading', 'peak',
                
                # English - Business Impact
                'profit', 'profitable', 'revenue', 'save', 'saved', 'savings',
                'reduce cost', 'cost reduction', 'valuable', 'value', 'benefit',
                'advantageous', 'breakthrough', 'innovative', 'innovation',
                
                # English - Technical Success
                'accurate', 'accuracy', 'precise', 'precision', 'reliable',
                'robust', 'stable', 'scalable', 'high performance', 'fast',
                'efficient', 'streamline', 'streamlined', 'automate', 'automated'
            ]
            
            # ==================== NEGATIVE KEYWORDS ====================
            negative_keywords = [
                # Indonesian - Failure & Problems
                'gagal', 'kegagalan', 'turun', 'menurun', 'buruk', 'jelek',
                'sulit', 'kesulitan', 'masalah', 'kendala', 'hambatan', 'halangan',
                'rintangan', 'tantangan', 'kurang', 'kekurangan', 'rendah', 'lemah',
                'tidak', 'belum', 'minus', 'rugi', 'kerugian', 'loss', 'berkurang',
                'penurunan', 'merosot', 'anjlok', 'jatuh', 'drop',
                
                # Indonesian - Uncertainty & Risk
                'ragu', 'bingung', 'tidak yakin', 'tidak pasti', 'risiko', 'bahaya',
                'mengkhawatirkan', 'khawatir', 'was-was', 'takut', 'cemas',
                
                # Indonesian - Quality Issues
                'error', 'bug', 'cacat', 'rusak', 'salah', 'keliru', 'menyimpang',
                'meleset', 'tidak akurat', 'bias', 'overfitting', 'underfitting',
                'lambat', 'lelet', 'hang', 'crash', 'down', 'downtime',
                
                # English - Failure & Problems
                'fail', 'failed', 'failure', 'decrease', 'decreased', 'decline',
                'declined', 'drop', 'dropped', 'fall', 'fell', 'poor', 'bad',
                'worse', 'worst', 'terrible', 'awful', 'horrible', 'weak',
                'difficult', 'difficulty', 'problem', 'issue', 'trouble',
                'challenge', 'challenging', 'obstacle', 'barrier', 'bottleneck',
                
                # English - Uncertainty & Risk
                'uncertain', 'unsure', 'doubt', 'doubtful', 'risk', 'risky',
                'danger', 'dangerous', 'concern', 'concerning', 'worry', 'worried',
                
                # English - Quality Issues
                'error', 'bug', 'buggy', 'broken', 'crash', 'crashed', 'wrong',
                'incorrect', 'inaccurate', 'imprecise', 'unreliable', 'unstable',
                'slow', 'lag', 'latency', 'delay', 'timeout', 'downtime',
                'overfitting', 'underfitting', 'biased', 'skewed', 'noisy',
                
                # English - Loss & Negative Impact
                'loss', 'lose', 'losing', 'waste', 'wasted', 'inefficient',
                'inefficiency', 'costly', 'expensive', 'leak', 'leakage'
            ]
            
            # ==================== TECHNICAL KEYWORDS ====================
            technical_keywords = [
                # Programming Languages
                'python', 'r', 'sql', 'java', 'scala', 'julia', 'javascript',
                'typescript', 'c++', 'cpp', 'go', 'rust', 'bash', 'shell',
                
                # ML/DL Frameworks
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
                'scikit', 'xgboost', 'lightgbm', 'catboost', 'h2o', 'mllib',
                'fastai', 'huggingface', 'transformers', 'bert', 'gpt',
                
                # Data Tools
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
                'dask', 'polars', 'pyspark', 'spark', 'hadoop', 'hive', 'pig',
                'airflow', 'dagster', 'prefect', 'kafka', 'flink', 'storm',
                
                # Databases
                'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra',
                'elasticsearch', 'neo4j', 'dynamodb', 'snowflake', 'bigquery',
                'redshift', 'oracle', 'mssql', 'sqlite', 'mariadb',
                
                # Cloud Platforms
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
                's3', 'ec2', 'lambda', 'sagemaker', 'databricks', 'emr',
                'glue', 'athena', 'kinesis', 'cloudformation', 'terraform',
                
                # ML/DL Algorithms
                'regression', 'logistic regression', 'linear regression',
                'classification', 'clustering', 'k-means', 'dbscan', 'hierarchical',
                'random forest', 'decision tree', 'gradient boosting', 'adaboost',
                'svm', 'support vector', 'naive bayes', 'knn', 'k-nearest',
                'neural network', 'deep learning', 'cnn', 'convolutional',
                'rnn', 'recurrent', 'lstm', 'gru', 'transformer', 'attention',
                'autoencoder', 'gan', 'generative', 'reinforcement learning',
                
                # ML Techniques
                'feature engineering', 'feature selection', 'dimensionality reduction',
                'pca', 'principal component', 't-sne', 'umap', 'cross validation',
                'hyperparameter tuning', 'grid search', 'random search', 'bayesian optimization',
                'ensemble', 'bagging', 'boosting', 'stacking', 'blending',
                'regularization', 'l1', 'l2', 'lasso', 'ridge', 'elastic net',
                'dropout', 'batch normalization', 'data augmentation', 'transfer learning',
                
                # NLP/CV Specific
                'nlp', 'natural language processing', 'tokenization', 'embedding',
                'word2vec', 'glove', 'fasttext', 'tfidf', 'tf-idf', 'bow',
                'sentiment analysis', 'named entity', 'ner', 'pos tagging',
                'computer vision', 'object detection', 'image classification',
                'segmentation', 'yolo', 'rcnn', 'maskrcnn', 'unet', 'resnet',
                'vgg', 'inception', 'mobilenet', 'efficientnet',
                
                # Metrics (Indo + English)
                'akurasi', 'accuracy', 'precision', 'recall', 'f1-score', 'f1',
                'auc', 'roc', 'rmse', 'mse', 'mae', 'r-squared', 'r2', 'mape',
                'confusion matrix', 'classification report', 'loss', 'epoch',
                
                # Business Metrics (Indo + English)
                'kpi', 'roi', 'revenue', 'profit', 'margin', 'conversion',
                'churn', 'retention', 'ctr', 'click-through', 'engagement',
                'user', 'customer', 'pelanggan', 'transaksi', 'transaction',
                
                # Numbers & Scale (Indo + English)
                '%', 'persen', 'percent', 'juta', 'million', 'miliar', 'billion',
                'ribu', 'thousand', 'ratus', 'hundred', 'triliun', 'trillion',
                
                # MLOps & Deployment
                'docker', 'kubernetes', 'k8s', 'mlflow', 'kubeflow', 'ci/cd',
                'jenkins', 'github actions', 'gitlab ci', 'deployment', 'deploy',
                'api', 'rest api', 'fastapi', 'flask', 'django', 'endpoint',
                'model serving', 'inference', 'prediction', 'monitoring',
                'logging', 'metrics', 'observability', 'a/b test', 'experiment',
                
                # Data Engineering
                'etl', 'elt', 'data pipeline', 'data warehouse', 'data lake',
                'batch processing', 'stream processing', 'real-time', 'realtime',
                'data quality', 'data validation', 'schema', 'partitioning'
            ]
            
            # Count keywords
            pos_count = sum(1 for word in positive_keywords if word in answer_lower)
            neg_count = sum(1 for word in negative_keywords if word in answer_lower)
            tech_count = sum(1 for word in technical_keywords if word in answer_lower)
            
            # Calculate polarity (-1 to 1)
            total_sentiment = pos_count + neg_count
            if total_sentiment > 0:
                polarity = (pos_count - neg_count) / total_sentiment
            else:
                polarity = 0.0
            
            # Calculate subjectivity (0 to 1)
            # More technical terms = less subjective (more objective/factual)
            word_count = len(answer.split())
            if word_count > 0:
                tech_ratio = min(tech_count / word_count, 1.0)
                subjectivity = max(0.3, 1.0 - tech_ratio)  # At least 0.3
            else:
                subjectivity = 0.5
            
            # Determine polarity label
            if polarity > 0.15:
                polarity_label = "Positive & Confident"
            elif polarity < -0.15:
                polarity_label = "Negative/Uncertain"
            else:
                polarity_label = "Neutral"
            
            # Score calculation
            sentiment_score = ((polarity + 1) / 2) * 5  # -1 to 1 → 0 to 5
            balance_score = (1 - abs(subjectivity - 0.4)) * 5  # Ideal subjectivity ~0.4
            overall_score = (sentiment_score * 0.6 + balance_score * 0.4)
            
            return {
                'original_text': answer,
                'polarity': float(polarity),
                'subjectivity': float(subjectivity),
                'polarity_label': polarity_label,
                'sentiment_score': sentiment_score,
                'balance_score': balance_score,
                'score': overall_score,
                'positive_keywords_found': pos_count,
                'negative_keywords_found': neg_count,
                'technical_keywords_found': tech_count
            }
        
        except Exception as e:
            # Fallback
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'polarity_label': 'Neutral',
                'sentiment_score': 2.5,
                'balance_score': 2.5,
                'score': 2.5,
                'positive_keywords_found': 0,
                'negative_keywords_found': 0,
                'technical_keywords_found': 0
            }
        
    
    def readability_analysis(self, answer):
        """
        Analyze readability and structure
        
        Args:
            answer (str): User's answer
            
        Returns:
            dict: Readability metrics
        """
        sentences = sent_tokenize(answer)
        words = answer.split()
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        if sentence_count == 0 or word_count == 0:
            return {
                'score': 0,
                'avg_sentence_length': 0,
                'sentence_count': 0,
                'word_count': 0,
                'assessment': 'Too short'
            }
        
        avg_sentence_length = word_count / sentence_count
        
        # Ideal: 15-20 words per sentence for professional communication
        if 15 <= avg_sentence_length <= 20:
            readability_score = 5.0
            assessment = "Excellent"
        elif 12 <= avg_sentence_length <= 25:
            readability_score = 4.0
            assessment = "Good"
        elif 10 <= avg_sentence_length <= 30:
            readability_score = 3.0
            assessment = "Fair"
        elif 8 <= avg_sentence_length <= 35:
            readability_score = 2.5
            assessment = "Needs improvement"
        else:
            readability_score = 2.0
            assessment = "Poor structure"
        
        # Calculate variance in sentence length (good writing has variation)
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Moderate variance is good
        if 10 <= length_variance <= 40:
            variance_bonus = 0.3
        else:
            variance_bonus = 0
        
        return {
            'score': min(readability_score + variance_bonus, 5.0),
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'length_variance': float(length_variance),
            'assessment': assessment
        }
    
    def structural_analysis(self, answer, ideal_length):
        """
        Analyze answer structure and organization
        
        Args:
            answer (str): User's answer
            ideal_length (tuple): (min_words, max_words)
            
        Returns:
            dict: Structural analysis results
        """
        word_count = len(answer.split())
        min_len, max_len = ideal_length
        
        # Length score
        if min_len <= word_count <= max_len:
            length_score = 5.0
            length_appropriate = True
        elif word_count < min_len:
            length_score = (word_count / min_len) * 5.0
            length_appropriate = False
        else:
            # Penalty for being too long
            excess = word_count - max_len
            penalty = min(excess / max_len * 2, 2)
            length_score = max(5.0 - penalty, 2.0)
            length_appropriate = False
        
        # Check for examples/specific cases
        example_indicators = [
            'contoh', 'example', 'misalnya', 'seperti', 'instance', 'case',
            'project', 'pengalaman', 'saya pernah', 'pada saat', 'ketika',
            'for example', 'for instance', 'such as'
        ]
        has_examples = any(indicator in answer.lower() for indicator in example_indicators)
        
        # Check for structure (paragraphs, organization)
        has_structure = '\n' in answer or len(sent_tokenize(answer)) > 3
        
        # Check for quantitative mentions (shows concrete results)
        has_numbers = bool(re.search(r'\d+', answer))
        
        # Bonus points
        example_bonus = 0.5 if has_examples else 0
        structure_bonus = 0.3 if has_structure else 0
        numbers_bonus = 0.2 if has_numbers else 0
        
        total_score = min(length_score + example_bonus + structure_bonus + numbers_bonus, 5.0)
        
        return {
            'length_score': length_score,
            'length_appropriate': length_appropriate,
            'has_examples': has_examples,
            'has_structure': has_structure,
            'has_numbers': has_numbers,
            'score': total_score
        }
    
    def coherence_analysis(self, answer):
        """
        Analyze coherence and logical flow
        
        Args:
            answer (str): User's answer
            
        Returns:
            dict: Coherence metrics
        """
        sentences = sent_tokenize(answer)
        
        if len(sentences) < 2:
            return {
                'score': 2.5,
                'sentence_connections': 0,
                'transition_words_count': 0
            }
        
        # Check for transition/connection words
        transition_words = [
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'additionally', 'similarly', 'in contrast', 'for example',
            'specifically', 'first', 'second', 'finally', 'sebagai tambahan',
            'oleh karena itu', 'selain itu', 'dengan demikian', 'misalnya'
        ]
        
        answer_lower = answer.lower()
        transition_count = sum(1 for tw in transition_words if tw in answer_lower)
        
        # Calculate lexical cohesion (repeated important terms)
        tokens = self.preprocess_text(answer)
        word_freq = Counter(tokens)
        repeated_terms = sum(1 for count in word_freq.values() if count > 1)
        
        # Score based on transitions and cohesion
        transition_score = min(transition_count / len(sentences) * 5, 3)
        cohesion_score = min(repeated_terms / len(set(tokens)) * 5, 2) if tokens else 0
        
        coherence_score = min(transition_score + cohesion_score, 5.0)
        
        return {
            'score': coherence_score,
            'sentence_connections': transition_count,
            'transition_words_count': transition_count,
            'repeated_terms': repeated_terms
        }
    
    def comprehensive_analysis(self, answer, question_data, best_answer="", category_keywords=None):
        """
        Run all text mining analyses
        
        Args:
            answer (str): User's answer
            question_data (dict): Question metadata
            best_answer (str): Reference best answer
            category_keywords (list): Additional category keywords
            
        Returns:
            dict: Comprehensive analysis results
        """
        all_keywords = question_data['keywords']
        if category_keywords:
            all_keywords = list(set(all_keywords + category_keywords))
        
        return {
            'keyword_analysis': self.keyword_analysis(answer, all_keywords),
            'tfidf': self.tfidf_analysis(answer, [best_answer] if best_answer else None),
            'similarity': self.calculate_cosine_similarity(answer, best_answer) if best_answer else {},
            'ngrams': self.ngram_analysis(answer),
            'ner': self.named_entity_recognition(answer),
            'sentiment': self.sentiment_analysis(answer),
            'readability': self.readability_analysis(answer),
            'structural': self.structural_analysis(answer, question_data['ideal_length']),
            'coherence': self.coherence_analysis(answer)
        }
