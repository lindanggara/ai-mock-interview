"""
Scoring Engine Module
Calculates final scores and generates feedback
"""

import re
from difflib import SequenceMatcher


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

    # ------------------------------
    #  FIXED MISSING FUNCTIONS
    # ------------------------------

    def _extract_key_points(self, text):
        """
        Extract key points by splitting the text into meaningful pieces.
        This is a simple rule-based extractor designed to avoid errors.
        """
        if not text:
            return []

        # Split by punctuation
        parts = re.split(r'[.;:\n]', text)

        # Clean & filter
        points = []
        for p in parts:
            cleaned = p.strip().lower()
            if len(cleaned.split()) >= 3:  # at least 3 words
                points.append(cleaned)

        return points[:15]  # limit to avoid overflow

    def _similarity_check(self, a, b, threshold=0.55):
        """
        Fuzzy similarity check using SequenceMatcher.
        Returns True if similarity above threshold.
        """
        if not a or not b:
            return False

        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio >= threshold

    # ------------------------------
    #  ORIGINAL FUNCTIONS (UNCHANGED)
    # ------------------------------

    def calculate_scores(self, analysis_result, question_weights, difficulty='Mid-level'):
        # Extract component scores
        keyword_score = analysis_result['keyword_analysis']['score']
        tfidf_score = analysis_result['tfidf']['score']
        ner_score = analysis_result['ner']['score']
        sentiment_score = analysis_result['sentiment']['score']
        readability_score = analysis_result['readability']['score']
        structural_score = analysis_result['structural']['score']
        coherence_score = analysis_result['coherence']['score']

        similarity_score = analysis_result['similarity'].get('score', 0) if analysis_result.get('similarity') else 0
        ngram_score = analysis_result['ngrams'].get('score', 0)

        # Composite scores
        technical_accuracy = (
            keyword_score * 0.35 +
            ner_score * 0.35 +
            similarity_score * 0.30
        )

        depth_of_knowledge = (
            tfidf_score * 0.40 +
            structural_score * 0.40 +
            ngram_score * 0.20
        )

        communication_clarity = (
            readability_score * 0.40 +
            coherence_score * 0.35 +
            sentiment_score * 0.25
        )

        weights = question_weights
        overall_score = (
            technical_accuracy * weights['technical'] +
            depth_of_knowledge * weights['depth'] +
            communication_clarity * weights['structure']
        )

        multiplier = self.difficulty_multipliers.get(difficulty, 1.0)
        overall_score = min(overall_score * multiplier, 5.0)

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
        strengths = []
        improvements = []
        recommendations = []

        # Compare answers
        answer_points = self._extract_key_points(answer)
        best_points = self._extract_key_points(best_answer)

        gaps = []
        for point in best_points:
            if not any(self._similarity_check(point, ap) for ap in answer_points):
                gaps.append(f"Missing: {point}")

        gaps = gaps[:5]

        keyword_data = analysis_result['keyword_analysis']
        if keyword_data['coverage'] >= 60:
            strengths.append(
                f"Excellent keyword coverage ({keyword_data['coverage']:.0f}%) - you mentioned most key terms"
            )
        elif keyword_data['coverage'] >= 40:
            strengths.append(
                f"Good keyword coverage ({keyword_data['coverage']:.0f}%)"
            )
        else:
            improvements.append(
                f"Low keyword coverage ({keyword_data['coverage']:.0f}%)"
            )
            recommendations.append("Add more relevant technical terms.")

        ner_data = analysis_result['ner']
        if ner_data['total'] >= 5:
            strengths.append(f"Strong technical depth ({ner_data['total']} technical terms)")
        elif ner_data['total'] < 2:
            improvements.append("Too few technical terms")
            recommendations.append("Mention specific tools or methods.")

        struct_data = analysis_result['structural']
        if struct_data['has_examples']:
            strengths.append("Good use of examples")
        else:
            improvements.append("No specific examples")
            recommendations.append("Use STAR method for better structure.")

        read_data = analysis_result['readability']
        if read_data['assessment'] in ['Excellent', 'Good']:
            strengths.append("Good readability")
        else:
            improvements.append("Readability needs improvement")

        sent_data = analysis_result['sentiment']
        if sent_data['polarity'] < -0.1:
            improvements.append("Tone seems negative")
            recommendations.append("Use more confident language.")

        coh_data = analysis_result['coherence']
        if coh_data['score'] < 2.5:
            improvements.append("Ideas lack coherence")

        tfidf_data = analysis_result['tfidf']
        if tfidf_data['lexical_diversity'] < 0.3:
            improvements.append("Vocabulary is repetitive")

        overall_score = scores['overall']
        if overall_score >= 4.0:
            recommendations.append("Great answer! Try more advanced questions.")
        elif overall_score >= 3.0:
            recommendations.append("Improve technical depth.")
        else:
            recommendations.append("Review basic concepts.")

        return {
            'strengths': strengths[:4],
            'improvements': improvements[:4],
            'recommendations': recommendations[:5]
        }

    def get_percentile_rank(self, score, historical_scores):
        if not historical_scores:
            return 50.0

        below_count = sum(1 for s in historical_scores if s < score)
        percentile = (below_count / len(historical_scores)) * 100

        return round(percentile, 1)

    def generate_improvement_plan(self, scores, analysis_result):
        focus_areas = []

        component_scores = scores['components']
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])

        weak_components = [comp for comp, score in sorted_components[:3] if score < 3.5]

        improvement_map = {
            'keyword': {
                'area': 'Technical Terminology',
                'action': 'Study key terms',
                'resources': ['Blogs', 'Documentation']
            },
            'ner': {
                'area': 'Tools Knowledge',
                'action': 'Learn specific tools',
                'resources': ['Courses', 'Projects']
            },
            'readability': {
                'area': 'Communication',
                'action': 'Write clearer explanations',
                'resources': ['Writing guides']
            },
            'structural': {
                'area': 'Answer Structure',
                'action': 'Use STAR method',
                'resources': ['Interview prep books']
            },
            'coherence': {
                'area': 'Logical Flow',
                'action': 'Practice transitions',
                'resources': ['Writing exercises']
            },
            'similarity': {
                'area': 'Expected Content',
                'action': 'Study sample answers',
                'resources': ['Interview sites']
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
        if overall_score >= 4.5:
            return "Outstanding! You're interview-ready."
        elif overall_score >= 4.0:
            return "Very good! Practice under time pressure."
        elif overall_score >= 3.5:
            return "Good foundation. Add technical depth."
        elif overall_score >= 3.0:
            return "Improve technical accuracy."
        elif overall_score >= 2.5:
            return "Needs improvement. Review basics."
        else:
            return "Start with fundamentals and practice more."
