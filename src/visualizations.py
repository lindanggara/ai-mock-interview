"""
Visualization Module
Generates charts and visual representations of analysis results
"""

import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


class VisualizationGenerator:
    """
    Generator for various visualizations
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db'
        }
    
    def create_radar_chart(self, scores):
        """
        Create radar chart for score visualization
        
        Args:
            scores (dict): Score breakdown
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        categories = [
            'Technical Accuracy',
            'Depth of Knowledge',
            'Communication Clarity',
            'Overall Performance'
        ]
        
        values = [
            scores['technical_accuracy'],
            scores['depth_of_knowledge'],
            scores['communication_clarity'],
            scores['overall']
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Score',
            line=dict(color=self.color_palette['primary'], width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Add reference line at 3.5 (good threshold)
        fig.add_trace(go.Scatterpolar(
            r=[3.5, 3.5, 3.5, 3.5],
            theta=categories,
            fill='toself',
            name='Target (3.5)',
            line=dict(color=self.color_palette['success'], width=1, dash='dash'),
            fillcolor='rgba(46, 204, 113, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            showlegend=True,
            title="Performance Radar Chart",
            height=400
        )
        
        return fig
    
    def create_score_breakdown(self, scores):
        """
        Create bar chart for score breakdown
        
        Args:
            scores (dict): Score breakdown
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        components = scores['components']
        
        labels = [
            'Keyword Match',
            'TF-IDF',
            'Named Entities',
            'Sentiment',
            'Readability',
            'Structure',
            'Coherence',
            'Similarity'
        ]
        
        values = [
            components['keyword'],
            components['tfidf'],
            components['ner'],
            components['sentiment'],
            components['readability'],
            components['structural'],
            components['coherence'],
            components['similarity']
        ]
        
        # Color based on value
        colors = [
            self.color_palette['success'] if v >= 4.0 else
            self.color_palette['info'] if v >= 3.0 else
            self.color_palette['warning'] if v >= 2.0 else
            self.color_palette['danger']
            for v in values
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}" for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Component Score Breakdown",
            xaxis_title="Analysis Component",
            yaxis_title="Score (out of 5.0)",
            yaxis=dict(range=[0, 5.5]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_wordcloud(self, text, stopwords=None):
        """
        Create word cloud visualization
        
        Args:
            text (str): Input text
            stopwords (set): Stopwords to exclude
            
        Returns:
            matplotlib.figure.Figure: Word cloud figure
        """
        if not text or len(text.split()) < 10:
            # Return empty figure if text is too short
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Not enough text for word cloud', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords,
            colormap='viridis',
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud - Most Frequent Terms', fontsize=14, pad=20)
        
        return fig
    
    def create_progress_chart(self, history):
        """
        Create line chart showing score progression
        
        Args:
            history (list): List of historical scores
            
        Returns:
            plotly.graph_objects.Figure: Line chart
        """
        if not history:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No history available yet",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig
        
        attempts = list(range(1, len(history) + 1))
        scores = [item['score'] for item in history]
        
        # Calculate moving average
        window = min(3, len(scores))
        moving_avg = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(scores[start:i+1]))
        
        fig = go.Figure()
        
        # Add actual scores
        fig.add_trace(go.Scatter(
            x=attempts,
            y=scores,
            mode='lines+markers',
            name='Score',
            line=dict(color=self.color_palette['primary'], width=2),
            marker=dict(size=8)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=attempts,
            y=moving_avg,
            mode='lines',
            name='Trend',
            line=dict(color=self.color_palette['success'], width=2, dash='dash')
        ))
        
        # Add target line
        fig.add_hline(
            y=3.5,
            line_dash="dot",
            line_color=self.color_palette['warning'],
            annotation_text="Target (3.5)"
        )
        
        fig.update_layout(
            title="Score Progression Over Time",
            xaxis_title="Attempt Number",
            yaxis_title="Score",
            yaxis=dict(range=[0, 5.5]),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_comparison_chart(self, current_scores, avg_scores):
        """
        Create comparison chart between current and average scores
        
        Args:
            current_scores (dict): Current attempt scores
            avg_scores (dict): Average historical scores
            
        Returns:
            plotly.graph_objects.Figure: Grouped bar chart
        """
        categories = [
            'Technical',
            'Depth',
            'Communication',
            'Overall'
        ]
        
        current = [
            current_scores['technical_accuracy'],
            current_scores['depth_of_knowledge'],
            current_scores['communication_clarity'],
            current_scores['overall']
        ]
        
        average = [
            avg_scores.get('technical_accuracy', 0),
            avg_scores.get('depth_of_knowledge', 0),
            avg_scores.get('communication_clarity', 0),
            avg_scores.get('overall', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                name='Current',
                x=categories,
                y=current,
                marker_color=self.color_palette['primary']
            ),
            go.Bar(
                name='Your Average',
                x=categories,
                y=average,
                marker_color=self.color_palette['info']
            )
        ])
        
        fig.update_layout(
            title="Current vs. Average Performance",
            xaxis_title="Category",
            yaxis_title="Score",
            yaxis=dict(range=[0, 5.5]),
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_heatmap(self, score_matrix, categories, components):
        """
        Create heatmap for multi-dimensional analysis
        
        Args:
            score_matrix (list): 2D array of scores
            categories (list): Row labels
            components (list): Column labels
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=components,
            y=categories,
            colorscale='RdYlGn',
            text=score_matrix,
            texttemplate='%{text:.1f}',
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title="Performance Heatmap Across Categories",
            xaxis_title="Analysis Component",
            yaxis_title="Question Category",
            height=500
        )
        
        return fig
    
    def create_gauge_chart(self, score, title="Overall Score"):
        """
        Create gauge chart for single score
        
        Args:
            score (float): Score value (0-5)
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Gauge chart
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 3.5, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': self.color_palette['primary']},
                'steps': [
                    {'range': [0, 2], 'color': self.color_palette['danger']},
                    {'range': [2, 3], 'color': self.color_palette['warning']},
                    {'range': [3, 4], 'color': self.color_palette['info']},
                    {'range': [4, 5], 'color': self.color_palette['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        return fig
    
    def create_improvement_tracker(self, scores_by_category):
        """
        Create improvement tracker showing performance across categories
        
        Args:
            scores_by_category (dict): Scores grouped by question category
            
        Returns:
            plotly.graph_objects.Figure: Stacked bar chart
        """
        categories = list(scores_by_category.keys())
        attempts = max(len(scores) for scores in scores_by_category.values())
        
        fig = go.Figure()
        
        for i in range(attempts):
            attempt_scores = []
            for cat in categories:
                if i < len(scores_by_category[cat]):
                    attempt_scores.append(scores_by_category[cat][i])
                else:
                    attempt_scores.append(0)
            
            fig.add_trace(go.Bar(
                name=f'Attempt {i+1}',
                x=categories,
                y=attempt_scores
            ))
        
        fig.update_layout(
            title="Performance by Category",
            xaxis_title="Question Category",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        return fig
