"""
Voice Handler Module
Handles voice recording and transcription for voice mode interviews
"""

import streamlit as st


class VoiceHandler:
    """
    Handles voice recording and speech-to-text transcription
    """
    
    def __init__(self):
        self.sample_transcriptions = {
            'technical': """
            I have extensive experience with Python and its data science ecosystem. 
            In my recent project analyzing customer churn for an e-commerce company, 
            I used pandas for data manipulation handling over 2 million records. 
            I implemented feature engineering using numpy and built predictive models 
            with scikit-learn's RandomForestClassifier achieving 87% accuracy. 
            The project involved data cleaning, handling missing values, and creating 
            visualizations with matplotlib to present findings to stakeholders. 
            I also deployed the model using Flask API with Docker containerization.
            """,
            'statistical': """
            To explain supervised versus unsupervised learning to non-technical stakeholders, 
            I use practical analogies. Supervised learning is like teaching with flashcards - 
            you show examples with correct answers. For instance, email spam detection where 
            we train models with labeled data. Unsupervised learning is like organizing without 
            instructions - the algorithm finds patterns itself. Customer segmentation is a good 
            example where we group customers by behavior without predefined categories. 
            The key difference is whether we provide the correct answers during training.
            """,
            'problem_solving': """
            For a customer churn prediction project, I would follow a systematic approach. 
            First, understand the business context and define what constitutes churn. 
            Second, collect historical customer data including demographics and behavior patterns. 
            Third, perform exploratory data analysis to identify key patterns. 
            Fourth, engineer features like recency metrics and lifetime value. 
            Fifth, build baseline models before trying complex algorithms. 
            Sixth, evaluate using appropriate metrics like precision-recall. 
            Finally, deploy with monitoring for data drift and model performance.
            """
        }
    
    def transcribe_audio(self, question_category='technical'):
        """
        Simulate voice transcription
        In production, this would use actual speech recognition
        
        Args:
            question_category (str): Category to determine sample response
            
        Returns:
            str: Transcribed text
        """
        # In a real implementation, this would:
        # 1. Record audio from microphone
        # 2. Send to speech recognition service (Google Speech API, Whisper, etc.)
        # 3. Return transcribed text
        
        # For now, return a sample transcription
        category_map = {
            'Technical Skills & Background': 'technical',
            'Statistical Knowledge': 'statistical',
            'Problem Solving & Case Study': 'problem_solving'
        }
        
        selected = category_map.get(question_category, 'technical')
        return self.sample_transcriptions[selected].strip()
    
    def get_microphone_button_html(self):
        """
        Returns HTML for a styled microphone button
        """
        return """
        <style>
        .mic-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50%;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
        }
        
        .mic-button:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
        }
        
        .mic-button:active {
            transform: scale(0.95);
        }
        
        .mic-icon {
            color: white;
            font-size: 32px;
        }
        
        .recording {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        </style>
        
        <button class="mic-button" id="micButton">
            <span class="mic-icon">🎤</span>
        </button>
        """
    
    def record_audio_streamlit(self):
        """
        Streamlit-compatible audio recording interface
        """
        st.markdown("### 🎤 Voice Recording")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(self.get_microphone_button_html(), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #666;'>Click to start recording</p>", 
                       unsafe_allow_html=True)
        
        recording_instructions = """
        **Recording Tips:**
        - 🎯 Speak clearly and at a moderate pace
        - 🔇 Find a quiet environment
        - ⏱️ Aim for 2-3 minutes
        - 📝 Structure your answer (STAR method)
        - 🎤 Keep microphone at consistent distance
        """
        
        st.info(recording_instructions)
    
    def provide_voice_mode_guide(self):
        """
        Provides comprehensive guide for voice mode
        """
        guide = """
        ## 🎤 Voice Mode Guide
        
        ### How It Works
        1. Click "Start Recording" button
        2. Speak your answer clearly
        3. Click "Stop Recording" when done
        4. Review the transcription
        5. Edit if needed, then analyze
        
        ### Best Practices
        - **Environment**: Choose a quiet location
        - **Microphone**: Use a good quality mic if possible
        - **Pace**: Speak at normal conversational speed
        - **Clarity**: Enunciate technical terms clearly
        - **Structure**: Plan your answer before speaking
        
        ### Tips for Technical Terms
        - Spell out abbreviations: "S-Q-L" instead of "sequel"
        - Pause briefly before technical terms
        - Use context to help recognition
        
        ### Common Issues
        - **Background Noise**: Use headphones with mic
        - **Accent**: Speak slowly and clearly
        - **Technical Words**: May need manual correction
        """
        
        return guide
    
    def get_transcription_confidence(self, text):
        """
        Estimate transcription confidence (placeholder)
        In production, this would come from the speech recognition API
        
        Args:
            text (str): Transcribed text
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple heuristic: longer text = higher confidence
        word_count = len(text.split())
        
        if word_count < 20:
            return 0.6
        elif word_count < 100:
            return 0.75
        elif word_count < 200:
            return 0.85
        else:
            return 0.9
    
    def format_transcription_display(self, text, confidence):
        """
        Format transcription with confidence indicator
        
        Args:
            text (str): Transcribed text
            confidence (float): Confidence score
            
        Returns:
            str: Formatted HTML
        """
        if confidence >= 0.8:
            color = "#28a745"
            label = "High Confidence"
        elif confidence >= 0.6:
            color = "#ffc107"
            label = "Medium Confidence"
        else:
            color = "#dc3545"
            label = "Low Confidence - Please Review"
        
        html = f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
            <p style="margin: 0; color: {color}; font-weight: bold;">
                🎤 Transcription {label} ({confidence:.0%})
            </p>
            <p style="margin-top: 0.5rem; color: #333;">
                {text}
            </p>
        </div>
        """
        
        return html
    
    def get_voice_mode_settings(self):
        """
        Get settings for voice mode
        """
        settings = {
            'language': 'en-US',  # Default to English
            'sample_rate': 16000,
            'channels': 1,
            'max_duration': 300,  # 5 minutes max
            'auto_punctuation': True,
            'filter_profanity': False
        }
        
        return settings
    
    def validate_audio_quality(self, audio_data):
        """
        Validate audio quality before transcription
        
        Args:
            audio_data: Audio data
            
        Returns:
            dict: Validation results
        """
        # Placeholder for audio quality validation
        # In production, check:
        # - Volume level
        # - Background noise
        # - Sample rate
        # - Duration
        
        return {
            'valid': True,
            'quality_score': 0.85,
            'issues': [],
            'recommendations': []
        }
    
    def suggest_improvements(self, transcription):
        """
        Suggest improvements for voice delivery
        
        Args:
            transcription (str): Transcribed text
            
        Returns:
            list: Suggestions
        """
        suggestions = []
        
        # Check for filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        filler_count = sum(transcription.lower().count(word) for word in filler_words)
        
        if filler_count > 5:
            suggestions.append("Try to reduce filler words (um, uh, like) - practice pausing instead")
        
        # Check pace (words per minute)
        word_count = len(transcription.split())
        # Assuming average speaking time of 2-3 minutes
        if word_count < 150:
            suggestions.append("You might be speaking too slowly - try to be more concise")
        elif word_count > 400:
            suggestions.append("You might be speaking too fast - slow down for clarity")
        
        # Check for technical terms
        technical_terms = ['python', 'sql', 'machine learning', 'data', 'model', 'algorithm']
        tech_count = sum(transcription.lower().count(term) for term in technical_terms)
        
        if tech_count < 3:
            suggestions.append("Include more technical terms to demonstrate expertise")
        
        return suggestions
