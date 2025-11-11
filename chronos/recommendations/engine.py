"""Recommendation engine for productivity coaching."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from chronos.features.embeddings import EmbeddingGenerator
from chronos.features.extractor import extract_behavioral_features


class RecommendationEngine:
    """Generate intelligent productivity recommendations."""
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        trend_threshold: float = 0.1
    ):
        """Initialize recommendation engine.
        
        Args:
            embedding_model: Sentence transformer model name
            trend_threshold: Threshold for detecting significant trends
        """
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.trend_threshold = trend_threshold
        self.recommendation_templates = self._load_recommendation_templates()
    
    def _load_recommendation_templates(self) -> Dict[str, List[str]]:
        """Load recommendation templates by category."""
        return {
            'high_performance': [
                "Maintain your current productivity rhythm. Consider setting slightly more ambitious goals.",
                "Your performance is strong. Focus on consistency and avoid burnout by scheduling breaks.",
                "Excellent productivity levels. Try optimizing your peak hours for the most challenging tasks."
            ],
            'declining': [
                "Your productivity has been declining. Consider reviewing your recent task patterns and identifying bottlenecks.",
                "Take time to rest and recover. Schedule lighter tasks and focus on maintaining energy levels.",
                "Review your time allocation. You may be overcommitting or experiencing task switching overhead."
            ],
            'plateauing': [
                "Your productivity has plateaued. Try introducing new challenges or varying your routine.",
                "Consider breaking larger tasks into smaller, more manageable chunks to regain momentum.",
                "Experiment with different work techniques or time management strategies to break the plateau."
            ],
            'recovering': [
                "You're showing signs of recovery. Gradually increase task complexity while maintaining balance.",
                "Focus on building sustainable habits. Don't rush back to peak performance too quickly.",
                "Your productivity is improving. Maintain this trajectory with consistent daily routines."
            ],
            'anomaly_detected': [
                "Unusual patterns detected in your productivity. Review recent changes in your schedule or workload.",
                "Consider whether external factors are affecting your performance. Take time to assess and adjust.",
                "Anomalies in your data suggest a need for recalibration. Review your goals and expectations."
            ]
        }
    
    def detect_trend(self, series: pd.Series, window_size: int = 30) -> str:
        """Detect productivity trend.
        
        Args:
            series: Time series data
            window_size: Window size for trend analysis
        
        Returns:
            Trend classification: 'rising', 'declining', 'plateauing', 'recovering'
        """
        if len(series) < window_size:
            return 'plateauing'
        
        recent = series.tail(window_size).values
        earlier = series.tail(window_size * 2).head(window_size).values
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        change = (recent_mean - earlier_mean) / (earlier_mean + 1e-8)
        
        if change > self.trend_threshold:
            return 'rising'
        elif change < -self.trend_threshold:
            if earlier_mean > recent_mean * 1.2:
                return 'recovering'
            return 'declining'
        else:
            return 'plateauing'
    
    def detect_bottlenecks(self, series: pd.Series) -> List[str]:
        """Detect potential bottlenecks in productivity.
        
        Args:
            series: Time series data
        
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        features = extract_behavioral_features(series)
        
        # High volatility suggests inconsistency
        if features.get('volatility', 0) > 0.3:
            bottlenecks.append("High volatility in productivity suggests inconsistent work patterns")
        
        # Low autocorrelation suggests lack of momentum
        if features.get('autocorr_lag1', 0) < 0.3:
            bottlenecks.append("Low momentum detected - tasks may lack continuity")
        
        # Declining trend
        trend = self.detect_trend(series)
        if trend == 'declining':
            bottlenecks.append("Declining productivity trend - review workload and priorities")
        
        return bottlenecks
    
    def generate_recommendations(
        self,
        series: pd.Series,
        user_profile: Optional[Dict] = None,
        num_recommendations: int = 3,
        multi_day: bool = True
    ) -> List[Dict[str, any]]:
        """Generate personalized recommendations.
        
        Args:
            series: User's productivity time series
            user_profile: Optional user profile dictionary
            num_recommendations: Number of recommendations to generate
            multi_day: Whether to generate multi-day plans
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Detect trend
        trend = self.detect_trend(series)
        
        # Detect bottlenecks
        bottlenecks = self.detect_bottlenecks(series)
        
        # Select recommendation category
        if trend == 'declining':
            category = 'declining'
        elif trend == 'plateauing':
            category = 'plateauing'
        elif trend == 'recovering':
            category = 'recovering'
        else:
            category = 'high_performance'
        
        # Generate base recommendations
        templates = self.recommendation_templates.get(category, self.recommendation_templates['plateauing'])
        
        for i, template in enumerate(templates[:num_recommendations]):
            rec = {
                'id': f"rec_{i+1}",
                'text': template,
                'category': category,
                'trend': trend,
                'priority': 'high' if trend == 'declining' else 'medium',
                'reasoning': self._generate_reasoning(trend, bottlenecks),
                'actionable_steps': self._generate_actionable_steps(trend, category)
            }
            
            if multi_day:
                rec['multi_day_plan'] = self._generate_multi_day_plan(trend, num_days=7)
            
            recommendations.append(rec)
        
        # Add bottleneck-specific recommendations
        if bottlenecks:
            for bottleneck in bottlenecks[:2]:
                rec = {
                    'id': f"bottleneck_{len(recommendations)+1}",
                    'text': f"Bottleneck detected: {bottleneck}",
                    'category': 'anomaly_detected',
                    'trend': trend,
                    'priority': 'high',
                    'reasoning': bottleneck,
                    'actionable_steps': self._generate_bottleneck_steps(bottleneck)
                }
                recommendations.append(rec)
        
        return recommendations[:num_recommendations]
    
    def _generate_reasoning(self, trend: str, bottlenecks: List[str]) -> str:
        """Generate reasoning for recommendation."""
        reasoning_parts = [f"Trend analysis indicates {trend} productivity."]
        
        if bottlenecks:
            reasoning_parts.append(f"Detected issues: {', '.join(bottlenecks[:2])}.")
        
        reasoning_parts.append("Recommendations are tailored to address these patterns.")
        return " ".join(reasoning_parts)
    
    def _generate_actionable_steps(self, trend: str, category: str) -> List[str]:
        """Generate actionable steps based on trend."""
        steps = []
        
        if trend == 'declining':
            steps = [
                "Review your task list and prioritize essential items",
                "Schedule regular breaks every 90 minutes",
                "Identify and eliminate time-wasting activities"
            ]
        elif trend == 'plateauing':
            steps = [
                "Introduce a new productivity technique this week",
                "Break one large task into smaller milestones",
                "Vary your work routine to maintain engagement"
            ]
        elif trend == 'recovering':
            steps = [
                "Gradually increase task complexity",
                "Maintain consistent daily routines",
                "Track progress to build momentum"
            ]
        else:
            steps = [
                "Maintain current productivity patterns",
                "Optimize peak performance hours",
                "Set slightly more challenging goals"
            ]
        
        return steps
    
    def _generate_multi_day_plan(self, trend: str, num_days: int = 7) -> List[Dict[str, any]]:
        """Generate multi-day actionable plan."""
        plan = []
        today = datetime.now()
        
        for day in range(num_days):
            day_date = today + timedelta(days=day)
            
            if trend == 'declining':
                focus = "Recovery and rest" if day < 2 else "Gradual productivity increase"
                tasks = ["Light tasks", "Review priorities", "Plan next week"]
            elif trend == 'plateauing':
                focus = "Introduce variation" if day < 3 else "Build momentum"
                tasks = ["Try new technique", "Break large task", "Track progress"]
            elif trend == 'recovering':
                focus = "Maintain consistency" if day < 4 else "Increase challenge"
                tasks = ["Stick to routine", "Gradual complexity", "Monitor progress"]
            else:
                focus = "Optimize performance"
                tasks = ["Peak hour tasks", "Maintain rhythm", "Set goals"]
            
            plan.append({
                'day': day + 1,
                'date': day_date.strftime('%Y-%m-%d'),
                'focus': focus,
                'suggested_tasks': tasks,
                'expected_outcome': f"Progress toward {trend} productivity"
            })
        
        return plan
    
    def _generate_bottleneck_steps(self, bottleneck: str) -> List[str]:
        """Generate steps to address specific bottleneck."""
        if "volatility" in bottleneck.lower():
            return [
                "Establish consistent daily routines",
                "Use time-blocking to structure your day",
                "Track productivity patterns to identify optimal times"
            ]
        elif "momentum" in bottleneck.lower():
            return [
                "Group related tasks together",
                "Minimize context switching",
                "Build task sequences that flow naturally"
            ]
        elif "declining" in bottleneck.lower():
            return [
                "Review and reduce workload if necessary",
                "Prioritize tasks by importance and urgency",
                "Schedule recovery time"
            ]
        else:
            return [
                "Review recent changes",
                "Assess external factors",
                "Adjust expectations and goals"
            ]


def generate_recommendations(
    series: pd.Series,
    user_profile: Optional[Dict] = None,
    num_recommendations: int = 3
) -> List[Dict[str, any]]:
    """Convenience function to generate recommendations."""
    engine = RecommendationEngine()
    return engine.generate_recommendations(series, user_profile, num_recommendations)

