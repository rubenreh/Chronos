"""
Recommendation engine for Chronos productivity coaching.

This is the intelligence layer that turns raw time-series data into human-readable
productivity advice. The pipeline works as follows:

  1. Trend detection   — Compare the recent window mean to the earlier window mean
                         to classify the user's trajectory as rising, declining,
                         plateauing, or recovering.
  2. Bottleneck scan   — Use extracted behavioural features (volatility,
                         autocorrelation, trend slope) to flag specific issues.
  3. Template matching  — Select coaching messages from a curated template bank
                         keyed by trend category.
  4. Action plan        — Optionally generate a 7-day structured plan with daily
                         focus areas and suggested tasks.

The /recommend FastAPI endpoint delegates to this module.
"""

import numpy as np                                   # Numerical computations for trend math
import pandas as pd                                  # Series input type for time-series data
from typing import List, Dict, Optional, Tuple       # Type annotations for function signatures
from datetime import datetime, timedelta             # Date arithmetic for multi-day plan generation

from chronos.features.embeddings import EmbeddingGenerator        # Semantic similarity (future use)
from chronos.features.extractor import extract_behavioral_features  # Feature extraction for bottleneck detection


class RecommendationEngine:
    """Generate intelligent, template-based productivity recommendations.

    Given a user's time-series, the engine detects the macro trend, identifies
    behavioural bottlenecks, picks the most relevant coaching templates, and
    assembles a structured response with actionable steps and (optionally)
    a multi-day improvement plan.
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        trend_threshold: float = 0.1
    ):
        """Initialise the recommendation engine.

        Args:
            embedding_model: Sentence-transformer model name used by the
                             EmbeddingGenerator for semantic similarity between
                             recommendations (reserved for future ranking).
            trend_threshold: Minimum fractional change between the recent and
                             earlier windows to classify a trend as rising or
                             declining (default 10 %).
        """
        # Load the sentence-transformer for potential semantic ranking of recommendations
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.trend_threshold = trend_threshold  # ±10 % change triggers a non-plateau classification
        # Pre-load the curated recommendation templates organised by category
        self.recommendation_templates = self._load_recommendation_templates()

    def _load_recommendation_templates(self) -> Dict[str, List[str]]:
        """Return a dictionary of coaching message templates keyed by trend category.

        Each category contains several alternative messages so the engine can
        vary its output across calls.
        """
        return {
            # User is performing well — encourage them to sustain and stretch
            'high_performance': [
                "Maintain your current productivity rhythm. Consider setting slightly more ambitious goals.",
                "Your performance is strong. Focus on consistency and avoid burnout by scheduling breaks.",
                "Excellent productivity levels. Try optimizing your peak hours for the most challenging tasks."
            ],
            # Productivity is falling — suggest rest and workload review
            'declining': [
                "Your productivity has been declining. Consider reviewing your recent task patterns and identifying bottlenecks.",
                "Take time to rest and recover. Schedule lighter tasks and focus on maintaining energy levels.",
                "Review your time allocation. You may be overcommitting or experiencing task switching overhead."
            ],
            # Productivity is flat — nudge toward experimentation
            'plateauing': [
                "Your productivity has plateaued. Try introducing new challenges or varying your routine.",
                "Consider breaking larger tasks into smaller, more manageable chunks to regain momentum.",
                "Experiment with different work techniques or time management strategies to break the plateau."
            ],
            # Recovering from a dip — encourage gradual ramp-up
            'recovering': [
                "You're showing signs of recovery. Gradually increase task complexity while maintaining balance.",
                "Focus on building sustainable habits. Don't rush back to peak performance too quickly.",
                "Your productivity is improving. Maintain this trajectory with consistent daily routines."
            ],
            # Statistical anomaly detected — flag for review
            'anomaly_detected': [
                "Unusual patterns detected in your productivity. Review recent changes in your schedule or workload.",
                "Consider whether external factors are affecting your performance. Take time to assess and adjust.",
                "Anomalies in your data suggest a need for recalibration. Review your goals and expectations."
            ]
        }

    def detect_trend(self, series: pd.Series, window_size: int = 30) -> str:
        """Classify the user's productivity trajectory over the recent window.

        Compares the mean of the most recent `window_size` points against the
        mean of the preceding `window_size` points and returns one of four
        trend labels.

        Args:
            series: Full productivity time-series.
            window_size: Number of points in each comparison window (default 30).

        Returns:
            One of 'rising', 'declining', 'plateauing', or 'recovering'.
        """
        # Not enough data to form two comparison windows → default to plateauing
        if len(series) < window_size:
            return 'plateauing'

        recent = series.tail(window_size).values                        # Most recent window
        earlier = series.tail(window_size * 2).head(window_size).values  # Window just before the recent one

        recent_mean = np.mean(recent)    # Average productivity in the recent window
        earlier_mean = np.mean(earlier)  # Average productivity in the earlier window

        # Fractional change: positive → improvement, negative → decline; ε prevents div-by-zero
        change = (recent_mean - earlier_mean) / (earlier_mean + 1e-8)

        if change > self.trend_threshold:
            return 'rising'        # Productivity increasing above threshold
        elif change < -self.trend_threshold:
            # Distinguish between a general decline and a recovery from even lower levels
            if earlier_mean > recent_mean * 1.2:
                return 'recovering'  # Earlier mean was much higher → still recovering
            return 'declining'       # Straightforward decline
        else:
            return 'plateauing'      # Change within ±threshold → flat

    def detect_bottlenecks(self, series: pd.Series) -> List[str]:
        """Scan extracted features for common productivity bottlenecks.

        Uses thresholds on volatility and autocorrelation to flag issues, plus
        re-uses `detect_trend` to catch declining trajectories.

        Args:
            series: Productivity time-series.

        Returns:
            List of human-readable bottleneck descriptions (may be empty).
        """
        bottlenecks = []
        # Extract the full feature dict from the series
        features = extract_behavioral_features(series)

        # High volatility → erratic day-to-day swings, suggesting inconsistent work habits
        if features.get('volatility', 0) > 0.3:
            bottlenecks.append("High volatility in productivity suggests inconsistent work patterns")

        # Low lag-1 autocorrelation → a productive day isn't followed by another; no momentum
        if features.get('autocorr_lag1', 0) < 0.3:
            bottlenecks.append("Low momentum detected - tasks may lack continuity")

        # A declining macro trend is itself a bottleneck worth flagging
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
        """Generate a list of personalised recommendation dicts for a user.

        This is the main entry point called by the /recommend API endpoint.

        Args:
            series: User's productivity time-series.
            user_profile: Optional dict of user metadata (reserved for future
                          personalisation).
            num_recommendations: How many recommendations to return.
            multi_day: If True, attach a 7-day action plan to each recommendation.

        Returns:
            List of recommendation dicts, each containing id, text, category,
            trend, priority, reasoning, actionable_steps, and optionally
            multi_day_plan.
        """
        recommendations = []

        # Step 1: Determine the overall trend category
        trend = self.detect_trend(series)

        # Step 2: Identify any specific bottlenecks in the data
        bottlenecks = self.detect_bottlenecks(series)

        # Step 3: Map the detected trend to a template category
        if trend == 'declining':
            category = 'declining'
        elif trend == 'plateauing':
            category = 'plateauing'
        elif trend == 'recovering':
            category = 'recovering'
        else:
            category = 'high_performance'  # 'rising' maps to high_performance

        # Step 4: Pull the matching coaching message templates
        templates = self.recommendation_templates.get(category, self.recommendation_templates['plateauing'])

        # Build one recommendation dict per template (up to num_recommendations)
        for i, template in enumerate(templates[:num_recommendations]):
            rec = {
                'id': f"rec_{i+1}",                                        # Unique identifier
                'text': template,                                           # The coaching message itself
                'category': category,                                       # Which template bank it came from
                'trend': trend,                                             # Detected trend label
                'priority': 'high' if trend == 'declining' else 'medium',   # Urgency level
                'reasoning': self._generate_reasoning(trend, bottlenecks),  # Why this recommendation was chosen
                'actionable_steps': self._generate_actionable_steps(trend, category)  # Concrete to-do items
            }

            # Optionally attach a structured 7-day improvement plan
            if multi_day:
                rec['multi_day_plan'] = self._generate_multi_day_plan(trend, num_days=7)

            recommendations.append(rec)

        # Step 5: Append bottleneck-specific recommendations (up to 2 extra)
        if bottlenecks:
            for bottleneck in bottlenecks[:2]:
                rec = {
                    'id': f"bottleneck_{len(recommendations)+1}",
                    'text': f"Bottleneck detected: {bottleneck}",
                    'category': 'anomaly_detected',
                    'trend': trend,
                    'priority': 'high',                                # Bottlenecks are always high priority
                    'reasoning': bottleneck,
                    'actionable_steps': self._generate_bottleneck_steps(bottleneck)
                }
                recommendations.append(rec)

        # Return at most `num_recommendations` total items
        return recommendations[:num_recommendations]

    def _generate_reasoning(self, trend: str, bottlenecks: List[str]) -> str:
        """Build a short natural-language explanation of why recommendations were generated.

        Args:
            trend: The detected trend label.
            bottlenecks: List of bottleneck descriptions.

        Returns:
            A multi-sentence reasoning string.
        """
        reasoning_parts = [f"Trend analysis indicates {trend} productivity."]

        if bottlenecks:
            # Include up to 2 bottleneck descriptions in the reasoning
            reasoning_parts.append(f"Detected issues: {', '.join(bottlenecks[:2])}.")

        reasoning_parts.append("Recommendations are tailored to address these patterns.")
        return " ".join(reasoning_parts)  # Join into a single paragraph

    def _generate_actionable_steps(self, trend: str, category: str) -> List[str]:
        """Return a list of concrete steps the user can take based on the trend.

        Args:
            trend: Detected trend label.
            category: Template category string.

        Returns:
            List of 3 actionable step strings.
        """
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
            # 'rising' / high_performance
            steps = [
                "Maintain current productivity patterns",
                "Optimize peak performance hours",
                "Set slightly more challenging goals"
            ]

        return steps

    def _generate_multi_day_plan(self, trend: str, num_days: int = 7) -> List[Dict[str, any]]:
        """Generate a structured multi-day improvement plan.

        Each day in the plan specifies a focus area, suggested tasks, and an
        expected outcome. The plan adapts its phasing to the detected trend.

        Args:
            trend: Detected productivity trend.
            num_days: Length of the plan in days (default 7).

        Returns:
            List of day-plan dicts with keys: day, date, focus, suggested_tasks,
            expected_outcome.
        """
        plan = []
        today = datetime.now()  # Anchor the plan to the current date

        for day in range(num_days):
            day_date = today + timedelta(days=day)  # Calendar date for this plan day

            # Tailor the daily focus and tasks to the macro trend
            if trend == 'declining':
                # First 2 days: rest and recovery; remaining days: gradual ramp-up
                focus = "Recovery and rest" if day < 2 else "Gradual productivity increase"
                tasks = ["Light tasks", "Review priorities", "Plan next week"]
            elif trend == 'plateauing':
                # First 3 days: try new approaches; remaining days: build on what works
                focus = "Introduce variation" if day < 3 else "Build momentum"
                tasks = ["Try new technique", "Break large task", "Track progress"]
            elif trend == 'recovering':
                # First 4 days: solidify routine; remaining days: stretch goals
                focus = "Maintain consistency" if day < 4 else "Increase challenge"
                tasks = ["Stick to routine", "Gradual complexity", "Monitor progress"]
            else:
                # Already performing well — optimise
                focus = "Optimize performance"
                tasks = ["Peak hour tasks", "Maintain rhythm", "Set goals"]

            plan.append({
                'day': day + 1,                                            # 1-indexed day number
                'date': day_date.strftime('%Y-%m-%d'),                     # ISO date string
                'focus': focus,                                            # Daily theme
                'suggested_tasks': tasks,                                  # 3 concrete tasks
                'expected_outcome': f"Progress toward {trend} productivity"  # Motivational note
            })

        return plan

    def _generate_bottleneck_steps(self, bottleneck: str) -> List[str]:
        """Return targeted action steps to address a specific bottleneck.

        Pattern-matches on keywords in the bottleneck description to select
        the most relevant remediation advice.

        Args:
            bottleneck: Human-readable bottleneck description.

        Returns:
            List of 3 remediation steps.
        """
        if "volatility" in bottleneck.lower():
            # High volatility → stabilise routines
            return [
                "Establish consistent daily routines",
                "Use time-blocking to structure your day",
                "Track productivity patterns to identify optimal times"
            ]
        elif "momentum" in bottleneck.lower():
            # Low momentum → reduce context-switching
            return [
                "Group related tasks together",
                "Minimize context switching",
                "Build task sequences that flow naturally"
            ]
        elif "declining" in bottleneck.lower():
            # Declining trend → reduce load and recover
            return [
                "Review and reduce workload if necessary",
                "Prioritize tasks by importance and urgency",
                "Schedule recovery time"
            ]
        else:
            # Generic fallback
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
    """One-shot convenience function: instantiate an engine and generate recommendations.

    Useful for scripts, notebooks, or quick API wrappers that don't need to
    retain the engine across calls.

    Args:
        series: User's productivity time-series.
        user_profile: Optional user metadata dict.
        num_recommendations: Number of recommendations to return.

    Returns:
        List of recommendation dicts (see RecommendationEngine.generate_recommendations).
    """
    engine = RecommendationEngine()  # Create engine with default settings
    return engine.generate_recommendations(series, user_profile, num_recommendations)
