#!/usr/bin/env python
"""
PowerPredict - Advanced Lottery Number Prediction System
Employs multiple intelligent strategies: statistical analysis, pattern recognition,
Markov chains, temporal analysis, and deep learning ensemble methods.
"""

# Suppress warnings before imports
from keras.regularizers import l2
from keras.models import Model
from keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Concatenate, BatchNormalization
)
import keras
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from itertools import combinations
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Keras 3.x imports (standalone keras, not tensorflow.keras)

# Game configurations
GAMES = {
    "megamillions": {
        "ball": "Mega Ball",
        "featured_ball": "Megaplier",
        "game": "Mega_Millions",
        "featured_range": 25,
        "high_range": 70
    },
    "powerball": {
        "ball": "Power Ball",
        "featured_ball": "Power Play",
        "game": "Powerball",
        "featured_range": 26,
        "high_range": 69
    }
}


class LotteryAnalyzer:
    """Advanced statistical analysis engine for lottery data."""

    def __init__(self, data, game):
        self.data = data
        self.game = game
        self.high_range = GAMES[game]["high_range"]
        self.ball_range = GAMES[game]["featured_range"]
        self.main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]
        self.ball_col = GAMES[game]["ball"]

        # Extract number arrays
        self.main_numbers = data[self.main_cols].values
        self.ball_numbers = data[self.ball_col].values
        self.all_main = self.main_numbers.flatten()

        # Run all analyses
        self._analyze_all()

    def _analyze_all(self):
        """Run comprehensive analysis suite."""
        self.frequency = self._frequency_analysis()
        self.gaps = self._gap_analysis()
        self.positional = self._positional_analysis()
        self.pairs = self._pair_analysis()
        self.triplets = self._triplet_analysis()
        self.delta = self._delta_analysis()
        self.sum_range = self._sum_range_analysis()
        self.odd_even = self._odd_even_analysis()
        self.high_low = self._high_low_analysis()
        self.decades = self._decade_analysis()
        self.markov = self._markov_chain_analysis()
        self.consecutive = self._consecutive_analysis()
        self.hot_cold_cycles = self._hot_cold_cycle_analysis()

    def _frequency_analysis(self):
        """Analyze overall number frequencies."""
        main_freq = Counter(self.all_main)
        ball_freq = Counter(self.ball_numbers)

        # Calculate expected frequency
        total_draws = len(self.data)
        expected_main = (total_draws * 5) / self.high_range
        expected_ball = total_draws / self.ball_range

        # Calculate deviation from expected
        main_deviation = {
            num: (main_freq.get(num, 0) - expected_main) / expected_main
            for num in range(1, self.high_range + 1)
        }
        ball_deviation = {
            num: (ball_freq.get(num, 0) - expected_ball) / expected_ball
            for num in range(1, self.ball_range + 1)
        }

        return {
            'main_freq': main_freq,
            'ball_freq': ball_freq,
            'main_deviation': main_deviation,
            'ball_deviation': ball_deviation,
            'expected_main': expected_main,
            'expected_ball': expected_ball
        }

    def _gap_analysis(self):
        """Analyze gaps between number appearances."""
        total_draws = len(self.data)

        # Current gaps (draws since last appearance)
        main_gaps = {}
        for num in range(1, self.high_range + 1):
            for i in range(total_draws - 1, -1, -1):
                if num in self.main_numbers[i]:
                    main_gaps[num] = total_draws - 1 - i
                    break
            else:
                main_gaps[num] = total_draws

        ball_gaps = {}
        for num in range(1, self.ball_range + 1):
            for i in range(total_draws - 1, -1, -1):
                if self.ball_numbers[i] == num:
                    ball_gaps[num] = total_draws - 1 - i
                    break
            else:
                ball_gaps[num] = total_draws

        # Average gap between appearances
        main_avg_gap = self._calculate_average_gaps(
            self.main_numbers, self.high_range)
        ball_avg_gap = self._calculate_average_gaps(
            self.ball_numbers.reshape(-1, 1), self.ball_range
        )

        return {
            'main_current': main_gaps,
            'ball_current': ball_gaps,
            'main_average': main_avg_gap,
            'ball_average': ball_avg_gap
        }

    def _calculate_average_gaps(self, numbers, max_num):
        """Calculate average gap between appearances for each number."""
        avg_gaps = {}
        for num in range(1, max_num + 1):
            appearances = []
            for i, row in enumerate(numbers):
                if num in row:
                    appearances.append(i)

            if len(appearances) > 1:
                gaps = np.diff(appearances)
                avg_gaps[num] = np.mean(gaps)
            else:
                avg_gaps[num] = len(numbers)  # Never or rarely appeared

        return avg_gaps

    def _positional_analysis(self):
        """Analyze which numbers appear most in each position."""
        positional_freq = {}
        for i, col in enumerate(self.main_cols):
            positional_freq[i] = Counter(self.data[col].values)
        return positional_freq

    def _pair_analysis(self):
        """Analyze frequently occurring number pairs."""
        pair_counts = Counter()
        for row in self.main_numbers:
            for pair in combinations(sorted(row), 2):
                pair_counts[pair] += 1
        return pair_counts

    def _triplet_analysis(self):
        """Analyze frequently occurring number triplets."""
        triplet_counts = Counter()
        for row in self.main_numbers:
            for triplet in combinations(sorted(row), 3):
                triplet_counts[triplet] += 1
        return triplet_counts

    def _delta_analysis(self):
        """Analyze differences between consecutive numbers in draws."""
        deltas = []
        for row in self.main_numbers:
            sorted_row = sorted(row)
            row_deltas = np.diff(sorted_row)
            deltas.extend(row_deltas)

        delta_freq = Counter(deltas)
        avg_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        return {
            'frequency': delta_freq,
            'average': avg_delta,
            'std': std_delta,
            'common_deltas': [d for d, _ in delta_freq.most_common(5)]
        }

    def _sum_range_analysis(self):
        """Analyze sum and range patterns of winning numbers."""
        sums = [sum(row) for row in self.main_numbers]
        ranges = [max(row) - min(row) for row in self.main_numbers]

        return {
            'sum_mean': np.mean(sums),
            'sum_std': np.std(sums),
            'sum_min': np.min(sums),
            'sum_max': np.max(sums),
            'range_mean': np.mean(ranges),
            'range_std': np.std(ranges),
            'optimal_sum_range': (
                np.mean(sums) - np.std(sums),
                np.mean(sums) + np.std(sums)
            )
        }

    def _odd_even_analysis(self):
        """Analyze odd/even distribution patterns."""
        patterns = []
        for row in self.main_numbers:
            odd_count = sum(1 for n in row if n % 2 == 1)
            patterns.append(odd_count)

        pattern_freq = Counter(patterns)
        return {
            'frequency': pattern_freq,
            'most_common': pattern_freq.most_common(1)[0][0],
            'average_odd': np.mean(patterns)
        }

    def _high_low_analysis(self):
        """Analyze high/low number distribution."""
        mid_point = self.high_range // 2
        patterns = []
        for row in self.main_numbers:
            low_count = sum(1 for n in row if n <= mid_point)
            patterns.append(low_count)

        pattern_freq = Counter(patterns)
        return {
            'frequency': pattern_freq,
            'most_common': pattern_freq.most_common(1)[0][0],
            'mid_point': mid_point
        }

    def _decade_analysis(self):
        """Analyze distribution across decades (1-10, 11-20, etc.)."""
        decade_counts = defaultdict(list)
        for row in self.main_numbers:
            row_decades = Counter((n - 1) // 10 for n in row)
            for decade in range(self.high_range // 10 + 1):
                decade_counts[decade].append(row_decades.get(decade, 0))

        return {
            decade: {
                'mean': np.mean(counts),
                'std': np.std(counts)
            }
            for decade, counts in decade_counts.items()
        }

    def _markov_chain_analysis(self):
        """Build Markov transition matrix for number sequences."""
        # Simplified: track which numbers follow which in consecutive draws
        transitions = defaultdict(Counter)

        for i in range(len(self.main_numbers) - 1):
            current_set = set(self.main_numbers[i])
            next_set = set(self.main_numbers[i + 1])

            for curr_num in current_set:
                for next_num in next_set:
                    transitions[curr_num][next_num] += 1

        # Normalize to probabilities
        transition_probs = {}
        for num, followers in transitions.items():
            total = sum(followers.values())
            transition_probs[num] = {
                k: v / total for k, v in followers.items()
            }

        return transition_probs

    def _consecutive_analysis(self):
        """Analyze patterns of consecutive numbers."""
        consecutive_counts = Counter()
        for row in self.main_numbers:
            sorted_row = sorted(row)
            consec_count = 0
            for i in range(len(sorted_row) - 1):
                if sorted_row[i + 1] - sorted_row[i] == 1:
                    consec_count += 1
            consecutive_counts[consec_count] += 1

        return {
            'frequency': consecutive_counts,
            'probability': {
                k: v / len(self.main_numbers)
                for k, v in consecutive_counts.items()
            }
        }

    def _hot_cold_cycle_analysis(self):
        """Analyze hot/cold cycles - numbers transitioning between states."""
        window = 20  # Rolling window
        if len(self.data) < window * 2:
            return None

        hot_threshold = 1.2  # 20% above average
        cold_threshold = 0.8  # 20% below average

        expected_per_window = (window * 5) / self.high_range

        cycles = defaultdict(list)
        for num in range(1, self.high_range + 1):
            for i in range(window, len(self.main_numbers)):
                window_data = self.main_numbers[i-window:i]
                count = sum(1 for row in window_data if num in row)
                ratio = count / expected_per_window

                if ratio >= hot_threshold:
                    cycles[num].append(('hot', i))
                elif ratio <= cold_threshold:
                    cycles[num].append(('cold', i))

        # Find numbers currently transitioning from cold to warming
        warming_numbers = []
        for num in range(1, self.high_range + 1):
            if cycles[num]:
                recent = cycles[num][-5:] if len(cycles[num]
                                                 ) >= 5 else cycles[num]
                cold_recent = sum(1 for state, _ in recent if state == 'cold')
                if cold_recent >= 3:  # Was cold recently
                    # Check if frequency is increasing
                    recent_freq = sum(
                        1 for row in self.main_numbers[-10:] if num in row
                    )
                    if recent_freq >= 2:  # Starting to appear more
                        warming_numbers.append(num)

        return {
            'cycles': dict(cycles),
            'warming_numbers': warming_numbers
        }


class IntelligentPredictor:
    """Multi-strategy prediction engine combining statistical and ML approaches."""

    def __init__(self, analyzer, data, game):
        self.analyzer = analyzer
        self.data = data
        self.game = game
        self.high_range = GAMES[game]["high_range"]
        self.ball_range = GAMES[game]["featured_range"]

    def calculate_number_scores(self):
        """Calculate composite scores for each number using multiple factors."""
        main_scores = {}
        ball_scores = {}

        freq = self.analyzer.frequency
        gaps = self.analyzer.gaps
        positional = self.analyzer.positional
        markov = self.analyzer.markov
        hot_cold = self.analyzer.hot_cold_cycles

        # Get last draw for Markov predictions
        last_draw = set(self.analyzer.main_numbers[-1])

        for num in range(1, self.high_range + 1):
            score = 0.0

            # 1. Frequency score (normalized deviation from expected)
            freq_dev = freq['main_deviation'].get(num, 0)
            freq_score = 0.5 + (freq_dev * 0.3)  # Slight bias toward frequent
            score += freq_score * 0.15

            # 2. Gap score (overdue numbers get bonus, but not too overdue)
            current_gap = gaps['main_current'].get(num, 0)
            avg_gap = gaps['main_average'].get(num, 10)
            gap_ratio = current_gap / max(avg_gap, 1)
            # Sweet spot: 1.0-2.0x average gap
            if 1.0 <= gap_ratio <= 2.0:
                gap_score = 0.8
            elif gap_ratio < 1.0:
                gap_score = 0.5 * gap_ratio
            else:
                gap_score = max(0.3, 0.8 - (gap_ratio - 2.0) * 0.1)
            score += gap_score * 0.20

            # 3. Positional tendency score
            pos_score = 0
            for pos, pos_freq in positional.items():
                if num in pos_freq:
                    pos_score += pos_freq[num] / len(self.data)
            score += min(pos_score, 1.0) * 0.15

            # 4. Markov transition probability
            markov_score = 0
            for last_num in last_draw:
                if last_num in markov and num in markov[last_num]:
                    markov_score += markov[last_num][num]
            markov_score = min(markov_score / len(last_draw), 1.0)
            score += markov_score * 0.20

            # 5. Pair synergy (how well this number pairs with hot numbers)
            pair_score = 0
            hot_nums = [n for n, _ in freq['main_freq'].most_common(15)]
            for hot in hot_nums:
                pair = tuple(sorted([num, hot]))
                if pair in self.analyzer.pairs:
                    pair_score += self.analyzer.pairs[pair]
            pair_score = min(pair_score / 100, 1.0)
            score += pair_score * 0.15

            # 6. Hot/cold cycle bonus
            if hot_cold and num in hot_cold.get('warming_numbers', []):
                score += 0.15 * 0.15  # Bonus for warming numbers

            main_scores[num] = max(0.01, score)  # Ensure positive

        # Ball scores (simpler calculation)
        for num in range(1, self.ball_range + 1):
            score = 0.0

            freq_dev = freq['ball_deviation'].get(num, 0)
            score += (0.5 + freq_dev * 0.3) * 0.4

            current_gap = gaps['ball_current'].get(num, 0)
            avg_gap = gaps['ball_average'].get(num, 10)
            gap_ratio = current_gap / max(avg_gap, 1)
            if 1.0 <= gap_ratio <= 2.0:
                gap_score = 0.8
            else:
                gap_score = 0.5
            score += gap_score * 0.6

            ball_scores[num] = max(0.01, score)

        return main_scores, ball_scores

    def generate_statistical_prediction(self, num_predictions):
        """Generate predictions using weighted statistical analysis."""
        main_scores, ball_scores = self.calculate_number_scores()
        predictions = []

        # Normalize scores to probabilities
        main_probs = np.array(list(main_scores.values()))
        main_probs = main_probs / main_probs.sum()
        main_nums_list = list(main_scores.keys())

        ball_probs = np.array(list(ball_scores.values()))
        ball_probs = ball_probs / ball_probs.sum()
        ball_nums_list = list(ball_scores.keys())

        for _ in range(num_predictions):
            # Select 5 unique main numbers
            selected = np.random.choice(
                main_nums_list, size=5, replace=False, p=main_probs
            )

            # Validate against sum/range constraints
            selected = self._optimize_selection(selected, main_scores)

            ball = np.random.choice(ball_nums_list, p=ball_probs)
            predictions.append(np.append(sorted(selected), ball))

        return np.array(predictions)

    def _optimize_selection(self, selected, scores):
        """Optimize selection to match historical patterns."""
        sr = self.analyzer.sum_range
        oe = self.analyzer.odd_even
        hl = self.analyzer.high_low

        max_attempts = 50
        best_selection = selected.copy()
        best_score = 0

        for _ in range(max_attempts):
            current = selected.copy()

            # Calculate pattern scores
            total_sum = sum(current)
            odd_count = sum(1 for n in current if n % 2 == 1)
            low_count = sum(1 for n in current if n <= hl['mid_point'])

            score = 0

            # Sum score
            if sr['optimal_sum_range'][0] <= total_sum <= sr['optimal_sum_range'][1]:
                score += 1.0
            else:
                distance = min(
                    abs(total_sum - sr['optimal_sum_range'][0]),
                    abs(total_sum - sr['optimal_sum_range'][1])
                )
                score += max(0, 1.0 - distance / 50)

            # Odd/even score
            if odd_count == oe['most_common']:
                score += 1.0
            elif abs(odd_count - oe['most_common']) == 1:
                score += 0.7

            # High/low score
            if low_count == hl['most_common']:
                score += 1.0
            elif abs(low_count - hl['most_common']) == 1:
                score += 0.7

            if score > best_score:
                best_score = score
                best_selection = current.copy()

            if score >= 2.5:  # Good enough
                break

            # Mutate for next attempt
            idx = np.random.randint(5)
            new_num = np.random.choice(list(scores.keys()))
            while new_num in current:
                new_num = np.random.choice(list(scores.keys()))
            selected[idx] = new_num

        return best_selection

    def generate_markov_prediction(self, num_predictions):
        """Generate predictions using Markov chain transitions with diversity."""
        markov = self.analyzer.markov
        predictions = []

        # Use different seed draws for diversity (last N draws)
        num_recent = min(num_predictions + 2, len(self.analyzer.main_numbers))
        recent_draws = self.analyzer.main_numbers[-num_recent:]

        for i in range(num_predictions):
            selected = set()

            # Use different recent draws as seeds for diversity
            seed_draw = list(recent_draws[-(i % num_recent) - 1])
            np.random.shuffle(seed_draw)  # Randomize order of seed processing

            # For each number in seed draw, predict likely followers
            for seed_num in seed_draw:
                if seed_num in markov and len(selected) < 5:
                    followers = markov[seed_num]
                    if followers:
                        nums = list(followers.keys())
                        probs = list(followers.values())
                        probs = np.array(probs) / sum(probs)

                        # Apply temperature to probabilities for diversity
                        temp = 0.8 + np.random.uniform(-0.2, 0.2)
                        probs = np.power(probs, 1/temp)
                        probs = probs / probs.sum()

                        # Sample from followers
                        # Fewer per seed = more diversity
                        sample_size = min(2, len(nums))
                        for candidate in np.random.choice(nums, size=sample_size,
                                                          replace=False, p=probs):
                            if candidate not in selected and len(selected) < 5:
                                selected.add(candidate)

            # Fill remaining with weighted random from frequency
            freq = self.analyzer.frequency['main_freq']
            freq_nums = list(freq.keys())
            freq_counts = np.array(list(freq.values()), dtype=float)
            freq_probs = freq_counts / freq_counts.sum()

            while len(selected) < 5:
                num = np.random.choice(freq_nums, p=freq_probs)
                if num not in selected:
                    selected.add(num)

            # Ball prediction from frequency (weighted random)
            ball_freq = self.analyzer.frequency['ball_freq']
            ball_nums = list(ball_freq.keys())
            ball_counts = np.array(list(ball_freq.values()), dtype=float)
            ball_probs = ball_counts / ball_counts.sum()
            ball = np.random.choice(ball_nums, p=ball_probs)

            predictions.append(np.append(sorted(list(selected)[:5]), ball))

        return np.array(predictions)

    def generate_pattern_prediction(self, num_predictions):
        """Generate predictions matching historical patterns with diversity."""
        predictions = []
        used_pairs = set()  # Track used pairs to ensure diversity

        sr = self.analyzer.sum_range
        oe = self.analyzer.odd_even
        hl = self.analyzer.high_low
        pairs = self.analyzer.pairs
        freq = self.analyzer.frequency['main_freq']

        # Get top pairs with weights for selection
        top_pairs = pairs.most_common(50)  # Larger pool for diversity
        pair_weights = np.array([count for _, count in top_pairs], dtype=float)
        pair_weights = pair_weights / pair_weights.sum()

        for pred_idx in range(num_predictions):
            # Select a starter pair using weighted random (not deterministic)
            # Avoid recently used pairs for diversity
            available_indices = [
                i for i, (pair, _) in enumerate(top_pairs)
                if pair not in used_pairs
            ]
            if not available_indices:
                available_indices = list(range(len(top_pairs)))
                used_pairs.clear()

            # Weighted selection from available pairs
            avail_weights = np.array([pair_weights[i]
                                     for i in available_indices])
            avail_weights = avail_weights / avail_weights.sum()
            chosen_idx = np.random.choice(available_indices, p=avail_weights)
            starter_pair = list(top_pairs[chosen_idx][0])
            used_pairs.add(top_pairs[chosen_idx][0])

            selected = set(starter_pair)

            # Vary target patterns slightly for each prediction
            target_sum = np.random.uniform(
                sr['optimal_sum_range'][0] - sr['sum_std'] * 0.5,
                sr['optimal_sum_range'][1] + sr['sum_std'] * 0.5
            )
            # Vary odd/even target (allow ±1 from most common)
            target_odd = oe['most_common'] + np.random.choice([-1, 0, 0, 0, 1])
            target_odd = max(1, min(4, target_odd))  # Keep reasonable
            target_low = hl['most_common'] + np.random.choice([-1, 0, 0, 0, 1])
            target_low = max(1, min(4, target_low))

            # Build candidate pool with weighted random selection
            freq_items = list(freq.items())
            freq_nums = [n for n, _ in freq_items]
            freq_weights = np.array([c for _, c in freq_items], dtype=float)
            freq_weights = freq_weights / freq_weights.sum()

            # Weighted shuffle of candidates
            shuffled_indices = np.random.choice(
                len(freq_nums), size=len(freq_nums), replace=False, p=freq_weights
            )
            candidates = [freq_nums[i] for i in shuffled_indices]

            for num in candidates:
                if len(selected) >= 5:
                    break
                if num in selected:
                    continue

                test_set = selected | {num}
                current_sum = sum(test_set)
                current_odd = sum(1 for n in test_set if n % 2 == 1)
                current_low = sum(1 for n in test_set if n <= hl['mid_point'])

                # More flexible constraints
                if len(test_set) < 5:
                    selected.add(num)
                else:
                    # Final number - check constraints with some flexibility
                    sum_ok = abs(current_sum - target_sum) < sr['sum_std'] * 2
                    odd_ok = abs(current_odd - target_odd) <= 1
                    low_ok = abs(current_low - target_low) <= 1

                    if sum_ok and odd_ok and low_ok:
                        selected.add(num)
                    elif sum_ok and (odd_ok or low_ok):
                        # Accept if sum is good and at least one pattern matches
                        selected.add(num)

            # Ensure we have 5 numbers with weighted random fill
            while len(selected) < 5:
                num = np.random.choice(freq_nums, p=freq_weights)
                if num not in selected:
                    selected.add(num)

            # Ball - weighted random selection (NOT deterministic max!)
            ball_freq = self.analyzer.frequency['ball_freq']
            ball_nums = list(ball_freq.keys())
            ball_counts = np.array(list(ball_freq.values()), dtype=float)
            ball_probs = ball_counts / ball_counts.sum()
            ball = np.random.choice(ball_nums, p=ball_probs)

            predictions.append(np.append(sorted(list(selected)[:5]), ball))

        return np.array(predictions)


class DeepLearningPredictor:
    """Advanced neural network models for sequence prediction."""

    def __init__(self, data, game, seq_length=15):
        self.data = data
        self.game = game
        self.seq_length = seq_length
        self.high_range = GAMES[game]["high_range"]
        self.ball_range = GAMES[game]["featured_range"]
        self.main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]
        self.ball_col = GAMES[game]["ball"]

        self.scaler_main = MinMaxScaler(feature_range=(0, 1))
        self.scaler_ball = MinMaxScaler(feature_range=(0, 1))

        self._prepare_data()

    def _prepare_data(self):
        """Prepare data with enhanced features."""
        main_data = self.data[self.main_cols].values
        ball_data = self.data[self.ball_col].values.reshape(-1, 1)

        # Scale data
        scaled_main = self.scaler_main.fit_transform(main_data)
        scaled_ball = self.scaler_ball.fit_transform(ball_data)

        # Add derived features
        features = []
        for i, row in enumerate(main_data):
            sorted_row = sorted(row)
            feature_row = list(scaled_main[i])

            # Sum (normalized)
            feature_row.append(sum(row) / (self.high_range * 5))

            # Range (normalized)
            feature_row.append((max(row) - min(row)) / self.high_range)

            # Odd count (normalized)
            feature_row.append(sum(1 for n in row if n % 2 == 1) / 5)

            # Average delta
            deltas = np.diff(sorted_row)
            feature_row.append(np.mean(deltas) / self.high_range)

            # Ball number
            feature_row.append(scaled_ball[i][0])

            features.append(feature_row)

        self.features = np.array(features)

        # Create sequences
        X, y = [], []
        for i in range(len(self.features) - self.seq_length):
            X.append(self.features[i:i + self.seq_length])
            y.append(self.features[i + self.seq_length][:6])  # Main 5 + ball

        self.X = np.array(X)
        self.y = np.array(y)

    def build_transformer_model(self):
        """Build a Transformer-based model."""
        inputs = Input(shape=(self.seq_length, self.features.shape[1]))

        # Multi-head attention layers
        x = inputs
        for _ in range(2):
            # Self-attention
            attn_output = MultiHeadAttention(
                num_heads=4, key_dim=32, dropout=0.1
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed-forward
            ffn = Dense(64, activation='relu')(x)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(self.features.shape[1])(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)

        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(6)(x)

        model = Model(inputs, outputs)
        model.compile(
            loss='huber',
            optimizer=keras.optimizers.Adam(learning_rate=0.001)
        )
        return model

    def build_hybrid_model(self):
        """Build a hybrid LSTM + GRU + Attention model."""
        inputs = Input(shape=(self.seq_length, self.features.shape[1]))

        # LSTM branch
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(32)(lstm_out)

        # GRU branch
        gru_out = Bidirectional(GRU(64, return_sequences=True))(inputs)
        gru_out = Dropout(0.2)(gru_out)
        gru_out = GRU(32)(gru_out)

        # Attention branch
        attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
        attn_out = GlobalAveragePooling1D()(attn_out)
        attn_out = Dense(32, activation='relu')(attn_out)

        # Merge branches
        merged = Concatenate()([lstm_out, gru_out, attn_out])
        merged = BatchNormalization()(merged)
        merged = Dense(64, activation='relu',
                       kernel_regularizer=l2(0.01))(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(32, activation='relu')(merged)
        outputs = Dense(6)(merged)

        model = Model(inputs, outputs)
        model.compile(
            loss='huber',
            optimizer=keras.optimizers.Adam(learning_rate=0.0005)
        )
        return model

    def train_ensemble(self, epochs=100, verbose=0):
        """Train ensemble of models."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
        )

        # Train transformer
        self.transformer = self.build_transformer_model()
        self.transformer.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        # Train hybrid model
        self.hybrid = self.build_hybrid_model()
        self.hybrid.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

    def predict(self, num_predictions, temperature=1.2):
        """
        Generate predictions with STRONG diversity enforcement.

        Features:
        - No repeated Power Balls across predictions
        - Minimum Hamming distance between main number sets
        - Aggressive temperature scaling to break mode collapse
        """
        predictions = []
        used_balls = set()  # NO REPEATED POWER BALLS
        all_used_mains = []  # Track all main numbers for Hamming distance

        # Use widely different starting points (as positive indices)
        max_offset = min(30, len(self.X) - 1)
        seq_offsets = np.linspace(
            0, max_offset, num_predictions + 5, dtype=int)
        np.random.shuffle(seq_offsets)

        for i in range(num_predictions):
            best_pred = None
            best_diversity_score = -1

            # Try multiple candidates, pick the most diverse
            for attempt in range(20):
                # Calculate index from end (using positive indexing to avoid slice issues)
                offset_idx = (i + attempt) % len(seq_offsets)
                offset = seq_offsets[offset_idx] + attempt
                idx = len(self.X) - 1 - (offset % len(self.X))
                idx = max(0, min(idx, len(self.X) - 1))  # Clamp to valid range

                # Get single sample using proper indexing
                current_seq = self.X[idx:idx + 1].copy()

                # Get predictions from both models
                trans_pred = self.transformer.predict(current_seq, verbose=0)
                hybrid_pred = self.hybrid.predict(current_seq, verbose=0)

                # Randomize model weights more aggressively
                w1 = np.random.uniform(0.2, 0.8)
                w2 = 1.0 - w1
                ensemble_pred = trans_pred * w1 + hybrid_pred * w2

                # AGGRESSIVE temperature noise - increases with attempt
                noise_scale = temperature * 0.2 * (1 + attempt * 0.1)
                noise = np.random.normal(0, noise_scale, ensemble_pred.shape)
                ensemble_pred = ensemble_pred + noise

                # Clip and transform
                ensemble_pred = np.clip(ensemble_pred, 0, 1)
                main_pred = self.scaler_main.inverse_transform(
                    ensemble_pred[:, :5])
                ball_pred = self.scaler_ball.inverse_transform(
                    ensemble_pred[:, 5:6])

                pred = np.column_stack((main_pred, ball_pred))[0]

                # Round values
                main_nums = set(int(np.clip(np.rint(n), 1, self.high_range))
                                for n in pred[:5])
                ball_num = int(np.clip(np.rint(pred[5]), 1, self.ball_range))

                # HARD CONSTRAINT: Ball must be unique
                if ball_num in used_balls:
                    available = set(range(1, self.ball_range + 1)) - used_balls
                    if available:
                        ball_num = np.random.choice(list(available))

                # Calculate diversity score
                diversity_score = 0

                # Bonus for unique ball
                if ball_num not in used_balls:
                    diversity_score += 10

                # Bonus for different main numbers (Hamming distance)
                for prev_mains in all_used_mains:
                    diff_count = len(main_nums - prev_mains)
                    diversity_score += diff_count

                # Bonus for spread across number range
                if main_nums:
                    spread = max(main_nums) - min(main_nums)
                    diversity_score += spread / 10

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_pred = pred.copy()
                    best_pred[5] = ball_num
                    best_main_nums = main_nums

                # If we found a very diverse prediction, use it
                if diversity_score > 30 + (i * 5):
                    break

            # Use best prediction found
            if best_pred is not None:
                used_balls.add(int(np.rint(best_pred[5])))
                all_used_mains.append(best_main_nums)
                predictions.append(best_pred)

        return np.array(predictions)


def validate_predictions(predictions, game, enforce_ball_diversity=True):
    """
    Ensure predictions are valid lottery numbers with diversity enforcement.

    - All main numbers unique within each prediction
    - Optional: All Power Balls unique across predictions
    """
    validated = []
    high_range = GAMES[game]["high_range"]
    ball_range = GAMES[game]["featured_range"]
    used_balls = set()
    all_main_sets = []

    for pred in predictions:
        main = np.rint(pred[:5]).astype(int)
        ball = int(np.rint(pred[5]))

        # Clip to valid ranges
        main = np.clip(main, 1, high_range)
        ball = np.clip(ball, 1, ball_range)

        # Ensure unique main numbers
        unique = []
        used = set()
        for num in main:
            attempts = 0
            while (num in used or num < 1 or num > high_range) and attempts < 100:
                num = np.random.randint(1, high_range + 1)
                attempts += 1
            unique.append(num)
            used.add(num)

        # Enforce ball diversity if requested
        if enforce_ball_diversity and ball in used_balls:
            available = set(range(1, ball_range + 1)) - used_balls
            if available:
                ball = np.random.choice(list(available))

        # Check Hamming distance from previous predictions
        unique_set = set(unique)
        for prev_set in all_main_sets:
            overlap = len(unique_set & prev_set)
            if overlap >= 4:  # Too similar - swap one number
                to_swap = list(unique_set & prev_set)[0]
                idx = unique.index(to_swap)
                new_num = np.random.randint(1, high_range + 1)
                while new_num in unique_set:
                    new_num = np.random.randint(1, high_range + 1)
                unique[idx] = new_num
                unique_set = set(unique)

        used_balls.add(ball)
        all_main_sets.append(unique_set)
        validated.append(np.append(sorted(unique), ball))

    return np.array(validated)


def print_predictions(predictions, game, method_name, agreement=None):
    """Print formatted predictions."""
    ball_name = GAMES[game]['ball']

    print(f"\n🎰 {method_name}:")
    print("-" * 50)

    for i, pred in enumerate(predictions):
        main_str = " - ".join(f"{int(n):2d}" for n in pred[:5])
        agr_str = f" (agreement: {agreement[i]:.0%})" if agreement else ""
        print(
            f"   #{i+1}: [{main_str}]  {ball_name}: {int(pred[5]):2d}{agr_str}")


def print_analysis_report(analyzer):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)

    # Frequency Analysis
    print("\n🔥 HOT NUMBERS (Most Frequent Main Numbers):")
    for num, count in analyzer.frequency['main_freq'].most_common(10):
        dev = analyzer.frequency['main_deviation'][num]
        status = "↑" if dev > 0 else "↓"
        print(f"   #{num:2d}: {count} times ({dev:+.1%} from expected) {status}")

    # Gap Analysis
    print("\n❄️  OVERDUE NUMBERS (Largest Current Gaps):")
    sorted_gaps = sorted(analyzer.gaps['main_current'].items(),
                         key=lambda x: x[1], reverse=True)[:10]
    for num, gap in sorted_gaps:
        avg = analyzer.gaps['main_average'].get(num, gap)
        ratio = gap / max(avg, 1)
        status = "🔴" if ratio > 1.5 else "🟡" if ratio > 1.0 else "🟢"
        print(f"   #{num:2d}: {gap} draws overdue (avg gap: {avg:.1f}) {status}")

    # Hot Ball Numbers
    print(f"\n🎱 HOT {GAMES[analyzer.game]['ball'].upper()}S:")
    for num, count in analyzer.frequency['ball_freq'].most_common(5):
        print(f"   #{num:2d}: {count} times")

    # Pattern Analysis
    print("\n📈 WINNING PATTERNS:")
    sr = analyzer.sum_range
    print(
        f"   Sum Range: {sr['optimal_sum_range'][0]:.0f} - {sr['optimal_sum_range'][1]:.0f}")
    print(
        f"   (Historical: min={sr['sum_min']}, max={sr['sum_max']}, avg={sr['sum_mean']:.0f})")

    oe = analyzer.odd_even
    print(
        f"   Most Common Odd/Even: {oe['most_common']} odd, {5-oe['most_common']} even")

    hl = analyzer.high_low
    print(
        f"   Most Common High/Low: {5-hl['most_common']} high, {hl['most_common']} low")

    # Top Pairs
    print("\n👯 TOP NUMBER PAIRS:")
    for pair, count in analyzer.pairs.most_common(5):
        print(
            f"   {pair[0]:2d} & {pair[1]:2d}: appeared together {count} times")

    # Warming Numbers
    if analyzer.hot_cold_cycles and analyzer.hot_cold_cycles['warming_numbers']:
        print("\n🌡️  WARMING NUMBERS (Cold → Hot Transition):")
        print(f"   {analyzer.hot_cold_cycles['warming_numbers']}")

    print("\n" + "=" * 70)


def run_single_matrix(predictor, dl_predictor, num_preds, game, quick_mode):
    """Run a single prediction matrix and return master predictions."""
    # Statistical predictions
    stat_preds = predictor.generate_statistical_prediction(num_preds)
    stat_preds = validate_predictions(stat_preds, game)

    # Markov predictions
    markov_preds = predictor.generate_markov_prediction(num_preds)
    markov_preds = validate_predictions(markov_preds, game)

    # Pattern predictions
    pattern_preds = predictor.generate_pattern_prediction(num_preds)
    pattern_preds = validate_predictions(pattern_preds, game)

    if not quick_mode and dl_predictor is not None:
        # Deep Learning predictions
        dl_preds = dl_predictor.predict(num_preds)
        dl_preds = validate_predictions(dl_preds, game)

        # Master Ensemble with DL
        master_preds = (
            stat_preds * 0.25 +
            markov_preds * 0.20 +
            pattern_preds * 0.20 +
            dl_preds * 0.35
        )
    else:
        # Quick ensemble (statistical methods only)
        master_preds = (
            stat_preds * 0.4 +
            markov_preds * 0.3 +
            pattern_preds * 0.3
        )

    master_preds = validate_predictions(master_preds, game)
    return master_preds


def analyze_matrix_consensus(all_matrices, game, num_preds):
    """Analyze multiple matrices and generate consensus predictions."""
    high_range = GAMES[game]["high_range"]
    ball_range = GAMES[game]["featured_range"]

    # Collect all numbers across matrices
    all_main_numbers = []
    all_ball_numbers = []

    for matrix in all_matrices:
        for pred in matrix:
            main_nums = [int(np.rint(n)) for n in pred[:5]]
            ball_num = int(np.rint(pred[5]))
            all_main_numbers.extend(main_nums)
            all_ball_numbers.append(ball_num)

    # Count frequencies
    main_freq = Counter(all_main_numbers)
    ball_freq = Counter(all_ball_numbers)

    # Calculate percentages
    total_main = len(all_main_numbers)
    total_balls = len(all_ball_numbers)

    main_percentages = {num: count / total_main *
                        100 for num, count in main_freq.items()}
    ball_percentages = {num: count / total_balls *
                        100 for num, count in ball_freq.items()}

    # Generate consensus predictions
    consensus_predictions = []

    # Get top numbers by frequency
    # Pool of top numbers
    top_main = [num for num, _ in main_freq.most_common(25)]
    top_balls = [num for num, _ in ball_freq.most_common(10)]

    # Create weights based on frequency
    main_weights = np.array([main_freq.get(n, 0)
                            for n in top_main], dtype=float)
    main_weights = main_weights / main_weights.sum()

    ball_weights = np.array([ball_freq.get(n, 0)
                            for n in top_balls], dtype=float)
    ball_weights = ball_weights / ball_weights.sum()

    used_balls = set()

    for _ in range(num_preds):
        # Select 5 unique main numbers weighted by cross-matrix frequency
        selected = set()
        attempts = 0
        while len(selected) < 5 and attempts < 100:
            idx = np.random.choice(len(top_main), p=main_weights)
            num = top_main[idx]
            if num not in selected:
                selected.add(num)
            attempts += 1

        # Fill if needed
        while len(selected) < 5:
            num = np.random.randint(1, high_range + 1)
            if num not in selected:
                selected.add(num)

        # Select ball (unique across predictions)
        ball = None
        for b in top_balls:
            if b not in used_balls:
                ball = b
                used_balls.add(b)
                break
        if ball is None:
            ball = np.random.randint(1, ball_range + 1)

        consensus_predictions.append(np.append(sorted(list(selected)), ball))

    return (
        np.array(consensus_predictions),
        main_freq,
        ball_freq,
        main_percentages,
        ball_percentages,
        len(all_matrices)
    )


def print_matrix_consensus(consensus_data, game):
    """Print the matrix consensus analysis and predictions."""
    (consensus_preds, main_freq, ball_freq,
     main_pct, ball_pct, num_matrices) = consensus_data

    ball_name = GAMES[game]['ball']

    print(f"\n{'='*70}")
    print(f"🔬 MATRIX CONSENSUS ANALYSIS ({num_matrices} matrices)")
    print(f"{'='*70}")

    # Top main numbers
    print(f"\n📊 Most Frequent Main Numbers Across All Matrices:")
    for num, count in main_freq.most_common(10):
        bar = "█" * int(main_pct[num] / 2)
        print(
            f"   #{num:2d}: {count:3d} appearances ({main_pct[num]:5.1f}%) {bar}")

    # Top ball numbers
    print(f"\n🎱 Most Frequent {ball_name}s Across All Matrices:")
    for num, count in ball_freq.most_common(5):
        bar = "█" * int(ball_pct[num] / 2)
        print(
            f"   #{num:2d}: {count:3d} appearances ({ball_pct[num]:5.1f}%) {bar}")

    # Consensus predictions
    print(f"\n{'='*70}")
    print(f"⭐ CONSENSUS PREDICTIONS (Based on {num_matrices}-Matrix Analysis)")
    print(f"{'='*70}")
    print(f"\n🎰 MATRIX CONSENSUS:")
    print("-" * 50)

    for i, pred in enumerate(consensus_preds):
        main_str = " - ".join(f"{int(n):2d}" for n in pred[:5])
        # Calculate consensus strength based on how often these numbers appeared
        nums_in_pred = [int(n) for n in pred[:5]]
        avg_pct = np.mean([main_pct.get(n, 0) for n in nums_in_pred])
        ball_strength = ball_pct.get(int(pred[5]), 0)
        overall_strength = (avg_pct + ball_strength) / 2

        print(
            f"   #{i+1}: [{main_str}]  {ball_name}: {int(pred[5]):2d} (consensus: {overall_strength:.1f}%)")

    print(
        f"\n   ℹ️  Consensus % = frequency across {num_matrices} independent prediction runs")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='PowerPredict - Advanced Lottery Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py powerball -n 5           Generate 5 Powerball predictions
  python main.py megamillions -n 10 -a    Generate 10 predictions with analysis
  python main.py powerball -n 3 -q        Quick mode (statistical only)
  python main.py powerball -n 5 -m 5      Run 5 matrices for consensus analysis
        """
    )
    parser.add_argument(
        'game',
        choices=['megamillions', 'powerball'],
        help='Lottery game to predict'
    )
    parser.add_argument(
        '-n', '--num-predictions',
        type=int,
        default=5,
        help='Number of predictions to generate (default: 5)'
    )
    parser.add_argument(
        '-a', '--analyze',
        action='store_true',
        help='Show detailed statistical analysis'
    )
    parser.add_argument(
        '-q', '--quick',
        action='store_true',
        help='Quick mode - skip deep learning (faster)'
    )
    parser.add_argument(
        '-m', '--matrix',
        type=int,
        default=1,
        help='Number of matrices to run for consensus analysis (default: 1)'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    game = args.game
    num_preds = args.num_predictions
    num_matrices = args.matrix

    print(f"\n{'='*70}")
    print(f"🔮 POWERPREDICT - INTELLIGENT LOTTERY ANALYSIS SYSTEM")
    print(f"{'='*70}")
    print(f"   Game: {game.upper()}")
    print(f"   Predictions: {num_preds}")
    print(
        f"   Mode: {'Quick (Statistical)' if args.quick else 'Full (Statistical + Deep Learning)'}")
    if num_matrices > 1:
        print(f"   Matrices: {num_matrices} (consensus analysis enabled)")

    # Load data
    print(f"\n📥 Loading historical data...")
    try:
        data = pd.read_csv(
            f"https://www.texaslottery.com/export/sites/lottery/Games/"
            f"{GAMES[game]['game']}/Winning_Numbers/{game}.csv",
            header=None
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        raise SystemExit(1)

    data.columns = [
        "Game Name", "Month", "Day", "Year",
        "Num1", "Num2", "Num3", "Num4", "Num5",
        GAMES[game]["ball"], GAMES[game]["featured_ball"]
    ]
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    data = data.sort_values('Date').reset_index(drop=True)
    print(f"   ✓ Loaded {len(data)} historical drawings")

    # Statistical Analysis
    print(f"\n📊 Running comprehensive statistical analysis...")
    analyzer = LotteryAnalyzer(data, game)
    print(f"   ✓ Analysis complete")

    if args.analyze:
        print_analysis_report(analyzer)

    # Initialize predictor
    predictor = IntelligentPredictor(analyzer, data, game)

    # Generate predictions
    print(f"\n{'='*70}")
    print(f"🎯 PREDICTION RESULTS")
    print(f"{'='*70}")

    # Statistical predictions
    stat_preds = predictor.generate_statistical_prediction(num_preds)
    stat_preds = validate_predictions(stat_preds, game)
    print_predictions(stat_preds, game, "WEIGHTED STATISTICAL MODEL")

    # Markov predictions
    markov_preds = predictor.generate_markov_prediction(num_preds)
    markov_preds = validate_predictions(markov_preds, game)
    print_predictions(markov_preds, game, "MARKOV CHAIN MODEL")

    # Pattern predictions
    pattern_preds = predictor.generate_pattern_prediction(num_preds)
    pattern_preds = validate_predictions(pattern_preds, game)
    print_predictions(pattern_preds, game, "PATTERN MATCHING MODEL")

    dl_predictor = None  # Initialize for matrix feature
    dl_preds = None

    if not args.quick:
        # Deep Learning predictions
        print(f"\n🧠 Training deep learning ensemble...")
        dl_predictor = DeepLearningPredictor(data, game, seq_length=15)
        dl_predictor.train_ensemble(epochs=80, verbose=0)
        print(f"   ✓ Training complete")

        dl_preds = dl_predictor.predict(num_preds)
        dl_preds = validate_predictions(dl_preds, game)
        print_predictions(
            dl_preds, game, "DEEP LEARNING ENSEMBLE (Transformer + Hybrid)")

        # Master Ensemble
        print(f"\n{'='*70}")
        print(f"⭐ MASTER ENSEMBLE PREDICTIONS (HIGHEST CONFIDENCE)")
        print(f"{'='*70}")

        # Weighted combination of all methods
        master_preds = (
            stat_preds * 0.25 +
            markov_preds * 0.20 +
            pattern_preds * 0.20 +
            dl_preds * 0.35
        )
        master_preds = validate_predictions(master_preds, game)

        # Calculate model agreement score (NOT prediction accuracy!)
        # This measures how much the different models agree, not likelihood of winning
        agreements = []
        for i in range(num_preds):
            all_preds = np.array([
                stat_preds[i], markov_preds[i],
                pattern_preds[i], dl_preds[i]
            ])
            # Lower std = higher agreement between models
            std = np.mean(np.std(all_preds, axis=0))
            # Scale to 30-70% range (honest about uncertainty)
            agreement = 0.30 + (0.40 * max(0, 1.0 - std / 15))
            agreements.append(agreement)

        print_predictions(master_preds, game, "MASTER ENSEMBLE", agreements)
        print("\n   ℹ️  Agreement % = how much models agree, NOT win probability")
    else:
        # Quick ensemble (statistical methods only)
        print(f"\n{'='*70}")
        print(f"⭐ QUICK ENSEMBLE PREDICTIONS")
        print(f"{'='*70}")

        quick_preds = (
            stat_preds * 0.4 +
            markov_preds * 0.3 +
            pattern_preds * 0.3
        )
        quick_preds = validate_predictions(quick_preds, game)
        print_predictions(quick_preds, game, "STATISTICAL ENSEMBLE")

    # Matrix consensus analysis (if requested)
    if num_matrices > 1:
        print(
            f"\n🔄 Running {num_matrices} additional matrices for consensus...")

        # Collect all matrices (first one already done above)
        all_matrices = []

        # Get the master predictions from the first run
        if not args.quick:
            first_master = (
                stat_preds * 0.25 +
                markov_preds * 0.20 +
                pattern_preds * 0.20 +
                dl_preds * 0.35
            )
        else:
            first_master = (
                stat_preds * 0.4 +
                markov_preds * 0.3 +
                pattern_preds * 0.3
            )
        first_master = validate_predictions(first_master, game)
        all_matrices.append(first_master)

        # Run additional matrices
        for m in range(1, num_matrices):
            print(f"   Running matrix {m + 1}/{num_matrices}...")
            matrix_preds = run_single_matrix(
                predictor,
                dl_predictor if not args.quick else None,
                num_preds,
                game,
                args.quick
            )
            all_matrices.append(matrix_preds)

        print(f"   ✓ All {num_matrices} matrices complete")

        # Analyze consensus
        consensus_data = analyze_matrix_consensus(
            all_matrices, game, num_preds)
        print_matrix_consensus(consensus_data, game)

    print(f"\n{'='*70}")
    print("⚠️  DISCLAIMER: Lottery outcomes are random. These predictions are")
    print("   for entertainment only. Please gamble responsibly.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
