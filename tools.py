

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from rich.console import Console

from constants import (
    FaciesConfig, Force2020Config, GeolinkConfig, SEAMConfig,
    get_dataset_config
)

console = Console()


class LithologyTool:
    
    def __init__(self, config: Type):
        self.config = config
        self.dataset_name = config.DATASET_NAME
        self.feature_columns = config.FEATURE_COLUMNS
        self.class_names = config.CLASS_NAMES
        self.label_mapping = config.LABEL_MAPPING
        self.feature_descriptions = config.FEATURE_DESCRIPTIONS
        
        self.train_data: Optional[pd.DataFrame] = None
        self.train_features: Optional[np.ndarray] = None
        self._index_built = False
        
        self.transition_matrix: Optional[Dict[str, Dict[str, float]]] = None
        self._transition_built = False
    
    def extract_trend_features(
        self,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        batch_size: int,
        llm_caller: Callable[[str], str],
        context_length: int = 16,
        post_length: int = 16
    ) -> Tuple[str, bool]:
        context_start = max(0, batch_start_idx - context_length)
        context_end = min(len(well_data), batch_start_idx + batch_size + post_length)
        actual_context = batch_start_idx - context_start
        actual_post = context_end - (batch_start_idx + batch_size)
        extended_context = well_data.iloc[context_start:context_end].copy()
        
        trend_prompt = self._build_trend_prompt(
            extended_context, actual_context, batch_size, actual_post
        )
        
        try:
            raw_response = llm_caller(trend_prompt)
            trend_description = raw_response.strip()
            is_valid = len(trend_description) > 0
            return trend_description, is_valid
        except Exception as e:
            console.print(f"[red]LLM trend analysis failed: {e}[/red]")
            return "", False
    
    def _build_trend_prompt(
        self,
        context_data: pd.DataFrame,
        context_length: int,
        batch_length: int,
        post_length: int
    ) -> str:
        batch_start = context_length
        batch_end = context_length + batch_length
        
        depth_col = self._get_depth_column()
        
        depth_list, position_list = [], []
        feature_lists = {col: [] for col in self.feature_columns}
        
        for i, (_, row) in enumerate(context_data.iterrows()):
            if i < batch_start:
                position = "Context"
            elif i < batch_end:
                target_num = i - batch_start + 1
                position = f"Target#{target_num}"
            else:
                position = "Post"
            
            depth_list.append(round(row.get(depth_col, 0.0), 1))
            for col in self.feature_columns:
                feature_lists[col].append(round(row.get(col, 0.0), 2))
            position_list.append(position)
        
        table_str = "df = pd.DataFrame({\n"
        table_str += f"    '{depth_col}': {depth_list},\n"
        for col in self.feature_columns:
            if col != depth_col:
                table_str += f"    '{col}': {feature_lists[col]},\n"
        table_str += f"    'Position': {position_list}\n}})"
        
        prompt = f"""# Well Log Data Trend Analysis Task - {self.dataset_name} Dataset

## Task Description
You are a well log data analysis expert. Please conduct detailed trend analysis on the given well log data window, focusing on the trend variations of each feature in the target sample segment (samples marked as "Target#N").

## Feature Descriptions
"""
        for feature, desc in self.feature_descriptions.items():
            prompt += f"- **{feature}**: {desc}\n"
        
        prompt += f"""
## Data

```python
{table_str}
```

## Analysis Requirements
Please organize your analysis results according to the following structure:

**Overall Trend Overview:**
[Basic characteristics and continuity analysis of the entire window]

**Detailed Feature Analysis:**
For EACH feature, provide quantitative trend descriptions including:
1. **Specific depth ranges** where changes occur
2. **Exact value changes** (e.g., "GR decreases from 85.3 to 42.1")
3. **Key turning points** with depths and values
4. **Overall trend pattern**

**Note:** Only analyze trends. DO NOT make inferences about lithology types."""
        
        return prompt
    
    def _get_depth_column(self) -> str:
        for col in self.feature_columns:
            if col.upper() in ['DEPTH', 'DEPT']:
                return col
        return 'Depth'
    
    def build_retrieval_index(self, train_data: pd.DataFrame):
        self.train_data = train_data
        self.train_features = train_data[self.feature_columns].values
        self._index_built = True
        console.print(f"[green]Retrieval index built successfully (samples: {len(train_data)})[/green]")
    
    def retrieve_nearest_neighbors(
        self,
        query_sample: np.ndarray,
        num_neighbors: int = 3,
        distance_metric: str = "euclidean"
    ) -> List[Dict]:
        if not self._index_built:
            raise ValueError("Index not built, please call build_retrieval_index() first")
        
        min_vals = self.train_features.min(axis=0)
        max_vals = self.train_features.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        norm_query = (query_sample - min_vals) / range_vals
        norm_train = (self.train_features - min_vals) / range_vals
        
        if distance_metric == "manhattan":
            distances = np.sum(np.abs(norm_train - norm_query), axis=1)
        elif distance_metric == "euclidean":
            distances = np.sqrt(np.sum((norm_train - norm_query) ** 2, axis=1))
        elif distance_metric == "cosine":
            q_norm = norm_query / (np.linalg.norm(norm_query) + 1e-8)
            t_norms = norm_train / (np.linalg.norm(norm_train, axis=1, keepdims=True) + 1e-8)
            distances = 1 - np.dot(t_norms, q_norm)
        else:
            distances = np.sqrt(np.sum((norm_train - norm_query) ** 2, axis=1))
        
        top_indices = np.argsort(distances)[:num_neighbors]
        
        neighbors = []
        for idx in top_indices:
            row = self.train_data.iloc[idx]
            label_raw = row.get(self.config.TARGET_COLUMN, 'Unknown')
            
            if isinstance(label_raw, (int, float, np.integer, np.floating)):
                label_int = int(label_raw)
                if label_int in self.label_mapping:
                    label = self.label_mapping[label_int]
                elif 1 <= label_int <= len(self.class_names):
                    label = self.class_names[label_int - 1]
                else:
                    label = f"Unknown_{label_int}"
            else:
                label = str(label_raw)
            
            dist = float(distances[idx])
            neighbors.append({
                'features': {col: float(row.get(col, 0)) for col in self.feature_columns},
                'label': label,
                'distance': dist,
                'similarity': 1 / (1 + dist)
            })
        
        return neighbors
    
    def knn_classify(
        self,
        batch_data: pd.DataFrame,
        num_neighbors: int = 5,
        distance_metric: str = "euclidean"
    ) -> List[str]:
        if not self._index_built:
            raise ValueError("Index not built, please call build_retrieval_index() first")
        
        predictions = []
        
        for _, row in batch_data.iterrows():
            query_sample = row[self.feature_columns].values
            
            neighbors = self.retrieve_nearest_neighbors(
                query_sample=query_sample,
                num_neighbors=num_neighbors,
                distance_metric=distance_metric
            )
            
            voted_label, _ = self._get_neighbor_vote(neighbors)
            
            if voted_label:
                if isinstance(voted_label, (int, float, np.integer, np.floating)):
                    voted_int = int(voted_label)
                    if voted_int in self.label_mapping:
                        voted_label = self.label_mapping[voted_int]
                    elif 1 <= voted_int <= len(self.class_names):
                        voted_label = self.class_names[voted_int - 1]
                    else:
                        voted_label = f"Unknown_{voted_int}"
                predictions.append(str(voted_label))
            else:
                predictions.append('Unknown')
        
        return predictions
    
    def get_nn_predictions(
        self,
        batch_data: pd.DataFrame,
        pred_label_col: str = "pred_label",
        include_probabilities: bool = True
    ) -> str:
        if pred_label_col not in batch_data.columns:
            return "No NN predictions available (pred_label column not found)"
        
        depth_col = self._get_depth_column()
        results = []
        
        prob_columns = [col for col in batch_data.columns if col.startswith('prob_')]
        
        for i, (_, row) in enumerate(batch_data.iterrows(), 1):
            pred_label_raw = row.get(pred_label_col, 'Unknown')
            
            if isinstance(pred_label_raw, (int, float, np.integer, np.floating)):
                pred_int = int(pred_label_raw)
                if pred_int in self.label_mapping:
                    pred_label = self.label_mapping[pred_int]
                elif 1 <= pred_int <= len(self.class_names):
                    pred_label = self.class_names[pred_int - 1]
                else:
                    pred_label = f"Unknown_{pred_int}"
            else:
                pred_label = str(pred_label_raw)
            
            depth = row.get(depth_col, 0.0)
            
            sample_str = f"Sample {i} ({depth_col}={depth:.1f}): {pred_label}"
            
            if include_probabilities and prob_columns:
                probs = []
                for prob_col in prob_columns:
                    prob_val = row.get(prob_col)
                    if prob_val is not None and prob_val > 0.05:
                        class_name = prob_col.replace('prob_', '')
                        probs.append(f"{class_name}={prob_val:.2f}")
                
                if probs:
                    sample_str += f" [{', '.join(probs)}]"
            
            results.append(sample_str)
        
        return "\n".join(results)
    
    def disagreement_analyzer(
        self,
        batch_data: pd.DataFrame,
        nn_predictions: List,
        llm_predictions: List,
        neighbors_info: Optional[List[List[Dict]]] = None
    ) -> str:
        if len(nn_predictions) != len(llm_predictions):
            raise ValueError(f"Prediction length mismatch: NN={len(nn_predictions)}, LLM={len(llm_predictions)}")
        
        if len(nn_predictions) != len(batch_data):
            raise ValueError(f"Prediction length mismatch with batch data: {len(nn_predictions)} vs {len(batch_data)}")
        
        depth_col = self._get_depth_column()
        disagreement_details = []
        disagreement_types_count = {}
        
        for i in range(len(nn_predictions)):
            nn_pred = nn_predictions[i]
            llm_pred = llm_predictions[i]
            
            neighbor_vote = None
            neighbor_confidence = None
            if neighbors_info and i < len(neighbors_info) and neighbors_info[i]:
                neighbor_vote, neighbor_confidence = self._get_neighbor_vote(neighbors_info[i])
            
            predictions_set = {nn_pred, llm_pred}
            if neighbor_vote:
                predictions_set.add(neighbor_vote)
            
            if len(predictions_set) > 1:
                disagreement_type = self._classify_disagreement_type(
                    nn_pred, llm_pred, neighbor_vote
                )
                
                confidence_info = self._extract_confidence_info(
                    batch_data.iloc[i], nn_pred, llm_pred, neighbor_confidence
                )
                
                disagreement_details.append({
                    'sample_idx': i,
                    'depth': float(batch_data.iloc[i].get(depth_col, 0)),
                    'nn_pred': nn_pred,
                    'llm_pred': llm_pred,
                    'neighbor_vote': neighbor_vote if neighbor_vote else None,
                    'disagreement_type': disagreement_type,
                    'confidence_info': confidence_info
                })
                
                disagreement_types_count[disagreement_type] = \
                    disagreement_types_count.get(disagreement_type, 0) + 1
        
        if not disagreement_details:
            return "No disagreements detected. All prediction methods (NN, LLM, and neighbors) are in full agreement."
        
        disagreement_intervals = self._identify_disagreement_intervals(disagreement_details)
        
        report = self._generate_disagreement_report(
            disagreement_details=disagreement_details,
            disagreement_intervals=disagreement_intervals,
            disagreement_types_count=disagreement_types_count,
            total_samples=len(nn_predictions)
        )
        
        return report
    
    def _get_neighbor_vote(self, neighbors: List[Dict]) -> Tuple[str, float]:
        if not neighbors:
            return None, 0.0
        
        label_votes = {}
        total_similarity = 0.0
        
        for nb in neighbors:
            label = nb['label']
            similarity = nb.get('similarity', 1.0)
            label_votes[label] = label_votes.get(label, 0) + similarity
            total_similarity += similarity
        
        voted_label = max(label_votes, key=label_votes.get)
        confidence = label_votes[voted_label] / total_similarity if total_similarity > 0 else 0.0
        
        return voted_label, confidence
    
    def _classify_disagreement_type(
        self,
        nn_pred: str,
        llm_pred: str,
        neighbor_vote: Optional[str]
    ) -> str:
        predictions = [nn_pred, llm_pred]
        if neighbor_vote:
            predictions.append(neighbor_vote)
        
        unique_preds = set(predictions)
        
        if len(unique_preds) == 1:
            return "no_disagreement"
        
        if len(predictions) == 2:
            return "nn_vs_llm"
        
        if len(unique_preds) == 3:
            return "three_way_disagreement"
        
        if nn_pred == llm_pred and nn_pred != neighbor_vote:
            return "nn_llm_agree_vs_neighbor"
        elif nn_pred == neighbor_vote and nn_pred != llm_pred:
            return "nn_neighbor_agree_vs_llm"
        elif llm_pred == neighbor_vote and llm_pred != nn_pred:
            return "llm_neighbor_agree_vs_nn"
        
        return "unknown"
    
    def _extract_confidence_info(
        self,
        sample_row: pd.Series,
        nn_pred: str,
        llm_pred: str,
        neighbor_confidence: Optional[float]
    ) -> Dict:
        confidence_info = {}
        
        nn_prob_col = f"prob_{nn_pred}"
        if nn_prob_col in sample_row.index:
            confidence_info['nn_confidence'] = float(sample_row[nn_prob_col])
        else:
            confidence_info['nn_confidence'] = None
        
        confidence_info['neighbor_confidence'] = neighbor_confidence
        
        llm_prob_col = f"llm_prob_{llm_pred}"
        if llm_prob_col in sample_row.index:
            confidence_info['llm_confidence'] = float(sample_row[llm_prob_col])
        else:
            confidence_info['llm_confidence'] = None
        
        return confidence_info
    
    def _identify_disagreement_intervals(
        self,
        disagreement_details: List[Dict]
    ) -> List[Dict]:
        if not disagreement_details:
            return []
        
        intervals = []
        current_interval = {
            'start': disagreement_details[0]['sample_idx'],
            'end': disagreement_details[0]['sample_idx'],
            'samples': [disagreement_details[0]['sample_idx']]
        }
        
        for i in range(1, len(disagreement_details)):
            curr_idx = disagreement_details[i]['sample_idx']
            prev_idx = disagreement_details[i-1]['sample_idx']
            
            if curr_idx - prev_idx <= 2:
                current_interval['end'] = curr_idx
                current_interval['samples'].append(curr_idx)
            else:
                current_interval['length'] = len(current_interval['samples'])
                intervals.append(current_interval)
                
                current_interval = {
                    'start': curr_idx,
                    'end': curr_idx,
                    'samples': [curr_idx]
                }
        
        current_interval['length'] = len(current_interval['samples'])
        intervals.append(current_interval)
        
        return intervals
    
    def _format_disagreement_type(self, dtype: str) -> str:
        type_mapping = {
            'no_disagreement': 'No disagreement (all agree)',
            'nn_vs_llm': 'NN vs LLM (no neighbor info)',
            'three_way_disagreement': 'Three-way disagreement (all different)',
            'nn_llm_agree_vs_neighbor': 'NN & LLM agree, but differ from neighbors',
            'nn_neighbor_agree_vs_llm': 'NN & Neighbors agree, but differ from LLM',
            'llm_neighbor_agree_vs_nn': 'LLM & Neighbors agree, but differ from NN',
            'unknown': 'Unknown disagreement pattern'
        }
        return type_mapping.get(dtype, dtype)
    
    def _generate_disagreement_report(
        self,
        disagreement_details: List[Dict],
        disagreement_intervals: List[Dict],
        disagreement_types_count: Dict[str, int],
        total_samples: int
    ) -> str:
        
        report_lines = []
        
        disagreement_count = len(disagreement_details)
        disagreement_rate = disagreement_count / total_samples if total_samples > 0 else 0
        
        report_lines.append("## Disagreement Analysis Report")
        report_lines.append("")
        report_lines.append(f"**Overall Statistics:**")
        report_lines.append(f"- Total samples: {total_samples}")
        report_lines.append(f"- Samples with disagreement: {disagreement_count} ({disagreement_rate:.1%})")
        report_lines.append(f"- Disagreement intervals: {len(disagreement_intervals)}")
        report_lines.append("")
        
        if disagreement_types_count:
            report_lines.append("**Disagreement Types:**")
            for dtype, count in sorted(disagreement_types_count.items(), key=lambda x: -x[1]):
                dtype_readable = self._format_disagreement_type(dtype)
                report_lines.append(f"- {dtype_readable}: {count} samples")
            report_lines.append("")
        
        if disagreement_intervals:
            report_lines.append("**Disagreement Intervals:**")
            for idx, interval in enumerate(disagreement_intervals, 1):
                report_lines.append(f"\n**Interval {idx}:** Samples {interval['start']+1} to {interval['end']+1} (length: {interval['length']})")
                
                interval_samples = set(interval['samples'])
                interval_details = [d for d in disagreement_details if d['sample_idx'] in interval_samples]
                
                for detail in interval_details:
                    sample_num = detail['sample_idx'] + 1
                    depth = detail['depth']
                    nn_pred = detail['nn_pred']
                    llm_pred = detail['llm_pred']
                    neighbor_vote = detail['neighbor_vote']
                    conf_info = detail['confidence_info']
                    
                    pred_comparison = f"NN={nn_pred}"
                    if conf_info.get('nn_confidence') is not None:
                        pred_comparison += f"({conf_info['nn_confidence']:.2f})"
                    
                    pred_comparison += f", LLM={llm_pred}"
                    if conf_info.get('llm_confidence') is not None:
                        pred_comparison += f"({conf_info['llm_confidence']:.2f})"
                    
                    if neighbor_vote:
                        pred_comparison += f", Neighbors={neighbor_vote}"
                        if conf_info.get('neighbor_confidence') is not None:
                            pred_comparison += f"({conf_info['neighbor_confidence']:.2f})"
                    
                    report_lines.append(f"  - Sample {sample_num} (Depth={depth:.1f}): {pred_comparison}")
            
            report_lines.append("")
        
        report_lines.append("**Key Observations:**")
        
        if disagreement_types_count:
            most_common_type = max(disagreement_types_count, key=disagreement_types_count.get)
            most_common_readable = self._format_disagreement_type(most_common_type)
            report_lines.append(f"- Most common disagreement pattern: {most_common_readable}")
        
        if disagreement_intervals:
            longest_interval = max(disagreement_intervals, key=lambda x: x['length'])
            report_lines.append(f"- Longest disagreement interval: Samples {longest_interval['start']+1}-{longest_interval['end']+1} ({longest_interval['length']} samples)")
        
        low_confidence_samples = []
        for detail in disagreement_details:
            nn_conf = detail['confidence_info'].get('nn_confidence')
            if nn_conf is not None and nn_conf < 0.6:
                low_confidence_samples.append(detail['sample_idx'] + 1)
        
        if low_confidence_samples:
            report_lines.append(f"- Samples with low NN confidence (<0.6): {', '.join(map(str, low_confidence_samples[:5]))}")
            if len(low_confidence_samples) > 5:
                report_lines.append(f"  (and {len(low_confidence_samples) - 5} more)")
        
        report_lines.append("")
        report_lines.append("**Recommendation:** Pay special attention to disagreement intervals, as they may indicate:")
        report_lines.append("- Lithology transition zones")
        report_lines.append("- Ambiguous or noisy data")
        report_lines.append("- Areas requiring more careful analysis")
        
        return "\n".join(report_lines)
    
    def _format_disagreement_type(self, dtype: str) -> str:
        type_mapping = {
            'no_disagreement': 'No disagreement (all agree)',
            'nn_vs_llm': 'NN vs LLM (no neighbor info)',
            'three_way_disagreement': 'Three-way disagreement (all different)',
            'nn_llm_agree_vs_neighbor': 'NN & LLM agree, but differ from neighbors',
            'nn_neighbor_agree_vs_llm': 'NN & Neighbors agree, but differ from LLM',
            'llm_neighbor_agree_vs_nn': 'LLM & Neighbors agree, but differ from NN',
            'unknown': 'Unknown disagreement pattern'
        }
        return type_mapping.get(dtype, dtype)
    
    
    def build_transition_matrix(self, train_data: pd.DataFrame, smoothing: float = 0.01):
        if self.config.TARGET_COLUMN not in train_data.columns:
            console.print(f"[red]Error: Training data missing label column {self.config.TARGET_COLUMN}[/red]")
            return
        
        transition_counts = {}
        raw_labels = train_data[self.config.TARGET_COLUMN].tolist()
        
        labels = []
        for label in raw_labels:
            if isinstance(label, (int, float, np.integer, np.floating)):
                label_int = int(label)
                if label_int in self.label_mapping:
                    labels.append(self.label_mapping[label_int])
                else:
                    labels.append(str(label))
            else:
                labels.append(str(label))
        
        for i in range(len(labels) - 1):
            from_label = labels[i]
            to_label = labels[i + 1]
            
            if from_label not in transition_counts:
                transition_counts[from_label] = {}
            
            transition_counts[from_label][to_label] = \
                transition_counts[from_label].get(to_label, 0) + 1
        
        self.transition_matrix = {}
        all_labels = set(self.class_names)
        
        for from_label in transition_counts:
            total_count = sum(transition_counts[from_label].values())
            self.transition_matrix[from_label] = {}
            
            for to_label in all_labels:
                count = transition_counts[from_label].get(to_label, 0)
                prob = (count + smoothing) / (total_count + smoothing * len(all_labels))
                self.transition_matrix[from_label][to_label] = prob
        
        self._transition_built = True
        console.print(f"[green]Transition matrix built successfully ({len(self.transition_matrix)} starting states)[/green]")
    
    def lithology_transition_check(
        self,
        predictions: List[str],
        batch_data: Optional[pd.DataFrame] = None,
        detail_level: str = "summary",
        include_self_transitions: bool = False
    ) -> str:
        if not self._transition_built:
            return "Transition matrix not built. Please call build_transition_matrix() first."
        
        if len(predictions) < 2:
            return "Sequence too short for transition analysis (need at least 2 samples)."
        
        depth_col = self._get_depth_column() if batch_data is not None else None
        
        mapped_predictions = []
        for pred in predictions:
            if isinstance(pred, (int, float, np.integer, np.floating)):
                pred_int = int(pred)
                if pred_int in self.label_mapping:
                    mapped_predictions.append(self.label_mapping[pred_int])
                else:
                    mapped_predictions.append(str(pred))
            else:
                mapped_predictions.append(str(pred))
        
        transitions = []
        cross_transitions = []
        unusual_transitions = []
        self_transition_count = 0
        total_prob = 0.0
        cross_prob_sum = 0.0
        
        for i in range(len(mapped_predictions) - 1):
            from_label = mapped_predictions[i]
            to_label = mapped_predictions[i + 1]
            
            is_self_transition = (from_label == to_label)
            
            prob = self._get_transition_probability(from_label, to_label)
            total_prob += prob
            
            depth_info = ""
            if batch_data is not None and depth_col and i < len(batch_data) - 1:
                depth_from = batch_data.iloc[i].get(depth_col, 0)
                depth_to = batch_data.iloc[i + 1].get(depth_col, 0)
                depth_info = f" (Depth {depth_from:.1f}→{depth_to:.1f})"
            
            transition_info = {
                'index': i,
                'from': from_label,
                'to': to_label,
                'prob': prob,
                'depth_info': depth_info,
                'level': self._classify_transition_level(prob),
                'is_self': is_self_transition
            }
            
            transitions.append(transition_info)
            
            if is_self_transition:
                self_transition_count += 1
            else:
                cross_transitions.append(transition_info)
                cross_prob_sum += prob
                
                if prob < 0.05:
                    unusual_transitions.append(transition_info)
        
        avg_prob = total_prob / len(transitions) if transitions else 0
        avg_cross_prob = cross_prob_sum / len(cross_transitions) if cross_transitions else 0
        
        report = self._generate_transition_report(
            transitions=transitions if include_self_transitions else cross_transitions,
            unusual_transitions=unusual_transitions,
            avg_prob=avg_prob,
            avg_cross_prob=avg_cross_prob,
            self_transition_count=self_transition_count,
            total_transitions=len(transitions),
            detail_level=detail_level,
            include_self_transitions=include_self_transitions
        )
        
        return report
    
    def _get_transition_probability(self, from_label: str, to_label: str) -> float:
        if from_label in self.transition_matrix:
            return self.transition_matrix[from_label].get(to_label, 0.01)
        return 0.01
    
    def _classify_transition_level(self, prob: float) -> str:
        if prob >= 0.4:
            return "COMMON"
        elif prob >= 0.1:
            return "MODERATE"
        elif prob >= 0.02:
            return "UNCOMMON"
        else:
            return "RARE"
    
    def _generate_transition_report(
        self,
        transitions: List[Dict],
        unusual_transitions: List[Dict],
        avg_prob: float,
        avg_cross_prob: float,
        self_transition_count: int,
        total_transitions: int,
        detail_level: str,
        include_self_transitions: bool
    ) -> str:
        report_lines = []
        
        report_lines.append("## Lithology Transition Analysis")
        report_lines.append("")
        
        if include_self_transitions:
            overall_score = avg_prob
        else:
            overall_score = avg_cross_prob if avg_cross_prob > 0 else avg_prob
        
        if overall_score >= 0.25:
            assessment = "GOOD"
        elif overall_score >= 0.15:
            assessment = "ACCEPTABLE"
        elif overall_score >= 0.08:
            assessment = "QUESTIONABLE"
        else:
            assessment = "POOR"
        
        report_lines.append("**Overall Assessment:**")
        if include_self_transitions:
            report_lines.append(f"- Average transition probability: {avg_prob:.3f} ({assessment})")
            report_lines.append(f"- Total transitions: {total_transitions}")
            report_lines.append(f"- Self-transitions: {self_transition_count} ({self_transition_count/total_transitions*100:.1f}%)")
            report_lines.append(f"- Cross-lithology transitions: {len(transitions)} ({len(transitions)/total_transitions*100:.1f}%)")
        else:
            report_lines.append(f"- Total transitions analyzed: {total_transitions}")
            report_lines.append(f"- Self-transitions (stable): {self_transition_count} ({self_transition_count/total_transitions*100:.1f}%)")
            report_lines.append(f"- Cross-lithology transitions: {len(transitions)} ({len(transitions)/total_transitions*100:.1f}%)")
            if len(transitions) > 0:
                report_lines.append(f"- Average cross-transition probability: {avg_cross_prob:.3f} ({assessment})")
        
        if len(unusual_transitions) > 0:
            report_lines.append(f"- Unusual cross-transitions: {len(unusual_transitions)} ({len(unusual_transitions)/len(transitions)*100:.1f}% of cross-transitions)")
        report_lines.append("")
        
        if unusual_transitions:
            report_lines.append("**Unusual Transitions (Probability < 0.05):**")
            for trans in unusual_transitions:
                idx = trans['index']
                from_label = trans['from']
                to_label = trans['to']
                prob = trans['prob']
                depth_info = trans['depth_info']
                level = trans['level']
                
                expected = self._get_expected_transitions(from_label, top_k=2)
                expected_str = ", ".join([f"{label} ({p:.2f})" for label, p in expected])
                
                report_lines.append(f"\n  Sample {idx+1}→{idx+2}{depth_info}: {from_label} → {to_label}")
                report_lines.append(f"  - Probability: {prob:.3f} ({level})")
                report_lines.append(f"  - More likely transitions: {expected_str}")
            
            report_lines.append("")
        
        if detail_level == "detailed" and len(transitions) > 0:
            if include_self_transitions:
                report_lines.append("**All Transitions:**")
            else:
                report_lines.append("**All Cross-Lithology Transitions:**")
            
            for trans in transitions:
                idx = trans['index']
                from_label = trans['from']
                to_label = trans['to']
                prob = trans['prob']
                level = trans['level']
                depth_info = trans['depth_info']
                
                report_lines.append(f"  {idx+1}→{idx+2}{depth_info}: {from_label} → {to_label} (p={prob:.3f}, {level})")
            
            report_lines.append("")
        elif detail_level == "detailed" and len(transitions) == 0:
            report_lines.append("**No cross-lithology transitions detected** - the sequence is entirely stable (all self-transitions).")
            report_lines.append("")
        
        report_lines.append("**Geological Interpretation:**")
        
        if len(transitions) == 0:
            report_lines.append("- The sequence is entirely stable with no lithology changes")
            report_lines.append("- This indicates a homogeneous geological unit")
            report_lines.append("- Prediction consistency is excellent")
        elif overall_score >= 0.2:
            report_lines.append("- Cross-lithology transitions follow typical geological patterns")
            report_lines.append("- Most lithology changes are geologically reasonable")
            if self_transition_count > 0:
                stability_pct = self_transition_count / total_transitions * 100
                report_lines.append(f"- Sequence shows good stability ({stability_pct:.0f}% self-transitions)")
        elif overall_score >= 0.1:
            report_lines.append("- Some cross-lithology transitions are unusual")
            report_lines.append("- Review these transitions for potential prediction errors")
        else:
            report_lines.append("- Many cross-lithology transitions are rare or unusual")
            report_lines.append("- This may indicate:")
            report_lines.append("  * Prediction errors at lithology boundaries")
            report_lines.append("  * Unusual geological setting")
            report_lines.append("  * Rapid environmental changes")
        
        if unusual_transitions:
            report_lines.append("")
            report_lines.append("**Recommendation:** Carefully review samples with unusual transitions, especially:")
            for trans in unusual_transitions[:3]:
                report_lines.append(f"- Sample {trans['index']+1}→{trans['index']+2}: {trans['from']} → {trans['to']}")
        
        return "\n".join(report_lines)
    
    def _get_expected_transitions(self, from_label: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if from_label not in self.transition_matrix:
            return []
        
        transitions = self.transition_matrix[from_label]
        sorted_transitions = sorted(transitions.items(), key=lambda x: -x[1])
        return sorted_transitions[:top_k]
    
    def reflection_decision(
        self,
        batch_data: pd.DataFrame,
        nn_predictions: List[str],
        llm_predictions: List[str],
        knn_predictions: List[str],
        llm_caller: Callable[[str], str],
        disagreement_report: Optional[str] = None,
        transition_report: Optional[str] = None
    ) -> Tuple[List[str], str, bool]:
        prompt = self._build_reflection_prompt(
            batch_data=batch_data,
            nn_predictions=nn_predictions,
            llm_predictions=llm_predictions,
            knn_predictions=knn_predictions,
            disagreement_report=disagreement_report,
            transition_report=transition_report
        )
        
        try:
            response = llm_caller(prompt)
            
            final_predictions, reasoning, format_error = self._parse_reflection_response(
                response=response,
                expected_length=len(batch_data)
            )
            
            if final_predictions is None:
                console.print("[yellow]Reflection parsing failed, using majority voting as fallback[/yellow]")
                final_predictions = self._majority_vote_fallback(
                    nn_predictions, llm_predictions, knn_predictions
                )
                reasoning = "Fallback to majority voting due to parsing failure"
                format_error = True
            
            return final_predictions, reasoning, format_error
            
        except Exception as e:
            console.print(f"[red]Reflection call failed: {e}[/red]")
            final_predictions = self._majority_vote_fallback(
                nn_predictions, llm_predictions, knn_predictions
            )
            return final_predictions, f"Fallback to majority voting due to error: {e}", True
    
    def _build_reflection_prompt(
        self,
        batch_data: pd.DataFrame,
        nn_predictions: List[str],
        llm_predictions: List[str],
        knn_predictions: List[str],
        disagreement_report: Optional[str],
        transition_report: Optional[str]
    ) -> str:
        
        depth_col = self._get_depth_column()
        
        prompt = f"""# Reflection Decision Task - {self.dataset_name} Dataset

## Task Description
You are an expert geologist making final classification decisions. Three prediction methods have been applied:
1. **Neural Network (NN)**: Deep learning model
2. **Large Language Model (LLM)**: Reasoning-based classification
3. **K-Nearest Neighbors (KNN)**: Case-based classification

Your task is to select the best prediction for each sample by considering:
- Prediction agreement/disagreement
- Model confidence
- Geological transition patterns
- Feature-lithology alignment

## Available Classes
{', '.join(self.class_names)}

## Predictions Comparison

"""
        
        prompt += "| Sample | Depth | NN | LLM | KNN | Agreement |\n"
        prompt += "|--------|-------|----|----|-----|----------|\n"
        
        for i in range(len(nn_predictions)):
            sample_num = i + 1
            depth = batch_data.iloc[i].get(depth_col, 0) if depth_col else 0
            nn_pred = nn_predictions[i]
            llm_pred = llm_predictions[i]
            knn_pred = knn_predictions[i]
            
            preds = [nn_pred, llm_pred, knn_pred]
            unique_preds = set(preds)
            if len(unique_preds) == 1:
                agreement = "✓ All agree"
            elif len(unique_preds) == 2:
                agreement = "⚠ 2/3 agree"
            else:
                agreement = "✗ All differ"
            
            prompt += f"| {sample_num} | {depth:.1f} | {nn_pred} | {llm_pred} | {knn_pred} | {agreement} |\n"
        
        prompt += "\n"
        
        if disagreement_report:
            prompt += disagreement_report + "\n\n"
        
        if transition_report:
            prompt += transition_report + "\n\n"
        
        prompt += """## Output Format

Please provide your final decisions in JSON format:

```json
{
    "decisions": ["class1", "class2", ...],
    "reasoning": "Brief explanation of your decision-making process"
}
```

**Important:**
- The "decisions" array must contain exactly """ + str(len(nn_predictions)) + """ elements
- Use exact class names from the available classes list
- Consider all available information (agreement, transitions, confidence)
- Prioritize geological reasonableness over simple majority voting
"""
        
        return prompt
    
    def _parse_reflection_response(
        self,
        response: str,
        expected_length: int
    ) -> Tuple[Optional[List[str]], Optional[str], bool]:
        import json
        import re
        
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            decisions = data.get('decisions', data.get('classifications', []))
            reasoning = data.get('reasoning', '')
            
            if len(decisions) != expected_length:
                console.print(f"[red]Reflection decision length mismatch: expected {expected_length}, got {len(decisions)}[/red]")
                return None, None, True
            
            return decisions, reasoning, False
            
        except Exception as e:
            console.print(f"[red]Reflection response parsing failed: {e}[/red]")
            return None, None, True
    
    def _majority_vote_fallback(
        self,
        nn_predictions: List[str],
        llm_predictions: List[str],
        knn_predictions: List[str]
    ) -> List[str]:
        final_predictions = []
        
        for i in range(len(nn_predictions)):
            votes = [nn_predictions[i], llm_predictions[i], knn_predictions[i]]
            vote_counts = {}
            for vote in votes:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
            
            final_pred = max(vote_counts, key=vote_counts.get)
            final_predictions.append(final_pred)
        
        return final_predictions
    
    def llm_classify(
        self,
        batch_data: pd.DataFrame,
        llm_caller: Callable[[str], str],
        trend_description: Optional[str] = None,
        nn_predictions: Optional[str] = None,
        neighbors_info: Optional[List[List[Dict]]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        prompt = self._build_classification_prompt(
            batch_data=batch_data,
            trend_description=trend_description,
            nn_predictions=nn_predictions,
            neighbors_info=neighbors_info
        )
        
        try:
            response = llm_caller(prompt)
            think, answer = self._extract_think_answer(response)
            return think, answer
        except Exception as e:
            console.print(f"[red]LLM classification call failed: {e}[/red]")
            return None, None
    
    def _build_classification_prompt(
        self,
        batch_data: pd.DataFrame,
        trend_description: Optional[str] = None,
        nn_predictions: Optional[str] = None,
        neighbors_info: Optional[List[List[Dict]]] = None
    ) -> str:
        
        prompt = f"# Well Log Data Classification Task - {self.dataset_name} Dataset\n\n"
        
        prompt += f"""## Task Description
You are a well log data analysis expert. Please classify each depth point into one of the following {self.config.NUM_CLASSES} categories based on the provided well log feature data:

"""
        for i, class_name in enumerate(self.class_names, 1):
            prompt += f"{i}. **{class_name}**\n"
        
        prompt += "\n"
        
        prompt += "## Feature Descriptions\n"
        for feature, description in self.feature_descriptions.items():
            prompt += f"- **{feature}**: {description}\n"
        prompt += "\n"
        
        prompt += "## Well Log Data to Classify\n\n"
        prompt += self._build_data_table(batch_data)
        
        if trend_description:
            prompt += "\n## Contextual Trend Analysis\n"
            prompt += trend_description + "\n"
        
        if nn_predictions:
            prompt += "\n## Neural Network Predictions (Reference)\n"
            prompt += nn_predictions + "\n"
        
        if neighbors_info:
            prompt += "\n## Nearest Neighbors from Training Data\n"
            for i, neighbors in enumerate(neighbors_info, 1):
                if neighbors:
                    prompt += f"\n**Sample {i} Neighbors:**\n"
                    for j, nb in enumerate(neighbors, 1):
                        prompt += f"  {j}. Label: {nb['label']}, Similarity: {nb['similarity']:.3f}\n"
        
        prompt += f"""
## Output Format
Please provide your classifications in JSON format:

```json
{{
    "classifications": ["class1", "class2", ...]
}}
```

**Important**:
- The "classifications" array must contain exactly {len(batch_data)} elements.
- Use exact class names from the list above.
- Consider trend patterns and segment continuity.
"""
        
        return prompt
    
    def _build_data_table(self, batch_data: pd.DataFrame) -> str:
        depth_col = self._get_depth_column()
        
        header_cols = [depth_col] + [col for col in self.feature_columns if col != depth_col]
        table = "| No. | " + " | ".join(header_cols) + " |\n"
        table += "|-----|" + "|".join(["-------"] * len(header_cols)) + "|\n"
        
        for i, (_, row) in enumerate(batch_data.iterrows(), 1):
            values = [f"{row.get(col, 0):.2f}" for col in header_cols]
            table += f"| {i} | " + " | ".join(values) + " |\n"
        
        return table
    
    def _extract_think_answer(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        import re
        
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
            answer = response.replace(think_match.group(0), '').strip()
            return think, answer
        else:
            return None, response
