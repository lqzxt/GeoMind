from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import json
import re
from rich.console import Console

console = Console()

REFLECTION_PROMPT_TEMPLATE = """# Multi-Source Lithology Classification Reflection Task

## Task Description
You are a lithology classification expert. You have received classification results from multiple sources for the same batch of samples. Please analyze and synthesize these results to produce the final classification.

## Available Sources

### 1. Neural Network Predictions
{nn_predictions}

### 2. LLM Expert Reasoning
{llm_reasoning}

**LLM Classifications**: {llm_classifications}

### 3. K-Nearest Neighbors Reference
{knn_references}

### 4. Disagreement Analysis
{disagreement_report}

### 5. Transition Probability Check
{transition_report}

## Batch Data Features
{batch_features}

## Output Format
Provide your reflection analysis and final classifications:

1. Brief analysis of source agreement/disagreement for each sample
2. Final JSON output:

```json
{{
    "reasoning": "Your comprehensive reasoning",
    "classifications": ["class1", "class2", ...]
}}
```

**Important**:
- The "classifications" array must contain exactly {num_samples} elements
- Use exact class names from: {class_names}
"""


class Reflector:
    """反思器 - 提供结果验证和辅助分析"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.DATASET_NAME
        self.class_names = config.CLASS_NAMES

    def reflect(
        self,
        execution_results: Dict[str, Any],
        batch_data: pd.DataFrame,
        llm_caller: Callable[[str], str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        
        nn_predictions = execution_results.get("nn_predictions", [])
        llm_predictions = execution_results.get("llm_predictions", [])
        knn_predictions = execution_results.get("knn_predictions", [])
        
        disagreement_report = execution_results.get("disagreement_report", "")
        transition_report = execution_results.get("transition_report", "")
        
        prompt = self._build_reflection_prompt(
            nn_predictions=nn_predictions,
            llm_predictions=llm_predictions,
            knn_predictions=knn_predictions,
            disagreement_report=disagreement_report,
            transition_report=transition_report,
            batch_data=batch_data
        )
        
        try:
            response = llm_caller(prompt)
            final_predictions, reasoning = self._parse_reflection_response(
                response, len(batch_data)
            )
            
            reflection_info = {
                "prompt": prompt,
                "response": response,
                "reasoning": reasoning,
                "success": final_predictions is not None
            }
            
            if final_predictions:
                console.print(f"[green]✓ Reflection completed: {len(final_predictions)} predictions[/green]")
                return final_predictions, reflection_info
            else:
                console.print("[yellow]⚠ Reflection parsing failed, using fallback[/yellow]")
                fallback = self._get_fallback_predictions(execution_results, len(batch_data))
                reflection_info["fallback_used"] = True
                return fallback, reflection_info
                
        except Exception as e:
            console.print(f"[red]❌ Reflection failed: {e}[/red]")
            fallback = self._get_fallback_predictions(execution_results, len(batch_data))
            return fallback, {"error": str(e), "fallback_used": True}

    def _build_reflection_prompt(
        self,
        nn_predictions: List[str],
        llm_predictions: List[str],
        knn_predictions: List[str],
        disagreement_report: str,
        transition_report: str,
        batch_data: pd.DataFrame
    ) -> str:
        nn_pred_str = self._format_predictions("NN", nn_predictions)
        
        llm_pred_str = self._format_predictions("LLM", llm_predictions)
        
        knn_pred_str = self._format_predictions("KNN", knn_predictions)
        
        batch_features_str = self._format_batch_features(batch_data)
        
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            nn_predictions=nn_pred_str,
            llm_reasoning=llm_pred_str,
            llm_classifications=str(llm_predictions) if llm_predictions else "[]",
            knn_references=knn_pred_str,
            disagreement_report=disagreement_report or "No disagreement analysis available",
            transition_report=transition_report or "No transition analysis available",
            batch_features=batch_features_str,
            num_samples=len(batch_data),
            class_names=", ".join(self.class_names)
        )
        
        return prompt

    def _format_predictions(self, source: str, predictions: List[str]) -> str:
        if not predictions:
            return f"No {source} predictions available"
        
        lines = [f"{source} Predictions:"]
        for i, pred in enumerate(predictions, 1):
            lines.append(f"  Sample {i}: {pred}")
        
        return "\n".join(lines)

    def _format_batch_features(self, batch_data: pd.DataFrame) -> str:
        exclude_cols = {'DEPTH_MD', 'WELL', 'FORMATION', 'GROUP', 'LITHOLOGY', 'pred_label'}
        feature_cols = [col for col in batch_data.columns if col not in exclude_cols]
        
        display_cols = feature_cols[:10]
        
        header = "| No. | " + " | ".join(display_cols) + " |"
        separator = "|-----|" + "|".join(["------" for _ in display_cols]) + "|"
        
        lines = [header, separator]
        
        for i, (_, row) in enumerate(batch_data.iterrows(), 1):
            values = [f"{row.get(col, 0):.2f}" if pd.notna(row.get(col)) else "N/A" 
                    for col in display_cols]
            line = f"| {i} | " + " | ".join(values) + " |"
            lines.append(line)
        
        return "\n".join(lines)

    def _parse_reflection_response(
        self, 
        response: str, 
        expected_length: int
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            classifications = data.get('classifications', [])
            reasoning = data.get('reasoning', '')
            
            if len(classifications) == expected_length:
                return classifications, reasoning
            else:
                console.print(
                    f"[yellow]⚠ Classification length mismatch: "
                    f"expected {expected_length}, got {len(classifications)}[/yellow]"
                )
                return None, reasoning
        
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to parse reflection response: {e}[/yellow]")
            return None, None

    def _get_fallback_predictions(
        self,
        execution_results: Dict[str, Any],
        expected_length: int
    ) -> List[str]:
        for key in ["final_predictions", "llm_predictions", "nn_predictions", "knn_predictions"]:
            predictions = execution_results.get(key)
            if predictions and len(predictions) == expected_length:
                console.print(f"[dim]Using fallback: {key}[/dim]")
                return predictions
        
        console.print("[red]⚠ All fallback methods failed, using default class[/red]")
        default_class = self.class_names[0] if self.class_names else "Unknown"
        return [default_class] * expected_length

    def validate_results(
        self,
        execution_results: Dict[str, Any],
        batch_data: pd.DataFrame
    ) -> Dict[str, Any]:
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        expected_length = len(batch_data)
        
        final_predictions = execution_results.get("final_predictions")
        if final_predictions is None:
            validation["is_valid"] = False
            validation["issues"].append("Missing final prediction results")
        elif len(final_predictions) != expected_length:
            validation["is_valid"] = False
            validation["issues"].append(
                f"Final prediction length mismatch: expected {expected_length}, actual {len(final_predictions)}"
            )
        
        for pred_type in ["nn_predictions", "llm_predictions", "knn_predictions"]:
            predictions = execution_results.get(pred_type)
            if predictions is None:
                validation["warnings"].append(f"Missing {pred_type}")
            elif len(predictions) != expected_length:
                validation["warnings"].append(
                    f"{pred_type} length mismatch: expected {expected_length}, actual {len(predictions)}"
                )
        
        if execution_results.get("disagreement_report") is None:
            validation["warnings"].append("Missing disagreement analysis report")
        
        if execution_results.get("transition_report") is None:
            validation["warnings"].append("Missing transition probability report")
        
        if final_predictions:
            invalid_classes = []
            for i, pred in enumerate(final_predictions):
                if pred not in self.class_names and pred != "Unknown":
                    invalid_classes.append((i, pred))
            
            if invalid_classes:
                validation["warnings"].append(
                    f"Invalid classes found: {invalid_classes[:5]}"
                )
        
        return validation

    def summarize_results(
        self,
        execution_results: Dict[str, Any],
        batch_data: pd.DataFrame
    ) -> str:
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Batch Processing Summary - {self.dataset_name} Dataset")
        lines.append("=" * 60)
        
        lines.append(f"\nBatch size: {len(batch_data)} samples")
        
        plan = execution_results.get("plan", [])
        lines.append(f"Execution plan: {' → '.join(plan)}")
        
        lines.append("\n[Three-Source Prediction Results]")
        
        for pred_type, label in [
            ("nn_predictions", "NN Predictions"),
            ("llm_predictions", "LLM Predictions"),
            ("knn_predictions", "KNN Predictions")
        ]:
            predictions = execution_results.get(pred_type)
            if predictions:
                pred_dist = self._get_prediction_distribution(predictions)
                lines.append(f"\n{label}:")
                for class_name, count in pred_dist.items():
                    pct = count / len(predictions) * 100
                    lines.append(f"  - {class_name}: {count} ({pct:.1f}%)")
            else:
                lines.append(f"\n{label}: None")
        
        lines.append("\n[Final Decision Results]")
        final_predictions = execution_results.get("final_predictions")
        if final_predictions:
            final_dist = self._get_prediction_distribution(final_predictions)
            for class_name, count in final_dist.items():
                pct = count / len(final_predictions) * 100
                lines.append(f"  - {class_name}: {count} ({pct:.1f}%)")
        else:
            lines.append("  No final decision")
        
        if all(execution_results.get(k) for k in ["nn_predictions", "llm_predictions", "knn_predictions"]):
            lines.append("\n[Agreement Analysis]")
            agreement_stats = self._calculate_agreement(
                execution_results["nn_predictions"],
                execution_results["llm_predictions"],
                execution_results["knn_predictions"]
            )
            lines.append(f"  - All three agree: {agreement_stats['all_agree']} ({agreement_stats['all_agree_pct']:.1f}%)")
            lines.append(f"  - Two agree: {agreement_stats['two_agree']} ({agreement_stats['two_agree_pct']:.1f}%)")
            lines.append(f"  - None agree: {agreement_stats['none_agree']} ({agreement_stats['none_agree_pct']:.1f}%)")
        
        reasoning = execution_results.get("reflection_reasoning")
        if reasoning:
            lines.append("\n[Reflection Reasoning]")
            lines.append(f"  {reasoning[:200]}..." if len(reasoning) > 200 else f"  {reasoning}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)

    def _get_prediction_distribution(self, predictions: List[str]) -> Dict[str, int]:
        distribution = {}
        for pred in predictions:
            distribution[pred] = distribution.get(pred, 0) + 1
        return dict(sorted(distribution.items(), key=lambda x: -x[1]))

    def _calculate_agreement(
        self,
        nn_preds: List[str],
        llm_preds: List[str],
        knn_preds: List[str]
    ) -> Dict[str, Any]:
        all_agree = 0
        two_agree = 0
        none_agree = 0
        
        for i in range(len(nn_preds)):
            preds = {nn_preds[i], llm_preds[i], knn_preds[i]}
            if len(preds) == 1:
                all_agree += 1
            elif len(preds) == 2:
                two_agree += 1
            else:
                none_agree += 1
        
        total = len(nn_preds)
        
        return {
            "all_agree": all_agree,
            "all_agree_pct": all_agree / total * 100 if total > 0 else 0,
            "two_agree": two_agree,
            "two_agree_pct": two_agree / total * 100 if total > 0 else 0,
            "none_agree": none_agree,
            "none_agree_pct": none_agree / total * 100 if total > 0 else 0
        }

    def compare_with_ground_truth(
        self,
        execution_results: Dict[str, Any],
        batch_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        if self.config.TARGET_COLUMN not in batch_data.columns:
            return None
        
        ground_truth = batch_data[self.config.TARGET_COLUMN].tolist()
        
        comparison = {}
        
        for pred_type in ["nn_predictions", "llm_predictions", "knn_predictions", "final_predictions"]:
            predictions = execution_results.get(pred_type)
            if predictions and len(predictions) == len(ground_truth):
                accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == ground_truth[i]) / len(predictions)
                comparison[pred_type] = {
                    "accuracy": accuracy,
                    "correct": sum(1 for i in range(len(predictions)) if predictions[i] == ground_truth[i]),
                    "total": len(predictions)
                }
        
        return comparison


def validate_and_summarize(
    execution_results: Dict[str, Any],
    batch_data: pd.DataFrame,
    config
) -> Tuple[Dict[str, Any], str]:
    reflector = Reflector(config)
    validation = reflector.validate_results(execution_results, batch_data)
    summary = reflector.summarize_results(execution_results, batch_data)
    
    return validation, summary
