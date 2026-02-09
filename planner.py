from typing import Callable, Dict, List, Optional, Type
import pandas as pd
import json
from rich.console import Console

console = Console()

TOOL_DESCRIPTIONS = """
## Available Tools

1. **extract_trend_features**: Extract well log curve trend features using LLM analysis.
   - Input: batch_data, context_data
   - Output: trend_description (str)
   - Use when: Need contextual trend analysis for better classification.

2. **get_nn_predictions**: Get neural network model predictions for current batch.
   - Input: batch_data
   - Output: prediction summary with probabilities
   - Use when: Want to see what the neural network predicts as reference.

3. **retrieve_nearest_neighbors**: Retrieve similar samples from training data using KNN.
   - Input: query_sample features
   - Output: list of similar samples with labels and distances
   - Use when: Need reference samples for comparison.

4. **llm_classify**: LLM-based classification using expert knowledge and domain reasoning.
   - Input: batch_data, trend_description (optional), nn_predictions (optional), neighbors (optional)
   - Output: (reasoning, classification) - reasoning process and classification result
   - Use when: Ready to perform autonomous classification.

5. **knn_classify**: KNN-based classification using nearest neighbor voting.
   - Input: batch_data, neighbors
   - Output: classification result based on neighbor labels
   - Use when: Want traditional KNN classification result.

6. **disagreement_analyzer**: Analyze disagreements between NN, LLM, and KNN predictions.
   - Input: nn_pred, llm_pred, knn_pred
   - Output: disagreement analysis report
   - Use when: Multiple models disagree and need to understand why.

7. **lithology_transition_check**: Check lithology transition probabilities based on geological rules.
   - Input: current_lithology, previous_lithology, depth_change
   - Output: transition probability and geological reasoning
   - Use when: Need to validate transitions against geological knowledge.

8. **reflection_decision**: Comprehensive decision making by integrating all model outputs.
   - Input: all previous tool outputs
   - Output: final classification decision with confidence
   - Use when: Ready to make final decision after gathering all evidence.
"""

PLANNING_PROMPT_TEMPLATE = """# Lithology Classification Planning Task

You are a planning agent for lithology classification from well log data. Based on the current batch data characteristics, decide which tools to use and in what order.

{tool_descriptions}

## Current Batch Information

- Dataset: {dataset_name}
- Batch size: {batch_size} samples
- Depth range: {depth_min:.1f}m - {depth_max:.1f}m
- Feature statistics:
{feature_stats}
- Missing ratio: {missing_ratio:.2%}

## Planning Guidelines

1. **Start with feature extraction**: Use `extract_trend_features` to understand log patterns.
2. **Gather predictions**: Use `get_nn_predictions` and `retrieve_nearest_neighbors` for reference.
3. **LLM reasoning**: Use `llm_classify` for autonomous classification.
4. **Traditional method**: Use `knn_classify` for comparison.
5. **Analyze disagreements**: If models might disagree, use `disagreement_analyzer`.
6. **Validate transitions**: Use `lithology_transition_check` for geological consistency.
7. **Final decision**: Always end with `reflection_decision` for comprehensive decision.

## Output Format

Provide your plan as a JSON object with reasoning and tool list:

```json
{{
    "reasoning": "Brief explanation of why these tools are needed",
    "plan": ["tool1", "tool2", "tool3", ...]
}}
```

Only include tools that are necessary. Keep the plan minimal and efficient.
"""


class Planner:
    def __init__(self, config: Type):
        self.config = config
        self.dataset_name = config.DATASET_NAME

    def create_plan(
        self,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        use_llm_planning: bool = True,
        llm_caller: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        
        if use_llm_planning and llm_caller:
            try:
                return self._llm_plan(batch_data, well_data, llm_caller)
            except Exception as e:
                console.print(f"[yellow]⚠ LLM planning failed: {e}, using default plan[/yellow]")
        
        return self._default_plan()

    def _default_plan(self) -> List[str]:
        return [
            "extract_trend_features",
            "get_nn_predictions",
            "retrieve_nearest_neighbors",
            "llm_classify",
            "knn_classify",
            "disagreement_analyzer",
            "lithology_transition_check",
            "reflection_decision"
        ]

    def _llm_plan(
        self,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        llm_caller: Callable[[str], str]
    ) -> List[str]:
        prompt = self._build_planning_prompt(batch_data, well_data)
        
        try:
            response = llm_caller(prompt)
            plan = self._parse_plan_response(response)
            if plan:
                return plan
        except Exception as e:
            console.print(f"[yellow]⚠ Planning failed: {e}, using default plan[/yellow]")
        
        return self._default_plan()

    def _build_planning_prompt(
        self,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame
    ) -> str:
        feature_stats_lines = []
        feature_cols = [col for col in batch_data.columns 
                        if col not in ['DEPTH_MD', 'WELL', 'FORMATION', 'GROUP', 'LITHOLOGY']]
        
        for col in feature_cols[:10]:
            if col in batch_data.columns:
                mean_val = batch_data[col].mean()
                std_val = batch_data[col].std()
                if pd.notna(mean_val) and pd.notna(std_val):
                    feature_stats_lines.append(f"  - {col}: mean={mean_val:.2f}, std={std_val:.2f}")
                else:
                    feature_stats_lines.append(f"  - {col}: mean=N/A, std=N/A")
        
        feature_stats = "\n".join(feature_stats_lines) if feature_stats_lines else "  - No feature statistics available"
        
        missing_ratio = batch_data.isnull().sum().sum() / (len(batch_data) * len(batch_data.columns))
        
        depth_col = 'DEPTH_MD' if 'DEPTH_MD' in batch_data.columns else 'DEPTH'
        depth_min = batch_data[depth_col].min() if depth_col in batch_data.columns else 0
        depth_max = batch_data[depth_col].max() if depth_col in batch_data.columns else 0
        
        prompt = PLANNING_PROMPT_TEMPLATE.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            dataset_name=self.dataset_name,
            batch_size=len(batch_data),
            depth_min=depth_min,
            depth_max=depth_max,
            feature_stats=feature_stats,
            missing_ratio=missing_ratio
        )
        
        return prompt

    def _parse_plan_response(self, response: str) -> List[str]:
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            if isinstance(data, dict) and "plan" in data:
                plan = data["plan"]
                if isinstance(plan, list):
                    valid_tools = {
                        "extract_trend_features",
                        "get_nn_predictions",
                        "retrieve_nearest_neighbors",
                        "llm_classify",
                        "knn_classify",
                        "disagreement_analyzer",
                        "lithology_transition_check",
                        "reflection_decision"
                    }
                    validated_plan = [tool for tool in plan if tool in valid_tools]
                    
                    if validated_plan:
                        if "reasoning" in data:
                            console.print(f"[dim]Planning reasoning: {data['reasoning']}[/dim]")
                        return validated_plan
        
        except json.JSONDecodeError:
            tools = [t.strip() for t in response.split(',')]
            valid_tools = {
                "extract_trend_features",
                "get_nn_predictions",
                "retrieve_nearest_neighbors",
                "llm_classify",
                "knn_classify",
                "disagreement_analyzer",
                "lithology_transition_check",
                "reflection_decision"
            }
            validated_plan = [tool for tool in tools if tool in valid_tools]
            if validated_plan:
                return validated_plan
        
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to parse plan response: {e}[/yellow]")
        
        return []


def create_plan(
    batch_data: pd.DataFrame,
    well_data: pd.DataFrame,
    config: Type,
    use_llm_planning: bool = True,
    llm_caller: Optional[Callable[[str], str]] = None
) -> List[str]:

    planner = Planner(config)
    return planner.create_plan(
        batch_data=batch_data,
        well_data=well_data,
        use_llm_planning=use_llm_planning,
        llm_caller=llm_caller
    )
