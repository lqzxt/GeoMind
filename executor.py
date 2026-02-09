from typing import Callable, Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from rich.console import Console

from tools import LithologyTool

console = Console()


class ActionExecutor:
    
    def __init__(self, tool: LithologyTool):
        self.tool = tool
    
    def _normalize_predictions_to_text(self, predictions: List) -> List[str]:
        normalized = []
        for pred in predictions:
            if isinstance(pred, (int, float, np.integer, np.floating)):
                pred_int = int(pred)
                if pred_int in self.tool.label_mapping:
                    normalized.append(self.tool.label_mapping[pred_int])
                else:
                    if 1 <= pred_int <= len(self.tool.class_names):
                        normalized.append(self.tool.class_names[pred_int - 1])
                    else:
                        normalized.append(f"Unknown_{pred_int}")
            else:
                normalized.append(str(pred))
        
        return normalized
    
    def execute(
        self,
        plan: List[str],
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        llm_caller: Callable[[str], str],
        context_length: int = 16,
        post_length: int = 16,
        silent: bool = False
    ) -> Dict[str, Any]:
        results = {
            "plan": plan,
            "tool_outputs": {},
            "nn_predictions": None,
            "llm_predictions": None,
            "knn_predictions": None,
            "disagreement_report": None,
            "transition_report": None,
            "final_predictions": None,
            "reflection_reasoning": None,
            "trend_description": None
        }
        
        trend_description = None
        nn_predictions_str = None
        neighbors_info = []
        llm_think = None
        llm_answer = None
        nn_predictions_list = []
        llm_predictions_list = []
        knn_predictions_list = []
        
        for tool_name in plan:
            
            try:
                if tool_name == "extract_trend_features":
                    trend_description, is_valid = self.tool.extract_trend_features(
                        well_data=well_data,
                        batch_start_idx=batch_start_idx,
                        batch_size=len(batch_data),
                        llm_caller=llm_caller,
                        context_length=context_length,
                        post_length=post_length
                    )
                    results["tool_outputs"]["extract_trend_features"] = {
                        "is_valid": is_valid,
                        "has_trend": trend_description is not None,
                        "format_error": False
                    }
                    results["trend_description"] = trend_description
                
                elif tool_name == "get_nn_predictions":
                    nn_predictions_str = self.tool.get_nn_predictions(
                        batch_data=batch_data,
                        include_probabilities=True
                    )
                    results["tool_outputs"]["get_nn_predictions"] = nn_predictions_str
                    
                    if "pred_label" in batch_data.columns:
                        nn_predictions_raw = batch_data["pred_label"].tolist()
                        nn_predictions_list = self._normalize_predictions_to_text(nn_predictions_raw)
                        results["nn_predictions"] = nn_predictions_list
                    
                
                elif tool_name == "retrieve_nearest_neighbors":
                    neighbors_info = []
                    for _, row in batch_data.iterrows():
                        query_sample = row[self.tool.feature_columns].values
                        neighbors = self.tool.retrieve_nearest_neighbors(
                            query_sample=query_sample,
                            num_neighbors=5
                        )
                        neighbors_info.append(neighbors)
                    
                    results["tool_outputs"]["retrieve_nearest_neighbors"] = {
                        "sample_count": len(neighbors_info)
                    }
                
                elif tool_name == "llm_classify":
                    llm_think, llm_answer = self.tool.llm_classify(
                        batch_data=batch_data,
                        llm_caller=llm_caller,
                        trend_description=trend_description,
                        nn_predictions=nn_predictions_str,
                        neighbors_info=neighbors_info
                    )
                    
                    format_error = False
                    if llm_answer:
                        llm_predictions_list, format_error = self._parse_llm_classifications(
                            llm_answer, len(batch_data)
                        )
                        results["llm_predictions"] = llm_predictions_list
                    
                    results["tool_outputs"]["llm_classify"] = {
                        "has_think": llm_think is not None,
                        "has_answer": llm_answer is not None,
                        "format_error": format_error
                    }
                
                elif tool_name == "knn_classify":
                    knn_predictions_raw = self.tool.knn_classify(
                        batch_data=batch_data,
                        num_neighbors=5
                    )
                    knn_predictions_list = self._normalize_predictions_to_text(knn_predictions_raw)
                    results["knn_predictions"] = knn_predictions_list
                    results["tool_outputs"]["knn_classify"] = {
                        "prediction_count": len(knn_predictions_list)
                    }
                
                elif tool_name == "disagreement_analyzer":
                    if nn_predictions_list and llm_predictions_list and knn_predictions_list:
                        disagreement_report = self.tool.disagreement_analyzer(
                            batch_data=batch_data,
                            nn_predictions=nn_predictions_list,
                            llm_predictions=llm_predictions_list,
                            neighbors_info=neighbors_info
                        )
                        results["disagreement_report"] = disagreement_report
                        results["tool_outputs"]["disagreement_analyzer"] = {
                            "report_length": len(disagreement_report)
                        }
                    else:
                        console.print("[yellow]⚠ Missing prediction results, skipping disagreement analysis[/yellow]")
                
                elif tool_name == "lithology_transition_check":
                    if llm_predictions_list:
                        transition_report = self.tool.lithology_transition_check(
                            predictions=llm_predictions_list,
                            batch_data=batch_data,
                            detail_level="summary",
                            include_self_transitions=False
                        )
                        results["transition_report"] = transition_report
                        results["tool_outputs"]["lithology_transition_check"] = {
                            "report_length": len(transition_report)
                        }
                    else:
                        console.print("[yellow]⚠ Missing LLM predictions, skipping transition check[/yellow]")
                
                elif tool_name == "reflection_decision":
                    if nn_predictions_list and llm_predictions_list and knn_predictions_list:
                        final_predictions, reasoning, format_error = self.tool.reflection_decision(
                            batch_data=batch_data,
                            nn_predictions=nn_predictions_list,
                            llm_predictions=llm_predictions_list,
                            knn_predictions=knn_predictions_list,
                            llm_caller=llm_caller,
                            disagreement_report=results.get("disagreement_report"),
                            transition_report=results.get("transition_report")
                        )
                        results["final_predictions"] = final_predictions
                        results["reflection_reasoning"] = reasoning
                        results["tool_outputs"]["reflection_decision"] = {
                            "prediction_count": len(final_predictions),
                            "format_error": format_error
                        }
                    else:
                        console.print("[yellow]⚠ Missing prediction results, skipping Reflection decision[/yellow]")
                
                else:
                    console.print(f"[yellow]⚠ Unknown tool: {tool_name}[/yellow]")
            
            except Exception as e:
                if not silent:
                    console.print(f"[red]❌ Tool {tool_name} execution failed: {e}[/red]")
                results["tool_outputs"][tool_name] = {"error": str(e), "traceback": str(e.__traceback__)}
        
        return results
    
    def _parse_llm_classifications(self, llm_answer: str, expected_length: int) -> tuple[List[str], bool]:
        import json
        import re
        
        try:
            if not llm_answer or not llm_answer.strip():
                console.print(f"[red]❌ LLM classification parsing failed: Empty response[/red]")
                return ["Unknown"] * expected_length, True
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_answer.strip()
            
            data = json.loads(json_str)
            classifications = data.get('classifications', [])
            
            if len(classifications) == expected_length:
                return classifications, False
            else:
                console.print(f"[yellow]⚠ LLM classification length mismatch: Expected {expected_length}, Got {len(classifications)}[/yellow]")
                return ["Unknown"] * expected_length, True
        
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ LLM classification parsing failed: JSON format error - {e}[/red]")
            console.print(f"[dim]Response content first 100 chars: {llm_answer[:100] if llm_answer else 'None'}[/dim]")
            return ["Unknown"] * expected_length, True
        except Exception as e:
            console.print(f"[red]❌ LLM classification parsing failed: {type(e).__name__}: {e}[/red]")
            return ["Unknown"] * expected_length, True


def execute_plan(
    plan: List[str],
    batch_data: pd.DataFrame,
    well_data: pd.DataFrame,
    batch_start_idx: int,
    llm_caller: Callable[[str], str],
    tool: LithologyTool,
    context_length: int = 16,
    post_length: int = 16
) -> Dict[str, Any]:
    executor = ActionExecutor(tool)
    return executor.execute(
        plan=plan,
        batch_data=batch_data,
        well_data=well_data,
        batch_start_idx=batch_start_idx,
        llm_caller=llm_caller,
        context_length=context_length,
        post_length=post_length
    )