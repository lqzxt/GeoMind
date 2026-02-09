from __future__ import annotations

import os
import json
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, cast
from collections import Counter

import pandas as pd
import numpy as np
from rich.console import Console

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

import agentlightning as agl
from agentlightning.types import AttemptedRollout

from data_loader import WellDataLoader
from tools import LithologyTool
from planner import Planner
from executor import ActionExecutor
from reflector import Reflector
from LLM_caller import get_reward_response
from constants import get_dataset_config
from agentlightning import emit_reward

console = Console()


class WellLogTask(TypedDict):
    dataset_name: str
    well_id: str
    batch_idx: int
    batch_data: Dict[str, Any]
    batch_start_idx: int
    ground_truth: List[int]


class LitWellLogClassifier(agl.LitAgent[WellLogTask]):
    
    def __init__(
        self,
        dataset_names: List[str] = None,
        batch_size: int = 16,
        stride: int = 4,
        context_length: int = 16,
        post_length: int = 16,
        use_llm_planning: bool = True,
        use_reflection: bool = True
    ):
        super().__init__()
        
        console.print("[bold blue]Initializing Well Log Classification Agent (RL Training Mode)...[/bold blue]")
        
        if dataset_names is None:
            dataset_names = ['facies']
        self.dataset_names = dataset_names
        self.configs = {name: get_dataset_config(name) for name in dataset_names}
        
        self.batch_size = batch_size
        self.stride = stride
        self.context_length = context_length
        self.post_length = post_length
        self.use_llm_planning = use_llm_planning
        self.use_reflection = use_reflection
        
        self.data_loaders = {}
        self.tools = {}
        self.planners = {}
        self.executors = {}
        self.reflectors = {}
        
        for dataset_name in dataset_names:
            config = self.configs[dataset_name]
            self.data_loaders[dataset_name] = WellDataLoader(config)
            self.tools[dataset_name] = LithologyTool(config)
            self.planners[dataset_name] = Planner(config)
            self.executors[dataset_name] = ActionExecutor(self.tools[dataset_name])
            self.reflectors[dataset_name] = Reflector(config)
        
        self.wells_by_dataset = {name: {} for name in dataset_names}
        
        self.well_to_dataset = {}
        
        console.print(f"[green]✓ Supported datasets: {', '.join(dataset_names)}[/green]")
        console.print("[bold green]✓ Agent initialization complete (Multi-dataset mixed mode)[/bold green]")

    def _save_llm_log(self, category: str, prompt: str, response: str):
        try:
            call_id = str(uuid.uuid4())
            
            base_dir = os.path.join("running_log", category, call_id)
            os.makedirs(base_dir, exist_ok=True)
            
            with open(os.path.join(base_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(str(prompt))
                
            with open(os.path.join(base_dir, "response.txt"), "w", encoding="utf-8") as f:
                f.write(str(response))
            
            console.print(f"[dim]✓ Log saved: {category}/{call_id[:8]}...[/dim]")
                
        except Exception as e:
            console.print(f"[red]❌ Log save failed ({category}): {str(e)}[/red]")
            import traceback
            traceback.print_exc()
    
    def load_training_data(
        self,
        dataset_splits: Dict[str, str] = None,
        model_name: str = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        if dataset_splits is None:
            dataset_splits = {name: 'train' for name in self.dataset_names}
        
        all_data = {}
        
        for dataset_name in self.dataset_names:
            split = dataset_splits.get(dataset_name, 'train')
            config = self.configs[dataset_name]
            
            directory = config.get_model_path(split, model_name)
            
            if model_name:
                console.print(f"[blue]Loading {dataset_name} dataset ({split}_{model_name})...[/blue]")
            else:
                console.print(f"[blue]Loading {dataset_name} dataset ({split})...[/blue]")
            
            wells = self.data_loaders[dataset_name].load_from_directory(
                directory=directory,
                split=split
            )
            all_data[dataset_name] = wells
            
            self.data_loaders[dataset_name].wells = wells
            
            console.print(f"[green]✓ {dataset_name}: {len(wells)} wells[/green]")
        
        return all_data
    
    def load_training_and_validation_data(
        self,
        model_name: str = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        if model_name:
            console.print(f"[cyan]Loading combined training + validation data (Model: {model_name})...[/cyan]")
        else:
            console.print("[cyan]Loading combined training + validation data (Original data)...[/cyan]")
        
        all_data = {}
        
        for dataset_name in self.dataset_names:
            console.print(f"\n[blue]Processing {dataset_name} dataset...[/blue]")
            config = self.configs[dataset_name]
            
            train_dir = config.get_model_path('train', model_name)
            if model_name:
                console.print(f"  Loading train_{model_name}...")
            else:
                console.print(f"  Loading train...")
            
            train_wells = self.data_loaders[dataset_name].load_from_directory(
                directory=train_dir,
                split='train'
            )
            
            val_dir = config.get_model_path('val', model_name)
            if model_name:
                console.print(f"  Loading val_{model_name}...")
            else:
                console.print(f"  Loading val...")
            
            val_wells = self.data_loaders[dataset_name].load_from_directory(
                directory=val_dir,
                split='val'
            )
            
            combined_wells = {**train_wells, **val_wells}
            all_data[dataset_name] = combined_wells
            
            self.data_loaders[dataset_name].wells = combined_wells
            
            console.print(
                f"[green]✓ {dataset_name}: "
                f"{len(train_wells)} training wells + {len(val_wells)} validation wells = "
                f"{len(combined_wells)} total wells[/green]"
            )
        
        return all_data
    
    def build_retrieval_index(
        self,
        train_data: Dict[str, Dict[str, pd.DataFrame]] = None,
        dataset_splits: Dict[str, str] = None
    ):
        if train_data is None:
            train_data = self.load_training_data(dataset_splits)
        
        for dataset_name in self.dataset_names:
            wells = train_data.get(dataset_name, {})
            
            if not wells:
                console.print(f"[yellow]Warning: {dataset_name} has no data, skipping index build[/yellow]")
                continue
            
            train_data_list = list(wells.values())
            if train_data_list:
                dataset_train_data = pd.concat(train_data_list, ignore_index=True)
                console.print(f"[blue]{dataset_name}: Building retrieval index using {len(dataset_train_data)} samples...[/blue]")
                
                self.tools[dataset_name].build_retrieval_index(dataset_train_data)
                self.tools[dataset_name].build_transition_matrix(dataset_train_data)
                
                console.print(f"[green]✓ {dataset_name} index and transition matrix build complete[/green]")
    
    def rollout(
        self,
        task: WellLogTask,
        resources: agl.NamedResources,
        rollout_info: agl.Rollout,
    ) -> float | None:
        dataset_name = task.get("dataset_name", "unknown")
        well_id = task.get("well_id", "unknown")
        batch_idx = task.get("batch_idx", -1)
        
        try:
            attempted_rollout = cast(AttemptedRollout, rollout_info)
            
            if dataset_name not in self.tools:
                console.print(f"[red]Error: Unknown dataset {dataset_name}[/red]")
                return -0.5
            
            tool = self.tools[dataset_name]
            planner = self.planners[dataset_name]
            executor = self.executors[dataset_name]
            reflector = self.reflectors[dataset_name]
            config = self.configs[dataset_name]
            
            if not tool._index_built:
                console.print(f"[yellow]Detected {dataset_name} index not built, attempting auto-build...[/yellow]")
                train_data = self.load_training_data({dataset_name: 'train'})
                self.build_retrieval_index(train_data)
                if not tool._index_built:
                    console.print(f"[red]Error: Unable to build {dataset_name} index[/red]")
                    return -0.5
            
            llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
            
            batch_df = pd.DataFrame(task["batch_data"])
            batch_start_idx = task["batch_start_idx"]
            ground_truth = task.get("ground_truth", [])
            
            data_loader = self.data_loaders[dataset_name]
            well_data = data_loader.get_well(well_id)
            if well_data is None:
                console.print(f"[red]Error: Data for well {well_id} not found[/red]")
                return -0.5
            
            llm_caller = self._create_llm_caller(llm, attempted_rollout)
            
            console.print(f"[blue] Starting classification: dataset={dataset_name}, well={well_id}, batch={batch_idx}[/blue]")
            
            execution_results = self._run_classification_with_rewards(
                dataset_name=dataset_name,
                batch_data=batch_df,
                well_data=well_data,
                batch_start_idx=batch_start_idx,
                ground_truth=ground_truth,
                llm_caller=llm_caller,
                attempted_rollout=attempted_rollout,
                planner=planner,
                executor=executor,
                reflector=reflector,
                config=config
            )
            
            final_predictions = execution_results.get("final_predictions", [])
            if final_predictions and ground_truth and len(final_predictions) == len(ground_truth):
                final_acc = self._calculate_accuracy(final_predictions, ground_truth, config)
                console.print(
                    f"[green]✓ {dataset_name}/{well_id} batch {batch_idx}: "
                    f"Final accuracy {final_acc:.4f}[/green]"
                )
                return final_acc
            else:
                console.print(f"[yellow]⚠ {dataset_name}/{well_id} batch {batch_idx}: Unable to calculate accuracy[/yellow]")
                return 0.0
            
        except Exception as e:
            console.print(
                f"[red]❌ Rollout Exception (dataset={dataset_name}, well={well_id}, batch={batch_idx}): "
                f"{type(e).__name__}: {str(e)}[/red]"
            )
            console.print(f"[red]Full Stack Trace:[/red]\n{traceback.format_exc()}")
            return -0.5
    
    def _create_llm_caller(
        self,
        llm: agl.LLM,
        attempted_rollout: AttemptedRollout
    ) -> Callable[[str], str]:
        endpoint = llm.get_base_url(
            attempted_rollout.rollout_id,
            attempted_rollout.attempt.attempt_id
        )
        langchain_llm = init_chat_model(
            llm.model,
            model_provider="openai",
            openai_api_base=endpoint,
            openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=1,
            max_retries=3,
            max_tokens=5000,
        )
        
        def llm_caller(content: str) -> str:
            try:
                result = langchain_llm.invoke([HumanMessage(content=content)])
                response_content = result.content
                
                if isinstance(response_content, bytes):
                    response_content = response_content.decode('utf-8', errors='replace')
                elif isinstance(response_content, list):
                    response_content = ''.join(str(item) for item in response_content)
                elif not isinstance(response_content, str):
                    response_content = str(response_content)
                
                if not response_content or not response_content.strip():
                    console.print("[yellow]⚠ LLM returned empty response[/yellow]")
                    response_content = "Error: Empty response from LLM"
                
                console.print(f"[dim]LLM Call: Type={type(result.content).__name__}, Length={len(response_content)}[/dim]")
                
                self._save_llm_log("agent_execution", content, response_content)
                
                return response_content
            except Exception as e:
                console.print(f"[red]LLM Call Failed: {str(e)}[/red]")
                raise
        
        return llm_caller

    
    def _run_classification_with_rewards(
        self,
        dataset_name: str,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        ground_truth: List[int],
        llm_caller: Callable[[str], str],
        attempted_rollout: AttemptedRollout,
        planner: Planner,
        executor: ActionExecutor,
        reflector: Reflector,
        config: Any
    ) -> Dict[str, Any]:
        results = {
            "trend_reward": 0.0,
            "llm_acc_reward": 0.0,
            "reflection_reward": 0.0,
            "format_penalty": 0.0
        }
        
        plan = planner.create_plan(
            batch_data=batch_data,
            well_data=well_data,
            llm_caller=llm_caller if self.use_llm_planning else None,
            use_llm_planning=self.use_llm_planning
        )
        
        execution_results = executor.execute(
            plan=plan,
            batch_data=batch_data,
            well_data=well_data,
            batch_start_idx=batch_start_idx,
            llm_caller=llm_caller,
            context_length=self.context_length,
            post_length=self.post_length,
            silent=True
        )
        
        trend_description = execution_results.get("trend_description")
        trend_format_error = execution_results.get("tool_outputs", {}).get("extract_trend_features", {}).get("format_error", False)
        
        if trend_description:
            trend_reward = self._evaluate_trend_quality(
                trend_description=trend_description,
                batch_data=batch_data,
                well_data=well_data,
                batch_start_idx=batch_start_idx,
                config=config
            )
            results["trend_reward"] = trend_reward
            
            if trend_format_error:
                format_penalty = -0.3
                results["format_penalty"] += format_penalty
                emit_reward(format_penalty)
                console.print(f"[yellow]⚠ Trend analysis format error penalty: {format_penalty:.4f}[/yellow]")
            
            emit_reward(trend_reward)
            console.print(f"[dim]Trend analysis reward: {trend_reward:.4f}[/dim]")
        
        llm_predictions = execution_results.get("llm_predictions", [])
        llm_format_error = execution_results.get("tool_outputs", {}).get("llm_classify", {}).get("format_error", False)
        
        if llm_predictions and ground_truth and len(llm_predictions) == len(ground_truth):
            llm_acc = self._calculate_accuracy(llm_predictions, ground_truth, config)
            results["llm_acc_reward"] = llm_acc
            
            if llm_format_error:
                format_penalty = -0.5
                results["format_penalty"] += format_penalty
                emit_reward(format_penalty)
                console.print(f"[yellow]⚠ LLM classification format error penalty: {format_penalty:.4f}[/yellow]")
            
            emit_reward(llm_acc)
            console.print(f"[dim]LLM classification accuracy: {llm_acc:.4f}[/dim]")
        elif llm_format_error:
            format_penalty = -0.8
            results["format_penalty"] += format_penalty
            emit_reward(format_penalty)
            console.print(f"[red]❌ LLM classification format error (unable to calc acc) penalty: {format_penalty:.4f}[/red]")
        
        if self.use_reflection:
            nn_predictions = execution_results.get("nn_predictions", [])
            knn_predictions = execution_results.get("knn_predictions", [])
            final_predictions = execution_results.get("final_predictions", [])
            reflection_format_error = execution_results.get("tool_outputs", {}).get("reflection_decision", {}).get("format_error", False)
            
            if all([nn_predictions, llm_predictions, knn_predictions, final_predictions, ground_truth]):
                reflection_reward = self._calculate_reflection_reward(
                    nn_predictions=nn_predictions,
                    llm_predictions=llm_predictions,
                    knn_predictions=knn_predictions,
                    final_predictions=final_predictions,
                    ground_truth=ground_truth,
                    config=config
                )
                results["reflection_reward"] = reflection_reward
                
                if reflection_format_error:
                    format_penalty = -0.5
                    results["format_penalty"] += format_penalty
                    emit_reward(format_penalty)
                    console.print(f"[yellow]⚠ Reflection format error penalty: {format_penalty:.4f}[/yellow]")
                
                emit_reward(reflection_reward)
                console.print(f"[dim]Reflection correction reward: {reflection_reward:.4f}[/dim]")
            elif reflection_format_error:
                format_penalty = -0.7
                results["format_penalty"] += format_penalty
                emit_reward(format_penalty)
                console.print(f"[red]❌ Reflection format error (unable to calc reward) penalty: {format_penalty:.4f}[/red]")
        
        results["final_predictions"] = execution_results.get("final_predictions", [])
        results["execution_results"] = execution_results
        
        return results

    
    def _evaluate_trend_quality(
        self,
        trend_description: str,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        config: Any
    ) -> float:
        prompt = self._build_trend_evaluation_prompt(
            trend_description=trend_description,
            batch_data=batch_data,
            well_data=well_data,
            batch_start_idx=batch_start_idx,
            config=config
        )
        
        try:
            response = get_reward_response(prompt)
            
            console.print(f"[dim]GPT5 Evaluation: Type={type(response).__name__}, Length={len(response) if response else 0}[/dim]")
            
            self._save_llm_log("trend", prompt, response)
            
            score = self._parse_trend_evaluation(response)
            return score
            
        except Exception as e:
            console.print(f"[yellow]Trend evaluation failed: {e}[/yellow]")
            import traceback
            traceback.print_exc()
            return 0.5
    
    def _build_trend_evaluation_prompt(
        self,
        trend_description: str,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        config: Any
    ) -> str:
        depth_col = self._get_depth_column(config)
        feature_stats = []
        
        for col in config.FEATURE_COLUMNS:
            if col in batch_data.columns and col != depth_col:
                mean_val = batch_data[col].mean()
                std_val = batch_data[col].std()
                min_val = batch_data[col].min()
                max_val = batch_data[col].max()
                feature_stats.append(
                    f"- {col}: mean={mean_val:.2f}, std={std_val:.2f}, "
                    f"range=[{min_val:.2f}, {max_val:.2f}]"
                )
        
        prompt = f"""# Trend Description Quality Evaluation Task

## Task
Evaluate the quality of the following well log trend description based on multiple dimensions.

## Trend Description to Evaluate
{trend_description}

## Actual Data Statistics
{chr(10).join(feature_stats)}

## Evaluation Dimensions
Please score each dimension from 0 to 1:

1. **Accuracy** (0-1): Does the description accurately reflect the actual data trends?
2. **Completeness** (0-1): Does it cover key features and their variations?
3. **Clarity** (0-1): Is the description clear and easy to understand?
4. **Depth Information** (0-1): Does it include specific depth ranges and values?

## Output Format
Provide your evaluation in JSON format:

```json
{{
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "depth_info": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "Brief explanation"
}}
The overall_score should be the average of the four dimension scores. """ 
    return prompt
    def _parse_trend_evaluation(self, response: str) -> float:
        import re
        
        try:
            if not response or not response.strip():
                console.print("Regarding trend evaluation result, LLM output nothing!")
                return 0.5
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            overall_score = data.get("overall_score", 0.5)
            
            if overall_score == 0.5 and any(k in data for k in ["accuracy", "completeness", "clarity", "depth_info"]):
                scores = [
                    data.get("accuracy", 0.5),
                    data.get("completeness", 0.5),
                    data.get("clarity", 0.5),
                    data.get("depth_info", 0.5)
                ]
                overall_score = sum(scores) / len(scores)
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            console.print(f"[yellow]Trend evaluation parsing failed: {e}, {response}[/yellow]")
            return 0.5

    def _calculate_accuracy(self, predictions: List[str], ground_truth: List[int], config: Any) -> float:
        if not predictions or not ground_truth or len(predictions) != len(ground_truth):
            return 0.0
        
        ground_truth_labels = []
        for idx in ground_truth:
            if 0 <= idx < len(config.CLASS_NAMES):
                ground_truth_labels.append(config.CLASS_NAMES[idx])
            else:
                ground_truth_labels.append("Unknown")
        
        correct = sum(1 for pred, gt in zip(predictions, ground_truth_labels) if pred == gt)
        return correct / len(ground_truth)

    def _calculate_reflection_reward(
        self,
        nn_predictions: List[str],
        llm_predictions: List[str],
        knn_predictions: List[str],
        final_predictions: List[str],
        ground_truth: List[int],
        config: Any
    ) -> float:
        if len(final_predictions) != len(ground_truth):
            return 0.0
        
        ground_truth_labels = []
        for idx in ground_truth:
            if 0 <= idx < len(config.CLASS_NAMES):
                ground_truth_labels.append(config.CLASS_NAMES[idx])
            else:
                ground_truth_labels.append("Unknown")
        
        rewards = []
        
        for i in range(len(final_predictions)):
            nn_pred = nn_predictions[i]
            llm_pred = llm_predictions[i]
            knn_pred = knn_predictions[i]
            final_pred = final_predictions[i]
            gt = ground_truth_labels[i]
            
            three_preds = [nn_pred, llm_pred, knn_pred]
            has_correct = gt in three_preds
            
            if not has_correct:
                rewards.append(0.0)
                continue
            
            vote_counts = Counter(three_preds)
            most_common_pred, most_common_count = vote_counts.most_common(1)[0]
            
            if most_common_count >= 2:
                if final_pred == most_common_pred:
                    rewards.append(0.0)
                else:
                    if final_pred == gt:
                        rewards.append(1.0)
                    else:
                        rewards.append(-1.0)
            else:
                if final_pred == gt:
                    rewards.append(1.0)
                else:
                    rewards.append(-0.5)
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        normalized_reward = (avg_reward + 1.0) / 2.0
        
        return normalized_reward

    def _get_depth_column(self, config: Any) -> str:
        for col in config.FEATURE_COLUMNS:
            if col.upper() in ['DEPTH', 'DEPT']:
                return col
        return 'Depth'

    def classify_batch(
        self,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        batch_start_idx: int,
        llm_caller: Callable[[str], str],
        dataset_name: str
    ) -> Optional[List[str]]:
        if dataset_name not in self.planners:
            console.print(f"[red]Error: Unknown dataset {dataset_name}[/red]")
            return None
        
        planner = self.planners[dataset_name]
        executor = self.executors[dataset_name]
        reflector = self.reflectors[dataset_name]
        
        plan = planner.create_plan(
            batch_data=batch_data,
            well_data=well_data,
            llm_caller=llm_caller if self.use_llm_planning else None,
            use_llm_planning=self.use_llm_planning
        )
        
        execution_results = executor.execute(
            plan=plan,
            batch_data=batch_data,
            well_data=well_data,
            batch_start_idx=batch_start_idx,
            llm_caller=llm_caller,
            context_length=self.context_length,
            post_length=self.post_length
        )
        
        if self.use_reflection:
            final_predictions, _ = reflector.reflect(
                execution_results=execution_results,
                batch_data=batch_data,
                llm_caller=llm_caller
            )
            return final_predictions
        else:
            return execution_results.get("final_predictions")

    def load_wells_for_inference(
        self,
        dataset_splits: Dict[str, str] = None,
        model_name: str = None
    ):
        if dataset_splits is None:
            dataset_splits = {name: 'test' for name in self.dataset_names}
        
        total_wells = 0
        
        for dataset_name, split in dataset_splits.items():
            if dataset_name not in self.data_loaders:
                console.print(f"[yellow]Warning: Skipping unknown dataset {dataset_name}[/yellow]")
                continue
            
            config = self.configs[dataset_name]
            directory = config.get_model_path(split, model_name)
            
            console.print(f"[blue]Loading {dataset_name} dataset for inference ({split})...[/blue]")
            
            wells = self.data_loaders[dataset_name].load_from_directory(
                directory=directory,
                split=split
            )
            
            self.wells_by_dataset[dataset_name] = wells
            
            for well_id in wells.keys():
                if well_id in self.well_to_dataset:
                    console.print(
                        f"[yellow]Warning: Well ID '{well_id}' exists in multiple datasets, "
                        f"overwriting with data from {dataset_name} (was {self.well_to_dataset[well_id]})[/yellow]"
                    )
                self.well_to_dataset[well_id] = dataset_name
            
            total_wells += len(wells)
            console.print(f"[green]✓ {dataset_name}: Loaded {len(wells)} wells[/green]")
        
        console.print(f"[bold green]✓ Total loaded {total_wells} wells (from {len(dataset_splits)} datasets)[/bold green]")

    def get_well_dataset(self, well_id: str) -> Optional[str]:
        return self.well_to_dataset.get(well_id)

    def get_well_data(self, well_id: str, dataset_name: str = None) -> Optional[pd.DataFrame]:
        if dataset_name is None:
            dataset_name = self.well_to_dataset.get(well_id)
            if dataset_name is None:
                console.print(f"[red]Error: Dataset for well '{well_id}' not found[/red]")
                return None
        
        if dataset_name not in self.wells_by_dataset:
            console.print(f"[red]Error: Dataset '{dataset_name}' not loaded[/red]")
            return None
        
        return self.wells_by_dataset[dataset_name].get(well_id)

    def classify_well(
        self,
        well_id: str,
        llm_caller: Callable[[str], str],
        dataset_name: str = None,
        batch_size: int = None,
        stride: int = None
    ) -> pd.DataFrame:
        well_data = self.get_well_data(well_id, dataset_name)
        if well_data is None:
            return pd.DataFrame()
        
        if dataset_name is None:
            dataset_name = self.well_to_dataset.get(well_id)
        
        if dataset_name not in self.planners:
            console.print(f"[red]Error: Components for dataset '{dataset_name}' not initialized[/red]")
            return pd.DataFrame()
        
        batch_size = batch_size or self.batch_size
        stride = stride or self.stride
        
        well_data = well_data.copy()
        well_data['prediction'] = None
        
        n_samples = len(well_data)
        
        console.print(f"[blue]Starting classification for well '{well_id}' (Dataset: {dataset_name}, {n_samples} samples)[/blue]")
        
        for start_idx in range(0, n_samples, stride):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = well_data.iloc[start_idx:end_idx].copy()
            
            if len(batch_data) == 0:
                continue
            
            predictions = self.classify_batch(
                batch_data=batch_data,
                well_data=well_data,
                batch_start_idx=start_idx,
                llm_caller=llm_caller,
                dataset_name=dataset_name
            )
            
            if predictions:
                for i, pred in enumerate(predictions):
                    idx = start_idx + i
                    if idx < n_samples:
                        well_data.iloc[idx, well_data.columns.get_loc('prediction')] = pred
        
        console.print(f"[green]✓ Classification complete for well '{well_id}'[/green]")
        return well_data

    def list_loaded_wells(self) -> Dict[str, List[str]]:
        result = {}
        for dataset_name, wells in self.wells_by_dataset.items():
            if wells:
                result[dataset_name] = list(wells.keys())
        return result