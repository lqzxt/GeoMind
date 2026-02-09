import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID

from constants import get_dataset_config
from data_loader import WellDataLoader
from LLM_caller import get_reward_response
from planner import Planner
from executor import ActionExecutor
from reflector import Reflector
from tools import LithologyTool

console = Console()


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger('test_runner')
    logger.setLevel(logging.DEBUG)
    
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger


class LLMCallLogger:
    
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
    
    def log_call(self, prompt: str, response: str, call_type: str = "general"):
        with self.lock:
            self.calls.append({
                "timestamp": datetime.now().isoformat(),
                "call_type": call_type,
                "prompt": prompt,
                "response": response
            })
    
    def get_all_calls(self) -> List[Dict[str, Any]]:
        with self.lock:
            return self.calls.copy()
    
    def clear(self):
        with self.lock:
            self.calls = []


class TestRunner:
    
    def __init__(
        self,
        dataset_name: str,
        exp_name: str = "default_exp",
        use_llm_planning: bool = True,
        batch_size: int = 10,
        stride: int = None,
        context_length: int = 16,
        post_length: int = 16,
        max_workers: int = 40,
        model_name: str = None
    ):
        self.dataset_name = dataset_name
        self.exp_name = exp_name
        self.use_llm_planning = use_llm_planning
        self.batch_size = batch_size
        self.stride = stride if stride is not None else batch_size
        self.context_length = context_length
        self.post_length = post_length
        self.max_workers = max_workers
        self.model_name = model_name
        
        self.config = get_dataset_config(dataset_name)
        
        self.data_loader = WellDataLoader(self.config)
        self.tool = LithologyTool(self.config)
        self.planner = Planner(self.config)
        self.reflector = Reflector(self.config)
        
        self.result_dir = Path(f"res/{exp_name}/{dataset_name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.result_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger(str(log_file))
        self.logger.info(f"Test Runner Initialized")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Experiment Name: {exp_name}")
        self.logger.info(f"Model Name: {model_name if model_name else 'None (Original Data)'}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Stride: {self.stride}")
        self.logger.info(f"Max Concurrent Threads: {self.max_workers}")
        self.logger.info(f"Log File: {log_file}")
        
        self.result_dir = Path(f"res/{exp_name}/{dataset_name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]✓ Test Runner Initialized[/green]")
        console.print(f"  Dataset: {dataset_name}")
        console.print(f"  Experiment Name: {exp_name}")
        console.print(f"  Model Name: {model_name if model_name else 'None (Original Data)'}")
        console.print(f"  Batch Size: {self.batch_size}")
        console.print(f"  Stride: {self.stride}")
        console.print(f"  Max Concurrent Threads: {self.max_workers}")
        if self.stride < self.batch_size:
            overlap = self.batch_size - self.stride
            console.print(f"  Overlap Length: {overlap} (Will produce overlapping predictions)")
        console.print(f"  Result Output Path: {self.result_dir}")
        console.print(f"  Log File: {log_file}")
    
    def build_index(self, split: str = "train"):
        console.print(f"\n[cyan]Building retrieval index...[/cyan]")
        train_wells = self.data_loader.load_from_directory(split=split)
        
        if not train_wells:
            console.print(f"[yellow]⚠ Training data not found, skipping index build[/yellow]")
            return
        
        all_train_data = pd.concat(train_wells.values(), ignore_index=True)
        
        self.tool.build_retrieval_index(all_train_data)
        console.print(f"[green]✓ Retrieval index built ({len(all_train_data)} samples)[/green]")
        
        if self.config.TARGET_COLUMN in all_train_data.columns:
            self.tool.build_transition_matrix(all_train_data)
            console.print(f"[green]✓ Transition probability matrix built[/green]")
    
    def process_batch(
        self,
        batch_idx: int,
        batch_start: int,
        batch_end: int,
        batch_data: pd.DataFrame,
        well_data: pd.DataFrame,
        global_llm_logger: LLMCallLogger,
        well_id: str = None,
        save_id: str = None
    ) -> Dict[str, Any]:
        self.logger.info(f"Start processing batch {batch_idx}: samples {batch_start}-{batch_end}")
        
        actual_batch_size = len(batch_data)
        
        batch_llm_logger = LLMCallLogger()
        
        def llm_caller_with_logging(prompt: str, call_type: str = "general") -> str:
            try:
                self.logger.debug(f"Batch {batch_idx} - LLM Call ({call_type})")
                response = get_reward_response(prompt)
                batch_llm_logger.log_call(prompt, response, call_type)
                global_llm_logger.log_call(prompt, response, call_type)
                self.logger.debug(f"Batch {batch_idx} - LLM Response Success")
                return response
            except Exception as e:
                self.logger.error(f"Batch {batch_idx} - LLM Call Failed: {e}")
                raise
        
        try:
            self.logger.debug(f"Batch {batch_idx} - Creating Plan")
            plan = self.planner.create_plan(
                batch_data=batch_data,
                well_data=well_data,
                use_llm_planning=self.use_llm_planning,
                llm_caller=lambda p: llm_caller_with_logging(p, "planning")
            )
            self.logger.info(f"Batch {batch_idx} - Executing Plan: {' → '.join(plan)}")
            
            self.logger.debug(f"Batch {batch_idx} - Creating Independent Tool Instance")
            tool = LithologyTool(self.config)
            tool.train_data = self.tool.train_data
            tool.train_features = self.tool.train_features
            tool._index_built = self.tool._index_built
            tool.transition_matrix = self.tool.transition_matrix
            tool._transition_built = self.tool._transition_built
            
            self.logger.debug(f"Batch {batch_idx} - Start Tool Execution")
            executor = ActionExecutor(tool)
            execution_results = executor.execute(
                plan=plan,
                batch_data=batch_data,
                well_data=well_data,
                batch_start_idx=batch_start,
                llm_caller=lambda p: llm_caller_with_logging(p, "execution"),
                context_length=self.context_length,
                post_length=self.post_length,
                silent=True
            )
            self.logger.debug(f"Batch {batch_idx} - Tool Execution Complete")
            
            reflector = Reflector(self.config)
            validation = reflector.validate_results(execution_results, batch_data)
            
            if not validation["is_valid"]:
                self.logger.warning(f"Batch {batch_idx} - Validation Failed: {validation['issues']}")
            
            final_predictions = execution_results.get("final_predictions", [])
            
            if final_predictions is None:
                self.logger.warning(f"Batch {batch_idx} - final_predictions is None, setting to empty list")
                final_predictions = []
            
            self.logger.info(f"Batch {batch_idx} - Complete, Prediction Count: {len(final_predictions)}")
            
            batch_result = {
                "batch_idx": batch_idx,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "actual_batch_size": actual_batch_size,
                "plan": plan,
                "execution_results": {
                    "nn_predictions": execution_results.get("nn_predictions"),
                    "llm_predictions": execution_results.get("llm_predictions"),
                    "knn_predictions": execution_results.get("knn_predictions"),
                    "final_predictions": final_predictions,
                    "disagreement_report": execution_results.get("disagreement_report"),
                    "transition_report": execution_results.get("transition_report"),
                    "reflection_reasoning": execution_results.get("reflection_reasoning"),
                    "trend_analysis": execution_results.get("trend_analysis"),
                    "similar_cases": execution_results.get("similar_cases"),
                    "transition_analysis": execution_results.get("transition_analysis")
                },
                "validation": validation,
                "position_mapping": {
                    "start_idx": batch_start,
                    "end_idx": batch_end,
                    "predictions_with_positions": [
                        {"global_idx": batch_start + i, "prediction": pred}
                        for i, pred in enumerate(final_predictions)
                    ] if final_predictions else []
                }
            }
            
            if well_id and save_id:
                batch_json_data = {
                    "metadata": {
                        "well_id": well_id,
                        "batch_idx": batch_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "actual_batch_size": actual_batch_size,
                        "timestamp": datetime.now().isoformat(),
                        "dataset_name": self.dataset_name,
                        "exp_name": self.exp_name
                    },
                    "batch_data": batch_data.to_dict(orient="records"),
                    "ground_truth": batch_data[self.config.TARGET_COLUMN].tolist() if self.config.TARGET_COLUMN in batch_data.columns else None,
                    "plan": plan,
                    "execution_results": {
                        "nn_predictions": execution_results.get("nn_predictions"),
                        "llm_predictions": execution_results.get("llm_predictions"),
                        "knn_predictions": execution_results.get("knn_predictions"),
                        "final_predictions": final_predictions,
                        "disagreement_report": execution_results.get("disagreement_report"),
                        "transition_report": execution_results.get("transition_report"),
                        "reflection_reasoning": execution_results.get("reflection_reasoning"),
                        "trend_analysis": execution_results.get("trend_analysis"),
                        "similar_cases": execution_results.get("similar_cases"),
                        "transition_analysis": execution_results.get("transition_analysis")
                    },
                    "validation": validation,
                    "llm_calls": batch_llm_logger.get_all_calls(),
                    "position_mapping": {
                        "start_idx": batch_start,
                        "end_idx": batch_end,
                        "predictions_with_positions": [
                            {"global_idx": batch_start + i, "prediction": pred}
                            for i, pred in enumerate(final_predictions)
                        ] if final_predictions else []
                    }
                }
                
                batch_json_path = self.result_dir / f"{save_id}_batch_{batch_idx:03d}.json"
                with open(batch_json_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_json_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Batch {batch_idx} - JSON Saved: {batch_json_path}")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Batch {batch_idx} - Processing Failed: {e}", exc_info=True)
            raise
    
    def process_well(
        self,
        well_id: str,
        well_data: pd.DataFrame,
        save_id: str
    ) -> Dict[str, Any]:
        console.print(f"\n[cyan]{'='*60}[/cyan]")
        console.print(f"[cyan]Processing Well: {well_id} ({len(well_data)} samples)[/cyan]")
        console.print(f"[cyan]{'='*60}[/cyan]")
        
        self.logger.info(f"="*60)
        self.logger.info(f"Start Processing Well: {well_id} ({len(well_data)} samples)")
        self.logger.info(f"="*60)
        
        llm_logger = LLMCallLogger()
        
        position_predictions = {i: [] for i in range(len(well_data))}
        position_lock = threading.Lock()
        
        batch_configs = []
        batch_start = 0
        batch_idx = 0
        
        while batch_start < len(well_data):
            batch_end = min(batch_start + self.batch_size, len(well_data))
            
            if batch_end - batch_start < self.batch_size and batch_start > 0:
                batch_start = max(0, len(well_data) - self.batch_size)
                batch_end = len(well_data)
            
            batch_configs.append({
                "batch_idx": batch_idx,
                "batch_start": batch_start,
                "batch_end": batch_end
            })
            
            batch_idx += 1
            batch_start += self.stride
        
        num_batches = len(batch_configs)
        
        console.print(f"[dim]Sliding Window Config: Size={self.batch_size}, Stride={self.stride}, Batch Count={num_batches}[/dim]")
        console.print(f"[dim]Using {self.max_workers} threads for concurrent processing[/dim]")
        
        self.logger.info(f"Sliding Window Config: Size={self.batch_size}, Stride={self.stride}, Batch Count={num_batches}")
        self.logger.info(f"Using {self.max_workers} threads for concurrent processing")
        self.logger.info(f"Batch Config: {batch_configs}")
        
        all_batch_results = [None] * num_batches
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing {well_id}", total=num_batches)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {}
                for config in batch_configs:
                    batch_idx = config["batch_idx"]
                    batch_start = config["batch_start"]
                    batch_end = config["batch_end"]
                    batch_data = well_data.iloc[batch_start:batch_end].copy()
                    
                    future = executor.submit(
                        self.process_batch,
                        batch_idx,
                        batch_start,
                        batch_end,
                        batch_data,
                        well_data,
                        llm_logger,
                        well_id,
                        save_id
                    )
                    future_to_batch[future] = batch_idx
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        self.logger.debug(f"Waiting for batch {batch_idx} completion")
                        batch_result = future.result()
                        all_batch_results[batch_idx] = batch_result
                        self.logger.info(f"Batch {batch_idx} completed successfully")
                        
                        final_predictions = batch_result["execution_results"]["final_predictions"]
                        batch_start = batch_result["batch_start"]
                        
                        if final_predictions:
                            with position_lock:
                                for i, pred in enumerate(final_predictions):
                                    global_idx = batch_start + i
                                    if global_idx < len(well_data):
                                        position_predictions[global_idx].append(pred)
                            self.logger.debug(f"Batch {batch_idx} - Updated predictions for {len(final_predictions)} positions")
                        else:
                            self.logger.warning(f"Batch {batch_idx} has no prediction results")
                        
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        self.logger.error(f"Batch {batch_idx} processing failed: {e}", exc_info=True)
                        console.print(f"[red]❌ Batch {batch_idx} processing failed: {e}[/red]")
                        all_batch_results[batch_idx] = {
                            "batch_idx": batch_idx,
                            "batch_start": batch_configs[batch_idx]["batch_start"],
                            "batch_end": batch_configs[batch_idx]["batch_end"],
                            "error": str(e),
                            "execution_results": {
                                "final_predictions": []
                            }
                        }
                        progress.update(task, advance=1)
        
        all_position_predictions = []
        for i in range(len(well_data)):
            preds = position_predictions[i]
            all_position_predictions.append({
                "position": i,
                "predictions": preds,
                "num_predictions": len(preds)
            })
        
        overlap_stats = {
            "positions_with_single_prediction": sum(1 for p in all_position_predictions if p["num_predictions"] == 1),
            "positions_with_multiple_predictions": sum(1 for p in all_position_predictions if p["num_predictions"] > 1),
            "max_predictions_per_position": max((p["num_predictions"] for p in all_position_predictions), default=0),
            "total_positions": len(all_position_predictions),
            "positions_with_no_prediction": sum(1 for p in all_position_predictions if p["num_predictions"] == 0)
        }
        
        console.print(f"\n[cyan]Overlap Prediction Statistics:[/cyan]")
        console.print(f"  Positions with no prediction: {overlap_stats['positions_with_no_prediction']}")
        console.print(f"  Positions with single prediction: {overlap_stats['positions_with_single_prediction']}")
        console.print(f"  Positions with multiple predictions: {overlap_stats['positions_with_multiple_predictions']}")
        console.print(f"  Max predictions per position: {overlap_stats['max_predictions_per_position']}")
        
        console.print(f"\n[green]{'='*60}[/green]")
        console.print(f"[green]Well {well_id} processing complete[/green]")
        console.print(f"[green]{'='*60}[/green]")
        
        result_data = {
            "metadata": {
                "well_id": well_id,
                "dataset_name": self.dataset_name,
                "exp_name": self.exp_name,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(well_data),
                "num_batches": num_batches,
                "batch_size": self.batch_size,
                "stride": self.stride,
                "overlap_length": self.batch_size - self.stride,
                "use_llm_planning": self.use_llm_planning,
                "max_workers": self.max_workers,
                "overlap_stats": overlap_stats
            },
            "original_data": well_data.to_dict(orient="records"),
            "ground_truth": well_data[self.config.TARGET_COLUMN].tolist() if self.config.TARGET_COLUMN in well_data.columns else None,
            "position_predictions": all_position_predictions,
            "batch_results": all_batch_results,
            "llm_calls": llm_logger.get_all_calls()
        }
        
        save_path = self.result_dir / f"{save_id}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Result saved to: {save_path}[/green]")
        console.print(f"[dim]  - LLM Calls: {len(result_data['llm_calls'])}[/dim]")
        console.print(f"[dim]  - Total Positions: {len(all_position_predictions)}[/dim]")
        console.print(f"[dim]  - Overlapping Positions: {overlap_stats['positions_with_multiple_predictions']}[/dim]")
        
        return result_data
    
    def run_test(self, split: str = "test", max_wells: int = None, custom_test_dir: str = None):
        console.print(f"\n[bold cyan]Start Test: {self.dataset_name} - {split}[/bold cyan]")
        
        self.build_index(split="train")
        
        if custom_test_dir:
            console.print(f"[yellow]Using custom test directory: {custom_test_dir}[/yellow]")
            wells = self.data_loader.load_from_directory(directory=custom_test_dir)
        elif self.model_name:
            test_dir = self.config.get_model_path(split, self.model_name)
            console.print(f"[yellow]Using model test data: {test_dir}[/yellow]")
            wells = self.data_loader.load_from_directory(directory=test_dir)
        else:
            wells = self.data_loader.load_from_directory(split=split)
        
        if not wells:
            console.print(f"[red]Error: No well data loaded[/red]")
            return
        
        well_ids = list(wells.keys())
        if max_wells:
            well_ids = well_ids[:max_wells]
            console.print(f"[yellow]Limiting to first {max_wells} wells[/yellow]")
        
        results = []
        for idx, well_id in enumerate(well_ids, 1):
            well_data = wells[well_id]
            save_id = f"{split}_{well_id}"
            
            try:
                result = self.process_well(well_id, well_data, save_id)
                results.append(result)
            except Exception as e:
                console.print(f"[red]❌ Error processing well {well_id}: {e}[/red]")
                import traceback
                traceback.print_exc()
        
        self._generate_summary_report(results, split)
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], split: str):
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Test Summary Report[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        
        summary = {
            "exp_name": self.exp_name,
            "dataset_name": self.dataset_name,
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "num_wells": len(results),
            "total_samples": sum(r["metadata"]["num_samples"] for r in results),
            "total_llm_calls": sum(len(r["llm_calls"]) for r in results),
            "wells": []
        }
        
        for result in results:
            well_summary = {
                "well_id": result["metadata"]["well_id"],
                "num_samples": result["metadata"]["num_samples"],
                "num_llm_calls": len(result["llm_calls"]),
                "batch_size": result["metadata"]["batch_size"],
                "stride": result["metadata"]["stride"],
                "overlap_length": result["metadata"]["overlap_length"],
                "overlap_stats": result["metadata"]["overlap_stats"]
            }
            summary["wells"].append(well_summary)
            
            console.print(f"\nWell {well_summary['well_id']}:")
            console.print(f"  Samples: {well_summary['num_samples']}")
            console.print(f"  LLM Calls: {well_summary['num_llm_calls']}")
            console.print(f"  Window Size: {well_summary['batch_size']}, Stride: {well_summary['stride']}")
            console.print(f"  Overlapping Positions: {well_summary['overlap_stats']['positions_with_multiple_predictions']}")
        
        console.print(f"\n[bold]Total:[/bold]")
        console.print(f"  Processed Wells: {summary['num_wells']}")
        console.print(f"  Total Samples: {summary['total_samples']}")
        console.print(f"  Total LLM Calls: {summary['total_llm_calls']}")
        
        summary_path = self.result_dir / f"summary_{split}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ Summary report saved to: {summary_path}[/green]")


def main():
    DATASET_NAME = "facies"
    EXP_NAME = "test_run_v3"
    SPLIT = "test"
    MAX_WELLS = 1
    use_llm_planning = True
    BATCH_SIZE = 16
    STRIDE = 16
    MAX_WORKERS = 20
    
    MODEL_NAME = "gpt4ts"
    
    CUSTOM_TEST_DIR = None
    
    runner = TestRunner(
        dataset_name=DATASET_NAME,
        exp_name=EXP_NAME,
        use_llm_planning=USE_LLM_PLANNING,
        batch_size=BATCH_SIZE,
        stride=STRIDE,
        max_workers=MAX_WORKERS,
        model_name=MODEL_NAME
    )
    
    runner.run_test(split=SPLIT, max_wells=MAX_WELLS, custom_test_dir=CUSTOM_TEST_DIR)


if __name__ == "__main__":
    main()