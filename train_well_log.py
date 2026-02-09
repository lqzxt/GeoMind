from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, cast

from rich.console import Console
import agentlightning as agl

from agent import LitWellLogClassifier, WellLogTask
from data_loader import DataLoader

console = Console()


def get_verl_config(
    model_name: str,
    epochs: int = 1,
    exp_name: str = "",
    n_gpus_per_node: int = 8,
    nnodes: int = 1,
    accelerator_type: str = "npu",
    device: str = "npu",
) -> Dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "kl_ctrl": {
                "kl_coef": 0.0
            },
            "norm_adv_by_std_in_grpo": False,
        },
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 8192,
            "max_response_length": 4096,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "prompt_key": "prompt",
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "tensor_model_parallel_size": n_gpus_per_node,
                "gpu_memory_utilization": 0.75,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "enable_chunked_prefill": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "val_kwargs": {
                    "temperature": 1.0,
                    "top_p": 0.7,
                    "top_k": -1,
                    "do_sample": True,
                    "n": 1,
                }
            },
            "actor": {
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "loss_agg_mode": "token-mean",
                "optim": {
                    "lr": 5e-7,
                    "lr_warmup_steps": 10,
                    "weight_decay": 0.1,
                },
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 10.0,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": model_name,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "logger": ["console"],
            "n_gpus_per_node": n_gpus_per_node,
            "nnodes": nnodes,
            "accelerator_type": accelerator_type,
            "device": device,
            "val_before_train": False,
            "test_freq": 50,
            "save_freq": 10,
            "total_epochs": epochs,
            "resume_mode": "auto",
            "default_local_dir": f"agent_output/{exp_name}",
        }
    }


def prepare_dataset(
    agent: LitWellLogClassifier,
    train_val_data: Dict[str, Dict[str, Any]],
) -> List[WellLogTask]:
    console.print("[blue]Preparing training tasks...[/blue]")
    
    tasks = []
    
    for dataset_name in agent.dataset_names:
        dataset_config = agent.configs[dataset_name]
        wells = train_val_data.get(dataset_name, {})
        
        if not wells:
            console.print(f"[yellow]Warning: {dataset_name} has no data, skipping[/yellow]")
            continue
        
        dataset_tasks = 0
        
        label_to_idx = None
        if dataset_name == "FORCE-2020":
            label_to_idx = {label: idx for idx, label in enumerate(dataset_config.CLASS_NAMES)}
        elif dataset_name == "facies":
            label_to_idx = {i: i-1 for i in range(1, 10)}
        elif dataset_name == "geolink":
            actual_labels = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 16]
            label_to_idx = {label: idx for idx, label in enumerate(actual_labels)}
        elif dataset_name == "SEAM":
            actual_labels = [2, 3, 4, 5, 8, 9, 10]
            label_to_idx = {label: idx for idx, label in enumerate(actual_labels)}
        
        for well_id, well_data in wells.items():
            batch_idx = 0
            
            for start in range(0, len(well_data), agent.stride):
                end = min(start + agent.batch_size, len(well_data))
                
                if end - start < agent.batch_size // 2:
                    continue
                
                batch_data = well_data.iloc[start:end].copy()
                
                if dataset_config.TARGET_COLUMN not in batch_data.columns:
                    continue
                
                ground_truth = batch_data[dataset_config.TARGET_COLUMN].tolist()
                
                if label_to_idx is not None:
                    ground_truth = [label_to_idx.get(label, 0) for label in ground_truth]
                
                ground_truth = [int(label) for label in ground_truth]
                
                task: WellLogTask = {
                    "dataset_name": dataset_name,
                    "well_id": well_id,
                    "batch_idx": batch_idx,
                    "batch_data": {col: batch_data[col].tolist() for col in batch_data.columns},
                    "batch_start_idx": start,
                    "ground_truth": ground_truth
                }
                
                tasks.append(task)
                batch_idx += 1
                dataset_tasks += 1
        
        console.print(f"  {dataset_name}: {len(wells)} wells, {dataset_tasks} tasks")
    
    console.print(f"[bold green]✓ Total {len(tasks)} training tasks[/bold green]")
    
    return tasks


def split_dataset(
    tasks: List[WellLogTask],
    val_size: int = 200,
    random_seed: int = 42
) -> tuple[List[WellLogTask], List[WellLogTask]]:
    random.seed(random_seed)
    shuffled_tasks = tasks.copy()
    random.shuffle(shuffled_tasks)
    
    if len(shuffled_tasks) < val_size:
        val_size = max(1, len(shuffled_tasks) // 5)
    
    val_dataset = shuffled_tasks[:val_size]
    train_dataset = shuffled_tasks[val_size:]
    
    console.print(f"Dataset split: Train {len(train_dataset)}, Validation {len(val_dataset)}")
    
    return train_dataset, val_dataset


def train(
    model: str = "qwen3-8b",
    epochs: int = 1,
    runners: int = 64,
    exp_name: str = "",
    datasets: List[str] = None,
    model_name: str = "gpt4ts",
    n_gpus_per_node: int = 8,
    nnodes: int = 1,
    accelerator_type: str = "npu",
    device: str = "npu",
):
    console.print(f"[bold blue]🚀 Starting training (Model: {model}, Epochs: {epochs})...[/bold blue]")
    
    model_map = {
        "qwen3-4b": "path/to/Qwen3-4B",
    }
    if model not in model_map:
        raise ValueError(f"Unsupported model: {model}, options: {list(model_map.keys())}")
    
    model_path = model_map[model]
    
    if datasets is None:
        datasets = ['facies', 'FORCE-2020', 'geolink', 'SEAM']
    
    console.print(f"Using datasets: {', '.join(datasets)}")
    
    agent = LitWellLogClassifier(
        dataset_names=datasets,
        batch_size=16,
        stride=4,
        context_length=16,
        post_length=16,
        use_llm_planning=False,
        use_reflection=True,
    )
    
    console.print("[blue]Loading training data...[/blue]")
    train_val_data = agent.load_training_and_validation_data(model_name=model_name)
    
    console.print("[blue]Building retrieval index...[/blue]")
    agent.build_retrieval_index(train_val_data)
    
    all_tasks = prepare_dataset(agent, train_val_data)
    train_dataset, val_dataset = split_dataset(all_tasks, val_size=200)
    
    verl_config = get_verl_config(
        model_path,
        epochs=epochs,
        exp_name=exp_name,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        accelerator_type=accelerator_type,
        device=device,
    )
    
    algorithm = agl.VERL(verl_config)
    
    trainer = agl.Trainer(
        n_runners=runners,
        algorithm=algorithm,
    )
    
    console.print("[bold green]Starting training...[/bold green]")
    
    trainer.fit(
        agent,
        train_dataset=cast(agl.Dataset[WellLogTask], train_dataset),
        val_dataset=cast(agl.Dataset[WellLogTask], val_dataset)
    )
    
    console.print("[bold green]✓ Training complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Well Log Classification RL Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Run mode")
    
    train_parser = subparsers.add_parser("train", help="Training mode")
    train_parser.add_argument("--model", choices=["qwen3-8b"], default="qwen3-8b", help="Model selection")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    train_parser.add_argument("--runners", type=int, default=64, help="Number of runners")
    train_parser.add_argument("--exp_name", type=str, default="well_log_agent", help="Experiment name")
    train_parser.add_argument("--datasets", nargs="+", 
                            choices=['facies', 'FORCE-2020', 'geolink', 'SEAM'],
                            help="Datasets to use")
    train_parser.add_argument("--model_name", type=str, default="gpt4ts", help="Data model name")
    train_parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs/NPUs per node")
    train_parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    train_parser.add_argument("--accelerator", type=str, default="npu", 
                            choices=["gpu", "npu"], help="Accelerator type")
    train_parser.add_argument("--device", type=str, default="npu",
                            choices=["cuda", "npu"], help="Device type")
    
    train_parser.set_defaults(func=lambda args: train(
        args.model, args.epochs, args.runners, args.exp_name,
        args.datasets, args.model_name, args.gpus, args.nodes,
        args.accelerator, args.device
    ))
    
    args = parser.parse_args()
    
    args.func(args)


if __name__ == "__main__":
    main()