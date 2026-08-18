"""Safe single-GPU stage execution used by Phase-0 rehearsals."""

from __future__ import annotations

import copy
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from dfc_sam.config import load_config, validate_experiment_config
from dfc_sam.data.collate import collate_pannuke
from dfc_sam.data.pannuke_dataset import (
    PanNukeDataset,
    load_supervision_manifest,
    load_teacher_partition,
)
from dfc_sam.data.transforms import build_model_input_transform
from dfc_sam.evaluation.official_metrics import (
    OfficialMetricAccumulator,
    load_official_primitives,
)
from dfc_sam.evaluation.validation import validate_student
from dfc_sam.models.factory import build_dfc_sam
from dfc_sam.models.teacher_adapter import BoxPromptedTeacher
from dfc_sam.models.teacher_sam import configure_teacher, load_sam_vit_b
from dfc_sam.pseudo.store import PseudoMaskStore
from dfc_sam.utils.hashing import atomic_write_json, sha256_file
from dfc_sam.utils.reproducibility import seed_everything

from .checkpoint import (
    BestCheckpointSelector,
    atomic_torch_save,
    checkpoint_payload,
    load_training_checkpoint,
    restore_training_checkpoint,
)
from .distributed import (
    DistributedContext,
    initialize_distributed,
    local_gradient_accumulation,
    synchronize_module_buffers,
)
from .optimization import build_stage_optimization
from .runtime import read_git_state, write_frozen_test_decision, write_run_provenance
from .sampling import DistributedSequentialSampler
from .stages import Stage, apply_stage_policy
from .trainer import (
    build_epoch_loader,
    build_grad_scaler,
    reduce_epoch_summary,
    train_student_epoch,
    train_teacher_epoch,
)


def _required_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_split(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_split_contract(config: dict[str, Any], split_payload: dict[str, Any]) -> None:
    expected_split = int(config["experiment"]["split_id"])
    if int(split_payload["split_id"]) != expected_split:
        raise ValueError(f"Config/split manifest mismatch: {expected_split} != {split_payload['split_id']}")
    expected_protocol = str(config["experiment"]["protocol"])
    actual_protocol = str(split_payload.get("protocol", "pannuke_from_generic"))
    if actual_protocol != expected_protocol:
        raise ValueError(f"Config/split protocol mismatch: {expected_protocol} != {actual_protocol}")
    if expected_protocol == "pannuke_standard_3fold":
        if int(split_payload["seed"]) != int(config["experiment"]["seed"]):
            raise ValueError("Config/split seed mismatch for pannuke_standard_3fold")
        counts = [float(value) for value in split_payload["class_instance_counts"]["train"]]
        inverse_roots = [1.0 / math.sqrt(value) for value in counts]
        mean_weight = sum(inverse_roots) / len(inverse_roots)
        expected_weights = [value / mean_weight for value in inverse_roots]
        configured_weights = [float(value) for value in config["loss"]["ugca_class_weights"]]
        if any(
            abs(actual - expected) > 1.0e-5
            for actual, expected in zip(configured_weights, expected_weights, strict=True)
        ):
            raise ValueError("UGCA class weights do not match train-only standard split counts")


def _early_stop_from_history(
    history: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    validation_rows = [row for row in history if row.get("validation") is not None]
    if not bool(settings.get("enabled", False)) or not validation_rows:
        return False, {"enabled": bool(settings.get("enabled", False))}
    min_delta = float(settings.get("min_delta", 0.0))
    best = float("-inf")
    best_index = -1
    for index, row in enumerate(validation_rows):
        metric = float(row["validation"]["mpq"])
        if metric > best + min_delta:
            best = metric
            best_index = index
    bad_validations = len(validation_rows) - best_index - 1
    epochs_completed = int(history[-1]["epoch"])
    report = {
        "enabled": True,
        "min_epochs": int(settings["min_epochs"]),
        "patience": int(settings["patience"]),
        "min_delta": min_delta,
        "meaningful_best_mpq": best,
        "validations_without_meaningful_improvement": bad_validations,
    }
    should_stop = epochs_completed >= int(settings["min_epochs"]) and bad_validations >= int(settings["patience"])
    return should_stop, report


def _load_warmup_initialization(model: torch.nn.Module, checkpoint: str | Path) -> None:
    payload = load_training_checkpoint(checkpoint)
    checkpoint_variant = str(payload.get("config", {}).get("ugca", {}).get("variant", "v1"))
    model_variant = str(getattr(model.ugca, "variant", "v1"))
    if checkpoint_variant == model_variant:
        model.load_state_dict(payload["model"], strict=True)
        return
    base_state = {name: value for name, value in payload["model"].items() if not name.startswith("ugca.")}
    incompatible = model.load_state_dict(base_state, strict=False)
    if incompatible.unexpected_keys or any(not name.startswith("ugca.") for name in incompatible.missing_keys):
        raise RuntimeError(
            "Warmup base-only initialization mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _joint_initialization(config: dict[str, Any]) -> str | Path:
    """Resolve the immutable parent used to initialize a Joint experiment."""
    if str(config["train"].get("joint_training_mode", "full")) in {
        "ugca_ranking",
        "controlled",
        "controlled_adaptive",
        "detector_head_adaptive",
    }:
        return config["weights"]["joint_initialization"]
    return config["weights"]["warmup_stage2"]


def _joint_initialization_hashes(config: dict[str, Any], stage: Stage) -> dict[str, str]:
    if stage is Stage.JOINT and config["train"]["flow"] == "warmup_joint":
        return {"joint_initialization": sha256_file(_joint_initialization(config))}
    return {}


def _model_initialization_hashes(config: dict[str, Any], stage: Stage) -> dict[str, str]:
    """Bind formal provenance to the exact detector and generic SAM bytes."""
    sam_variant = str(config["sam"]["variant"])
    sam_key = {"vit_b": "sam_vit_b", "vit_h": "sam_vit_h"}.get(sam_variant)
    if sam_key is None:
        raise ValueError(f"Unsupported SAM variant: {sam_variant}")
    hashes = {sam_key: sha256_file(config["weights"][sam_key])}
    if stage is not Stage.TEACHER:
        hashes["detector_stage1"] = sha256_file(config["weights"]["detector_stage1"])
    return hashes


def _atomic_alias(target: Path, alias: Path) -> None:
    """Atomically replace a relative symlink without duplicating a large checkpoint."""
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target.name)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_hardlink(target: Path, alias: Path) -> None:
    """Atomically point an independent checkpoint name at the same immutable inode."""
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.link")
    try:
        temporary.unlink(missing_ok=True)
        os.link(target, temporary)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_payload_is_selected_best(
    payload: dict[str, Any],
    *,
    initialize_parent_best: bool,
) -> bool:
    selector = payload.get("best_selector") or {}
    validation = payload.get("validation_metrics") or {}
    checkpoint_epoch = int(payload["epoch"])
    expected_selector_epoch = checkpoint_epoch if initialize_parent_best else checkpoint_epoch - 1
    return (
        int(selector.get("epoch", -1)) == expected_selector_epoch
        and abs(float(selector.get("mpq", float("nan"))) - float(validation.get("mpq", float("nan"))))
        <= 1.0e-12
        and abs(float(selector.get("bpq", float("nan"))) - float(validation.get("bpq", float("nan"))))
        <= 1.0e-12
    )


def _recover_history_row(
    payload: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    is_best: bool,
) -> dict[str, Any] | None:
    """Recover an epoch committed to last.pt before history.json was updated."""
    checkpoint_epoch = int(payload["epoch"])
    last_history_epoch = int(history[-1]["epoch"]) if history else 0
    if last_history_epoch == checkpoint_epoch:
        return None
    if last_history_epoch > checkpoint_epoch or checkpoint_epoch - last_history_epoch != 1:
        raise RuntimeError(
            f"Resume checkpoint/history gap is not recoverable: checkpoint={checkpoint_epoch}, "
            f"history={last_history_epoch}"
        )
    validation = payload.get("validation_metrics")
    if validation is None or int(payload.get("micro_step_in_epoch", 0)) != 0:
        raise RuntimeError("Resume checkpoint ahead of history is not a completed validation epoch")
    return {
        "epoch": checkpoint_epoch,
        "train": {"completed": True, "recovered_from_checkpoint": True},
        "validation": validation,
        "is_best": is_best,
        "epoch_elapsed_seconds": 0.0,
        "recovered_from_checkpoint": True,
    }


def _checkpoint_for_epoch(
    *,
    epoch: int,
    global_step: int,
    checkpoint_model: torch.nn.Module,
    optimization: Any,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    git_commit: str,
    split_payload: dict[str, Any],
    split_manifest: Path,
    pseudo_sha: str | None,
    sampler_state: dict[str, int],
    best_selector: BestCheckpointSelector | None,
    formal: bool,
    teacher: BoxPromptedTeacher | None,
    validation_metrics: dict[str, Any] | None = None,
    micro_step_in_epoch: int = 0,
    distributed_world_size: int = 1,
) -> dict[str, Any]:
    payload = checkpoint_payload(
        epoch=epoch,
        global_step=global_step,
        model=checkpoint_model,
        optimizer=optimization.optimizer,
        scheduler=optimization.scheduler,
        scaler=scaler,
        config=config,
        git_commit=git_commit,
        dataset_fingerprint=str(split_payload["dataset_fingerprint"]),
        split_manifest_sha256=sha256_file(split_manifest),
        pseudo_bank_sha256=pseudo_sha,
        sampler_state=sampler_state,
        best_selector_state=None if best_selector is None else best_selector.state_dict(),
        micro_step_in_epoch=micro_step_in_epoch,
    )
    payload["formal_experiment"] = formal
    payload["distributed_world_size"] = int(distributed_world_size)
    if teacher is not None:
        payload["teacher_mask_decoder"] = teacher.sam.mask_decoder.state_dict()
    if validation_metrics is not None:
        payload["validation_metrics"] = validation_metrics
    return payload


def _gather_validation_metrics(
    local_metrics: dict[str, Any],
    accumulator: OfficialMetricAccumulator,
    distributed: DistributedContext,
) -> dict[str, Any] | None:
    """Gather disjoint validation-image records and aggregate once on rank zero."""
    if not distributed.enabled:
        return local_metrics
    gathered: list[dict[str, Any] | None] = [None] * distributed.world_size
    dist.all_gather_object(
        gathered,
        {
            "per_image": local_metrics["per_image"],
            "instance_metrics": accumulator.instance_metrics.state_dict(),
        },
    )
    if not distributed.is_primary:
        return None
    per_image = [
        item
        for shard in gathered
        if shard is not None
        for item in shard["per_image"]
    ]
    per_image.sort(key=lambda item: str(item["image_id"]))
    if len({str(item["image_id"]) for item in per_image}) != len(per_image):
        raise RuntimeError("Distributed validation produced duplicate image IDs")
    accumulator.per_image = per_image
    accumulator.instance_metrics = type(accumulator.instance_metrics)(
        get_fast_pq=accumulator.get_fast_pq,
        match_iou=accumulator.match_iou,
    )
    for shard in gathered:
        if shard is not None:
            accumulator.instance_metrics.merge_state_dict(shard["instance_metrics"])
    return accumulator.compute()


def _prepare_dataset(
    config: dict[str, Any],
    all_samples: Path,
    split_manifest: Path,
    stage: Stage,
    max_samples: int | None,
) -> tuple[Subset, PseudoMaskStore | None]:
    supervision = None
    if config["supervision"]["mode"] == "mixed":
        supervision = load_supervision_manifest(
            config["supervision"]["manifest"],
            expected_split_id=int(config["experiment"]["split_id"]),
            expected_mask_ratio=float(config["supervision"]["mask_ratio"]),
        )
    dataset = PanNukeDataset(
        all_samples,
        split_manifest,
        role="train",
        supervision_by_image=supervision,
        train_color_augmentation=config.get("augmentation", {}).get("train_color"),
        augmentation_seed=int(config["experiment"]["seed"]),
        transform=build_model_input_transform(config),
    )
    indices = list(range(len(dataset)))
    if stage is Stage.TEACHER:
        partitions = load_teacher_partition(config["supervision"]["manifest"])
        indices = [
            index for index, record in enumerate(dataset.records) if str(record["image_id"]) in partitions["adaptation"]
        ]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        indices = indices[:max_samples]
    if not indices:
        raise RuntimeError("Selected training partition is empty")
    pseudo_store = None
    if stage in {Stage.WARMUP, Stage.JOINT} and config["supervision"]["use_pseudo_bank"]:
        pseudo_store = PseudoMaskStore(config["weights"]["pseudo_bank"])
    return Subset(dataset, indices), pseudo_store


def run_phase0_rehearsal(
    *,
    config_path: str | Path,
    stage: Stage,
    all_samples: str | Path,
    split_manifest: str | Path,
    output_root: str | Path,
    device_name: str,
    epochs: int = 1,
    max_samples: int | None = 16,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Exercise a real epoch/checkpoint path while marking every artifact non-formal."""
    if stage is Stage.DETECTOR:
        raise ValueError("Detector Stage I uses the already completed Ultralytics workflow")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    config_path = _required_file(config_path, "config")
    all_samples = _required_file(all_samples, "all-samples manifest")
    split_manifest = _required_file(split_manifest, "split manifest")
    config = load_config(config_path)
    validate_experiment_config(config)
    if stage is Stage.TEACHER and config["supervision"]["mode"] != "mixed":
        raise ValueError("Teacher rehearsal requires mixed supervision")
    if stage is Stage.WARMUP and config["train"]["flow"] != "warmup_joint":
        raise ValueError("Warm-up rehearsal requires train.flow=warmup_joint")
    if stage is Stage.JOINT and config["train"]["flow"] == "warmup_joint":
        _required_file(_joint_initialization(config), "Joint initialization checkpoint")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic=bool(config["train"]["deterministic"]))
    git_state = read_git_state(Path(__file__).parents[2])
    dataset, pseudo_store = _prepare_dataset(
        config,
        all_samples,
        split_manifest,
        stage,
        max_samples,
    )
    batch_size = int(config["train"]["batch_size_per_gpu"])
    steps_per_epoch = math.ceil(len(dataset) / batch_size)

    teacher: BoxPromptedTeacher | None = None
    if stage is Stage.TEACHER:
        teacher_sam = configure_teacher(load_sam_vit_b(config["weights"]["sam_vit_b"]))
        teacher = BoxPromptedTeacher(teacher_sam).to(device)
        apply_stage_policy(
            stage,
            detector=torch.nn.Identity(),
            bridge=torch.nn.Identity(),
            student_sam=teacher_sam,
            ugca=torch.nn.Identity(),
            teacher_sam=teacher_sam,
        )
        checkpoint_model: torch.nn.Module = teacher_sam
        optimization = build_stage_optimization(
            teacher,
            stage,
            config["train"],
            steps_per_epoch=steps_per_epoch,
            teacher_sam=teacher_sam,
        )
    else:
        model = build_dfc_sam(config).to(device)
        if stage is Stage.JOINT and config["train"]["flow"] == "warmup_joint":
            _load_warmup_initialization(model, _joint_initialization(config))
        apply_stage_policy(
            stage,
            detector=model.detector,
            bridge=model.feature_bridge,
            student_sam=model.student_sam,
            ugca=model.ugca,
            train_student_decoder_in_joint=bool(config["train"].get("train_student_decoder_in_joint", False)),
            joint_detector_scope=str(config["train"].get("joint_detector_scope", "all")),
            joint_training_mode=str(config["train"].get("joint_training_mode", "full")),
        )
        checkpoint_model = model
        optimization = build_stage_optimization(
            model,
            stage,
            config["train"],
            steps_per_epoch=steps_per_epoch,
        )
    scaler = build_grad_scaler(config["train"], device)

    split_payload = _load_split(split_manifest)
    _validate_split_contract(config, split_payload)
    pseudo_sha = None if pseudo_store is None else str(pseudo_store.metadata["bank_fingerprint"])
    start_epoch = global_step = 0
    sampler_state = {"seed": seed, "epoch": 0, "start_index": 0}
    if resume is not None:
        restored = restore_training_checkpoint(
            load_training_checkpoint(resume),
            model=checkpoint_model,
            optimizer=optimization.optimizer,
            scheduler=optimization.scheduler,
            scaler=scaler,
            dataset_fingerprint=str(split_payload["dataset_fingerprint"]),
            split_manifest_sha256=sha256_file(split_manifest),
            pseudo_bank_sha256=pseudo_sha,
            current_config=config,
            git_commit=git_state.commit,
        )
        start_epoch = restored["epoch"]
        global_step = restored["global_step"]
        sampler_state = load_training_checkpoint(resume)["sampler_state"]

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()) and resume is None:
        raise FileExistsError(f"Rehearsal output directory is not empty: {output_root}")
    write_run_provenance(
        output_root / "provenance.json",
        config=config,
        config_path=config_path,
        git_state=git_state,
        input_hashes={
            "all_samples": sha256_file(all_samples),
            "split_manifest": sha256_file(split_manifest),
            **_joint_initialization_hashes(config, stage),
        },
    )
    atomic_write_json(
        output_root / "rehearsal_meta.json",
        {
            "formal_experiment": False,
            "stage": stage.value,
            "max_samples": max_samples,
            "requested_epochs": epochs,
            "device": str(device),
            "parameter_groups": list(optimization.parameter_groups),
        },
    )

    summaries = []
    for epoch in range(start_epoch, start_epoch + epochs):
        loader, sampler = build_epoch_loader(
            dataset,
            seed=seed,
            epoch=epoch,
            start_index=0,
            batch_size=batch_size,
            num_workers=int(config["runtime"]["num_workers"]),
            pin_memory=bool(config["runtime"]["pin_memory"]),
            persistent_workers=bool(config["runtime"]["persistent_workers"]),
        )
        if teacher is not None:
            summary = train_teacher_epoch(
                teacher,
                loader,
                epoch=epoch,
                optimizer=optimization.optimizer,
                scheduler=optimization.scheduler,
                scaler=scaler,
                config=config,
                device=device,
            )
        else:
            summary = train_student_epoch(
                model,
                loader,
                epoch=epoch,
                stage=stage,
                strategy=str(config["supervision"]["strategy"]),
                pseudo_store=pseudo_store,
                optimizer=optimization.optimizer,
                scheduler=optimization.scheduler,
                scaler=scaler,
                config=config,
                device=device,
            )
        global_step += summary.optimizer_updates
        summaries.append(summary.to_dict())
        sampler_state = sampler.state_dict()
        sampler_state.update({"epoch": epoch + 1, "start_index": 0})
        payload = checkpoint_payload(
            epoch=epoch + 1,
            global_step=global_step,
            model=checkpoint_model,
            optimizer=optimization.optimizer,
            scheduler=optimization.scheduler,
            scaler=scaler,
            config=config,
            git_commit=git_state.commit,
            dataset_fingerprint=str(split_payload["dataset_fingerprint"]),
            split_manifest_sha256=sha256_file(split_manifest),
            pseudo_bank_sha256=pseudo_sha,
            sampler_state=sampler_state,
        )
        payload["formal_experiment"] = False
        payload["rehearsal_only"] = True
        if teacher is not None:
            payload["teacher_mask_decoder"] = teacher.sam.mask_decoder.state_dict()
        atomic_torch_save(payload, output_root / "last.pt")
        atomic_write_json(output_root / "epoch_summaries.json", summaries)
    result = {
        "status": "passed",
        "formal_experiment": False,
        "stage": stage.value,
        "epochs_completed": epochs,
        "start_epoch": start_epoch,
        "end_epoch": start_epoch + epochs,
        "global_step": global_step,
        "last_checkpoint": str((output_root / "last.pt").resolve()),
        "summaries": summaries,
    }
    atomic_write_json(output_root / "result.json", result)
    return result


def run_formal_stage(
    *,
    config_path: str | Path,
    stage: Stage,
    all_samples: str | Path,
    split_manifest: str | Path,
    output_root: str | Path,
    device_name: str,
    official_metrics_repository: str | Path | None = None,
    resume: str | Path | None = None,
    resume_source_git_commits: tuple[str, ...] = (),
    allow_world_size_transition: bool = False,
    reset_early_stopping_on_resume: bool = False,
) -> dict[str, Any]:
    """Run a frozen Teacher, warm-up, or joint stage on one or more rank-local GPUs."""
    if stage is Stage.DETECTOR:
        raise ValueError("Detector Stage I uses the already completed Ultralytics workflow")
    config_path = _required_file(config_path, "config")
    all_samples = _required_file(all_samples, "all-samples manifest")
    split_manifest = _required_file(split_manifest, "split manifest")
    config = load_config(config_path)
    validate_experiment_config(config)
    if config["experiment"].get("status") != "formal_frozen":
        raise RuntimeError("Formal execution requires experiment.status=formal_frozen")
    if stage is Stage.TEACHER and config["supervision"]["mode"] != "mixed":
        raise ValueError("Teacher stage requires mixed supervision")
    if stage is Stage.WARMUP and config["train"]["flow"] != "warmup_joint":
        raise ValueError("Warm-up stage requires train.flow=warmup_joint")
    if stage is Stage.JOINT and config["train"]["flow"] == "warmup_joint":
        _required_file(_joint_initialization(config), "Joint initialization checkpoint")

    project_root = Path(__file__).parents[2]
    git_state = read_git_state(project_root)
    if git_state.dirty:
        raise RuntimeError("Formal execution requires a clean committed Git worktree")
    if resume_source_git_commits and resume is None:
        raise ValueError("Resume source Git commits require a resume checkpoint")
    if allow_world_size_transition and resume is None:
        raise ValueError("World-size transition requires a resume checkpoint")
    if reset_early_stopping_on_resume and resume is None:
        raise ValueError("Early-stopping reset requires a resume checkpoint")
    distributed = initialize_distributed(device_name)
    device = distributed.device
    if stage is Stage.WARMUP:
        expected_world_size = int(config["train"].get("warmup_world_size_per_split", 1))
    elif stage is Stage.JOINT:
        expected_world_size = int(config["train"].get("joint_world_size_per_split", 1))
    else:
        expected_world_size = 1
    if distributed.world_size != expected_world_size:
        raise RuntimeError(
            f"{stage.value} requires world_size={expected_world_size}, but launcher provided {distributed.world_size}"
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal DFC-SAM stages require CUDA devices")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic=bool(config["train"]["deterministic"]))

    get_fast_pq = binarize = remap_label = None
    if stage in {Stage.WARMUP, Stage.JOINT}:
        configured_repository = config["evaluation"].get("official_metrics_repo")
        if official_metrics_repository is None:
            official_metrics_repository = configured_repository
        elif configured_repository and (
            Path(official_metrics_repository).expanduser().resolve()
            != Path(configured_repository).expanduser().resolve()
        ):
            raise RuntimeError("--metrics-repo differs from the frozen config")
        if official_metrics_repository is None:
            raise RuntimeError("Student validation requires an official metrics repository")
        get_fast_pq, binarize, remap_label = load_official_primitives(
            official_metrics_repository,
            expected_commit=str(config["evaluation"]["official_metrics_commit"]),
        )

    output_root = Path(output_root).expanduser().resolve()
    if distributed.is_primary:
        output_root.mkdir(parents=True, exist_ok=True)
        if any(output_root.iterdir()) and resume is None:
            raise FileExistsError(f"Formal output directory is not empty: {output_root}")
    distributed.barrier()
    dataset, pseudo_store = _prepare_dataset(
        config,
        all_samples,
        split_manifest,
        stage,
        max_samples=None,
    )
    batch_size = int(config["train"]["batch_size_per_gpu"])
    samples_per_rank = math.ceil(len(dataset) / distributed.world_size)
    steps_per_epoch = math.ceil(samples_per_rank / batch_size)
    runtime_config = copy.deepcopy(config)
    global_accumulation = int(config["train"]["gradient_accumulation"])
    runtime_config["train"]["gradient_accumulation"] = local_gradient_accumulation(
        global_accumulation,
        distributed.world_size,
    )

    teacher: BoxPromptedTeacher | None = None
    model: torch.nn.Module | None = None
    if stage is Stage.TEACHER:
        teacher_sam = configure_teacher(load_sam_vit_b(config["weights"]["sam_vit_b"]))
        teacher = BoxPromptedTeacher(teacher_sam).to(device)
        apply_stage_policy(
            stage,
            detector=torch.nn.Identity(),
            bridge=torch.nn.Identity(),
            student_sam=teacher_sam,
            ugca=torch.nn.Identity(),
            teacher_sam=teacher_sam,
        )
        checkpoint_model: torch.nn.Module = teacher_sam
        optimization = build_stage_optimization(
            teacher,
            stage,
            runtime_config["train"],
            steps_per_epoch=steps_per_epoch,
            teacher_sam=teacher_sam,
        )
        total_epochs = int(config["train"]["teacher_epochs"])
    else:
        model = build_dfc_sam(config).to(device)
        if stage is Stage.JOINT and config["train"]["flow"] == "warmup_joint":
            _load_warmup_initialization(model, _joint_initialization(config))
        apply_stage_policy(
            stage,
            detector=model.detector,
            bridge=model.feature_bridge,
            student_sam=model.student_sam,
            ugca=model.ugca,
            train_student_decoder_in_joint=bool(config["train"].get("train_student_decoder_in_joint", False)),
            joint_detector_scope=str(config["train"].get("joint_detector_scope", "all")),
            joint_training_mode=str(config["train"].get("joint_training_mode", "full")),
        )
        checkpoint_model = model
        optimization = build_stage_optimization(
            model,
            stage,
            runtime_config["train"],
            steps_per_epoch=steps_per_epoch,
        )
        epoch_key = "warmup_epochs" if stage is Stage.WARMUP else "joint_epochs"
        total_epochs = int(config["train"][epoch_key])
    scaler = build_grad_scaler(runtime_config["train"], device)

    split_payload = _load_split(split_manifest)
    _validate_split_contract(config, split_payload)
    pseudo_sha = None if pseudo_store is None else str(pseudo_store.metadata["bank_fingerprint"])
    start_epoch = global_step = 0
    best_selector = BestCheckpointSelector()
    best_teacher_loss = float("inf")
    resume_sampler_state: dict[str, int] | None = None
    resume_audit: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    resume_is_selected_best = False
    early_stopping_reset_epoch: int | None = None
    initialize_parent_best = bool(
        stage is Stage.JOINT
        and config["validation"].get("initialize_joint_best_from_parent", False)
    )
    if resume is not None:
        loaded = load_training_checkpoint(resume)
        resume_payload = loaded
        checkpoint_world_size = int(loaded.get("distributed_world_size", 1))
        if checkpoint_world_size != distributed.world_size:
            if not allow_world_size_transition:
                raise RuntimeError(
                    "Resume distributed world size differs from checkpoint: "
                    f"{distributed.world_size} != {checkpoint_world_size}"
                )
            if stage is not Stage.JOINT or int(loaded.get("micro_step_in_epoch", 0)) != 0:
                raise RuntimeError("World-size transition is allowed only at a Joint epoch boundary")
            sampler_state = loaded.get("sampler_state", {})
            if int(sampler_state.get("start_index", -1)) != 0:
                raise RuntimeError("World-size transition checkpoint is not at an epoch boundary")
            old_train = loaded["config"]["train"]
            old_effective_batch = int(old_train["batch_size_per_gpu"]) * int(
                old_train["gradient_accumulation"]
            )
            new_effective_batch = batch_size * global_accumulation
            if old_effective_batch != new_effective_batch:
                raise RuntimeError(
                    "World-size transition changes effective global batch size: "
                    f"{old_effective_batch} != {new_effective_batch}"
                )
        restored = restore_training_checkpoint(
            loaded,
            model=checkpoint_model,
            optimizer=optimization.optimizer,
            scheduler=optimization.scheduler,
            scaler=scaler,
            dataset_fingerprint=str(split_payload["dataset_fingerprint"]),
            split_manifest_sha256=sha256_file(split_manifest),
            pseudo_bank_sha256=pseudo_sha,
            current_config=config,
            git_commit=git_state.commit,
            allowed_checkpoint_git_commits=resume_source_git_commits,
            allowed_config_paths=(
                (("train", "joint_world_size_per_split"),)
                if allow_world_size_transition
                else ()
            ),
            allow_cuda_rng_device_count_reduction=allow_world_size_transition,
        )
        start_epoch = restored["epoch"]
        global_step = restored["global_step"]
        best_selector.load_state_dict(loaded.get("best_selector"))
        resume_is_selected_best = _resume_payload_is_selected_best(
            loaded,
            initialize_parent_best=initialize_parent_best,
        )
        best_teacher_loss = float(loaded.get("best_teacher_loss", float("inf")))
        resume_sampler_state = {
            "seed": int(loaded["sampler_state"]["seed"]),
            "epoch": int(loaded["sampler_state"]["epoch"]),
            "start_index": int(loaded["sampler_state"]["start_index"]),
        }
        if reset_early_stopping_on_resume:
            early_stopping_reset_epoch = start_epoch
        resume_audit = {
            "checkpoint": str(Path(resume).expanduser().resolve()),
            "checkpoint_epoch": int(loaded["epoch"]),
            "checkpoint_git_commit": str(loaded["git_commit"]),
            "current_git_commit": git_state.commit,
            "code_transition_authorized": loaded["git_commit"] != git_state.commit,
            "reason": "distributed_gradient_collective_recovery",
            "world_size_transition": {
                "enabled": bool(checkpoint_world_size != distributed.world_size),
                "from": checkpoint_world_size,
                "to": distributed.world_size,
                "epoch_boundary": True,
                "effective_global_batch_size": batch_size * global_accumulation,
            },
            "early_stopping_patience_reset_after_epoch": early_stopping_reset_epoch,
        }
        if distributed.is_primary and model is not None and resume_is_selected_best:
            best_path = output_root / "best_val_mpq.pt"
            _atomic_hardlink(Path(resume).expanduser().resolve(), best_path)
            _atomic_alias(best_path, output_root / "best.pt")
    if start_epoch >= total_epochs:
        raise RuntimeError(f"Stage is already complete at epoch {start_epoch}/{total_epochs}")

    if distributed.is_primary:
        write_run_provenance(
            output_root / "provenance.json",
            config=config,
            config_path=config_path,
            git_state=git_state,
            input_hashes={
                "all_samples": sha256_file(all_samples),
                "split_manifest": sha256_file(split_manifest),
                **_model_initialization_hashes(config, stage),
                **_joint_initialization_hashes(config, stage),
                **({"pseudo_bank": pseudo_sha} if pseudo_sha is not None else {}),
            },
        )
        atomic_write_json(
            output_root / "run_meta.json",
            {
                "formal_experiment": True,
                "stage": stage.value,
                "total_epochs": total_epochs,
                "device": str(device),
                "distributed": {
                    "world_size": distributed.world_size,
                    "global_gradient_accumulation": global_accumulation,
                    "local_gradient_accumulation": runtime_config["train"]["gradient_accumulation"],
                    "effective_global_batch_size": batch_size * global_accumulation,
                },
                "parameter_groups": list(optimization.parameter_groups),
                "early_stopping": dict(
                    config["validation"].get("early_stopping", {}).get(stage.value, {})
                ),
                "validation_checkpoint_rule": "max mpq, then max bpq, then earlier epoch",
                "resume": resume_audit,
            },
        )
        if resume_audit is not None:
            atomic_write_json(output_root / "resume_audit.json", resume_audit)
    distributed.barrier()
    validation_loader = None
    if model is not None:
        validation_dataset = PanNukeDataset(
            all_samples,
            split_manifest,
            role="validation",
            transform=build_model_input_transform(config),
        )
        validation_sampler = DistributedSequentialSampler(
            validation_dataset,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=int(config["runtime"]["num_workers"]),
            pin_memory=bool(config["runtime"]["pin_memory"]),
            persistent_workers=bool(
                config["runtime"]["persistent_workers"] and int(config["runtime"]["num_workers"]) > 0
            ),
            collate_fn=collate_pannuke,
        )

    history: list[dict[str, Any]] = []
    history_path = output_root / "history.json"
    if distributed.is_primary and history_path.is_file():
        with history_path.open(encoding="utf-8") as handle:
            history = json.load(handle)
    if distributed.is_primary and resume_payload is not None:
        recovered_row = _recover_history_row(
            resume_payload,
            history,
            is_best=resume_is_selected_best,
        )
        if recovered_row is not None:
            history.append(recovered_row)
            atomic_write_json(history_path, history)
            atomic_write_json(
                output_root / "resume_history_recovery.json",
                {
                    "checkpoint": str(Path(resume).expanduser().resolve()),
                    "recovered_epoch": int(recovered_row["epoch"]),
                    "validation_recovered": True,
                },
            )
    checkpoint_interval = int(config["train"].get("checkpoint_interval_epochs", 10))
    validation_interval = int(config["validation"]["interval_epochs"])
    if checkpoint_interval < 1 or validation_interval < 1:
        raise ValueError("Checkpoint and validation intervals must be positive")

    termination: dict[str, int | None] = {"signal": None}

    def request_stop(signum: int, _frame: Any) -> None:
        termination["signal"] = signum

    def distributed_stop_requested() -> bool:
        requested = distributed.any_flag(termination["signal"] is not None)
        if requested and termination["signal"] is None:
            termination["signal"] = signal.SIGTERM
        return requested

    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGUSR1"):
        handled_signals.append(signal.SIGUSR1)
    previous_handlers = {value: signal.getsignal(value) for value in handled_signals}
    for value in handled_signals:
        signal.signal(value, request_stop)

    if initialize_parent_best and resume is None:
        assert (
            model is not None
            and get_fast_pq is not None
            and binarize is not None
            and remap_label is not None
            and validation_loader is not None
        )
        baseline_accumulator = OfficialMetricAccumulator(
            get_fast_pq=get_fast_pq,
            binarize=binarize,
            remap_label=remap_label,
            match_iou=float(config["evaluation"]["pq_match_iou"]),
        )
        local_baseline_metrics = validate_student(
            model,
            validation_loader,
            stage=stage.value,
            config=config,
            device=device,
            accumulator=baseline_accumulator,
        )
        baseline_metrics = _gather_validation_metrics(
            local_baseline_metrics,
            baseline_accumulator,
            distributed,
        )
        if distributed.is_primary:
            assert baseline_metrics is not None
            if history:
                raise RuntimeError("Fresh parent-baseline initialization found non-empty history")
            if not best_selector.is_better(
                mpq=float(baseline_metrics["mpq"]),
                bpq=float(baseline_metrics["bpq"]),
                epoch=0,
            ):
                raise RuntimeError("Parent baseline failed to initialize the best-checkpoint selector")
            atomic_write_json(output_root / "validation" / "epoch_0000.json", baseline_metrics)
            baseline_sampler_state = {"seed": seed, "epoch": 0, "start_index": 0}
            baseline_payload = _checkpoint_for_epoch(
                epoch=0,
                global_step=0,
                checkpoint_model=checkpoint_model,
                optimization=optimization,
                scaler=scaler,
                config=config,
                git_commit=git_state.commit,
                split_payload=split_payload,
                split_manifest=split_manifest,
                pseudo_sha=pseudo_sha,
                sampler_state=baseline_sampler_state,
                best_selector=best_selector,
                formal=True,
                teacher=None,
                validation_metrics=baseline_metrics,
                distributed_world_size=distributed.world_size,
            )
            initial_path = output_root / "initial_parent.pt"
            best_path = output_root / "best_val_mpq.pt"
            atomic_torch_save(baseline_payload, initial_path)
            os.link(initial_path, best_path)
            os.link(initial_path, output_root / "last.pt")
            _atomic_alias(best_path, output_root / "best.pt")
            baseline_train = {
                "epoch": -1,
                "micro_batches": 0,
                "optimizer_updates": 0,
                "skipped_nonfinite_updates": 0,
                "skipped_no_gradient_batches": 0,
                "images": 0,
                "instances": 0,
                "total_loss": 0.0,
                "detection_loss": 0.0,
                "segmentation_loss": 0.0,
                "ugca_loss": 0.0,
                "learning_rates": {},
                "completed": True,
            }
            history.append(
                {
                    "epoch": 0,
                    "train": baseline_train,
                    "validation": baseline_metrics,
                    "is_best": True,
                    "parent_baseline": True,
                }
            )
            atomic_write_json(history_path, history)
            atomic_write_json(output_root / "parent_baseline.json", baseline_metrics)
        distributed.barrier()

    stopped_early = False
    early_stopping_report: dict[str, Any] | None = None
    for epoch in range(start_epoch, total_epochs):
        epoch_started_at = time.monotonic()
        start_index = (
            int(resume_sampler_state["start_index"]) if resume_sampler_state is not None and epoch == start_epoch else 0
        )
        loader, sampler = build_epoch_loader(
            dataset,
            seed=seed,
            epoch=epoch,
            start_index=start_index,
            batch_size=batch_size,
            num_workers=int(config["runtime"]["num_workers"]),
            pin_memory=bool(config["runtime"]["pin_memory"]),
            persistent_workers=bool(config["runtime"]["persistent_workers"]),
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        if teacher is not None:
            summary = train_teacher_epoch(
                teacher,
                loader,
                epoch=epoch,
                optimizer=optimization.optimizer,
                scheduler=optimization.scheduler,
                scaler=scaler,
                config=runtime_config,
                device=device,
                stop_requested=distributed_stop_requested,
            )
        else:
            assert model is not None
            local_summary = train_student_epoch(
                model,
                loader,
                epoch=epoch,
                stage=stage,
                strategy=str(config["supervision"]["strategy"]),
                pseudo_store=pseudo_store,
                optimizer=optimization.optimizer,
                scheduler=optimization.scheduler,
                scaler=scaler,
                config=runtime_config,
                device=device,
                stop_requested=distributed_stop_requested,
                live_progress_path=output_root / "live_progress.json",
            )
            summary = reduce_epoch_summary(local_summary, device)
        if teacher is not None:
            local_summary = summary
            summary = reduce_epoch_summary(local_summary, device)
        synchronize_module_buffers(checkpoint_model)
        global_step += summary.optimizer_updates
        if not summary.completed:
            sampler_state = sampler.state_dict()
            sampler_state.update(
                {
                    "epoch": epoch,
                    "start_index": start_index + local_summary.images,
                }
            )
            result = {
                "status": "interrupted_checkpoint_saved",
                "formal_experiment": True,
                "stage": stage.value,
                "epoch": epoch,
                "next_sample_index": sampler_state["start_index"],
                "signal": termination["signal"],
                "resume": str((output_root / "last.pt").resolve()),
            }
            if distributed.is_primary:
                emergency_payload = _checkpoint_for_epoch(
                    epoch=epoch,
                    global_step=global_step,
                    checkpoint_model=checkpoint_model,
                    optimization=optimization,
                    scaler=scaler,
                    config=config,
                    git_commit=git_state.commit,
                    split_payload=split_payload,
                    split_manifest=split_manifest,
                    pseudo_sha=pseudo_sha,
                    sampler_state=sampler_state,
                    best_selector=best_selector if model is not None else None,
                    formal=True,
                    teacher=teacher,
                    micro_step_in_epoch=start_index + local_summary.images,
                    distributed_world_size=distributed.world_size,
                )
                emergency_payload["termination_signal"] = termination["signal"]
                if teacher is not None:
                    emergency_payload["best_teacher_loss"] = best_teacher_loss
                emergency_path = output_root / "emergency.pt"
                atomic_torch_save(emergency_payload, emergency_path)
                _atomic_hardlink(emergency_path, output_root / "last.pt")
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train": summary.to_dict(),
                        "validation": None,
                        "is_best": False,
                        "interrupted": True,
                        "epoch_elapsed_seconds": time.monotonic() - epoch_started_at,
                    }
                )
                atomic_write_json(history_path, history)
                atomic_write_json(output_root / "result.json", result)
            distributed.close()
            for value, handler in previous_handlers.items():
                signal.signal(value, handler)
            return result
        validation_metrics = None
        is_best = False
        if model is not None and ((epoch + 1) % validation_interval == 0 or epoch + 1 == total_epochs):
            assert (
                get_fast_pq is not None
                and binarize is not None
                and remap_label is not None
                and validation_loader is not None
            )
            local_accumulator = OfficialMetricAccumulator(
                get_fast_pq=get_fast_pq,
                binarize=binarize,
                remap_label=remap_label,
                match_iou=float(config["evaluation"]["pq_match_iou"]),
            )
            local_validation_metrics = validate_student(
                model,
                validation_loader,
                stage=stage.value,
                config=config,
                device=device,
                accumulator=local_accumulator,
            )
            validation_metrics = _gather_validation_metrics(
                local_validation_metrics,
                local_accumulator,
                distributed,
            )
            if distributed.is_primary:
                assert validation_metrics is not None
                selection_epoch = epoch + 1 if initialize_parent_best else epoch
                passes_parent_guard = True
                if initialize_parent_best:
                    baseline_rows = [row for row in history if row.get("parent_baseline", False)]
                    if len(baseline_rows) != 1:
                        raise RuntimeError("Controlled probe requires exactly one parent baseline row")
                    baseline_mpq = float(baseline_rows[0]["validation"]["mpq"])
                    required_gain = float(
                        config["validation"]["early_stopping"][stage.value].get("min_delta", 0.0)
                    )
                    passes_parent_guard = float(validation_metrics["mpq"]) > baseline_mpq + required_gain
                if passes_parent_guard:
                    is_best = best_selector.is_better(
                        mpq=float(validation_metrics["mpq"]),
                        bpq=float(validation_metrics["bpq"]),
                        epoch=selection_epoch,
                    )
                atomic_write_json(
                    output_root / "validation" / f"epoch_{epoch + 1:04d}.json",
                    validation_metrics,
                )
        elif distributed.is_primary and teacher is not None and summary.total_loss < best_teacher_loss:
            best_teacher_loss = summary.total_loss
            is_best = True

        sampler_state = sampler.state_dict()
        sampler_state.update({"epoch": epoch + 1, "start_index": 0})
        if distributed.is_primary:
            payload = _checkpoint_for_epoch(
                epoch=epoch + 1,
                global_step=global_step,
                checkpoint_model=checkpoint_model,
                optimization=optimization,
                scaler=scaler,
                config=config,
                git_commit=git_state.commit,
                split_payload=split_payload,
                split_manifest=split_manifest,
                pseudo_sha=pseudo_sha,
                sampler_state=sampler_state,
                best_selector=best_selector if model is not None else None,
                formal=True,
                teacher=teacher,
                validation_metrics=validation_metrics,
                distributed_world_size=distributed.world_size,
            )
            if teacher is not None:
                payload["best_teacher_loss"] = best_teacher_loss
            last_path = output_root / "last.pt"
            atomic_torch_save(payload, last_path)
            if is_best:
                if teacher is not None:
                    _atomic_hardlink(last_path, output_root / "best.pt")
                else:
                    best_path = output_root / "best_val_mpq.pt"
                    _atomic_hardlink(last_path, best_path)
                    _atomic_alias(best_path, output_root / "best.pt")
            if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == total_epochs:
                milestone = output_root / f"epoch_{epoch + 1:04d}.pt"
                milestone.unlink(missing_ok=True)
                os.link(output_root / "last.pt", milestone)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": summary.to_dict(),
                    "validation": validation_metrics,
                    "is_best": is_best,
                    "epoch_elapsed_seconds": time.monotonic() - epoch_started_at,
                }
            )
            atomic_write_json(history_path, history)
            if model is not None and validation_metrics is not None:
                stage_settings = dict(config["validation"].get("early_stopping", {}).get(stage.value, {}))
                early_stopping_history = history
                if early_stopping_reset_epoch is not None:
                    early_stopping_history = [
                        row
                        for row in history
                        if int(row["epoch"]) >= early_stopping_reset_epoch
                    ]
                stopped_early, early_stopping_report = _early_stop_from_history(
                    early_stopping_history,
                    settings=stage_settings,
                )
                if early_stopping_report is not None and early_stopping_reset_epoch is not None:
                    early_stopping_report["patience_reset_after_epoch"] = early_stopping_reset_epoch
                if stopped_early:
                    atomic_write_json(
                        output_root / "early_stopping.json",
                        {
                            **(early_stopping_report or {}),
                            "stopped_at_epoch": epoch + 1,
                            "best_checkpoint": str((output_root / "best_val_mpq.pt").resolve()),
                        },
                    )
        stopped_early = distributed.any_flag(stopped_early)
        distributed.barrier()
        resume_sampler_state = None
        if stopped_early:
            break

    frozen_decision = None
    if distributed.is_primary and stage is Stage.JOINT:
        best_path = output_root / "best_val_mpq.pt"
        best_state = best_selector.state_dict()
        if not best_path.is_file() or best_state is None:
            raise RuntimeError("Joint stage completed without a validation-selected checkpoint")
        if bool(config["validation"].get("defer_test_freeze_until_threshold_calibration", False)):
            frozen_decision = {
                "status": "awaiting_validation_threshold_calibration",
                "checkpoint": str(best_path.resolve()),
                "validation_metrics": {"mpq": best_state["mpq"], "bpq": best_state["bpq"]},
                "test_access": False,
            }
            atomic_write_json(output_root / "threshold_calibration_required.json", frozen_decision)
        else:
            frozen_path = output_root / "frozen_test_decision.json"
            frozen_decision = write_frozen_test_decision(
                frozen_path,
                checkpoint=best_path,
                validation_metrics={"mpq": best_state["mpq"], "bpq": best_state["bpq"]},
                inference=dict(config["inference"]),
                config=config,
            )
    result = {
        "status": "completed",
        "formal_experiment": True,
        "stage": stage.value,
        "epochs": epoch + 1,
        "epoch_cap": total_epochs,
        "stopped_early": stopped_early,
        "early_stopping": early_stopping_report,
        "global_step": global_step,
        "best": ({"train_loss": best_teacher_loss} if teacher is not None else best_selector.state_dict()),
        "frozen_test_decision": frozen_decision,
    }
    if distributed.is_primary:
        atomic_write_json(output_root / "result.json", result)
    distributed.close()
    for value, handler in previous_handlers.items():
        signal.signal(value, handler)
    return result
