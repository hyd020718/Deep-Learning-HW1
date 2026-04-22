import itertools
import os
from typing import Dict, List, Optional
import numpy as np
from .config import *
from .data import *
from .tools import * 
from .model import *
from .visualize import *


def evaluate_split(
    model: MLP3,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    weight_decay: float = 0.0,
) -> Dict[str, object]:
    logits_list = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        logits_list.append(model.forward(X[start:end]))

    logits = np.concatenate(logits_list, axis=0)
    criterion = CrossEntropyLoss()
    ce_loss = criterion.forward(logits, y)
    loss = ce_loss + weight_decay * model.l2_penalty()
    y_pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y, y_pred)
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "y_pred": y_pred,
    }



def train_experiment(
    config: ExperimentConfig,
    save_best_model: bool = True,
    make_plots: bool = True,
    save_split_indices: bool = True,
) -> Dict[str, object]:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    plots_dir = os.path.join(config.output_dir, "plots")
    if make_plots:
        ensure_dir(plots_dir)

    save_json(config.to_dict(), os.path.join(config.output_dir, "config.json"))

    X, y, class_names = load_eurosat(config.data_root, config.image_size)
    split_indices = make_split_indices(len(X), config.train_ratio, config.val_ratio, config.seed)

    if save_split_indices:
        np.savez(
            os.path.join(config.output_dir, "split_indices.npz"),
            train=split_indices["train"],
            val=split_indices["val"],
            test=split_indices["test"],
        )

    splits = apply_split(X, y, split_indices)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    X_train = standardize(X_train)
    X_val = standardize(X_val)
    X_test = standardize(X_test)

    model = MLP3(
        input_dim=X_train.shape[1],
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        num_classes=config.num_classes,
        activation=config.activation,
    )
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters_and_grads, lr=config.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_path = os.path.join(config.output_dir, config.checkpoint_name)

    for epoch in range(config.epochs):
        total_loss = 0.0
        total_seen = 0

        for xb, yb in batch_iterator(X_train, y_train, batch_size=config.batch_size, shuffle=True):
            logits = model.forward(xb)
            ce_loss = criterion.forward(logits, yb)
            loss = ce_loss + config.weight_decay * model.l2_penalty()

            grad_logits = criterion.backward()
            model.backward(grad_logits)
            model.add_l2_grads(config.weight_decay)
            optimizer.step()

            batch_n = xb.shape[0]
            total_loss += float(loss) * batch_n
            total_seen += batch_n

        train_loss = total_loss / max(total_seen, 1)
        val_result = evaluate_split(
            model,
            X_val,
            y_val,
            batch_size=max(256, config.batch_size),
            weight_decay=config.weight_decay,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_result["loss"]))
        history["val_acc"].append(float(val_result["accuracy"]))
        history["lr"].append(float(optimizer.lr))

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = float(val_result["accuracy"])

            if save_best_model:
                metadata = {
                    "class_names": np.array(class_names, dtype=object),
                    "image_size": np.array(config.image_size),
                }
                model.save(best_path, metadata=metadata)

        print(
            f"Epoch [{epoch + 1:03d}/{config.epochs:03d}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_acc={val_result['accuracy']:.4f} | "
            f"lr={optimizer.lr:.6f}"
        )

        optimizer.set_lr(optimizer.lr * config.lr_decay)

    save_json(history, os.path.join(config.output_dir, "history.json"))

    if make_plots:
        plot_training_curves(history, plots_dir)

    cm = None
    test_acc = None
    checkpoint_path = best_path if save_best_model else None

    if save_best_model:
        if not os.path.exists(best_path):
            raise RuntimeError("Best checkpoint was not saved. Please check training logic.")

        best_model, _ = MLP3.from_checkpoint(best_path)
        test_result = evaluate_split(
            best_model,
            X_test,
            y_test,
            batch_size=max(256, config.batch_size),
        )
        test_acc = float(test_result["accuracy"])
        cm = confusion_matrix(y_test, test_result["y_pred"], config.num_classes)

        np.save(os.path.join(config.output_dir, "confusion_matrix.npy"), cm)
        np.savetxt(os.path.join(config.output_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

        if make_plots:
            plot_confusion_matrix(
                cm,
                class_names,
                os.path.join(plots_dir, "confusion_matrix.png"),
            )

        print("\nBest validation accuracy:", best_val_acc)
        print("Test accuracy:", test_acc)
        print("Confusion matrix:\n", cm)
    else:
        print("\nBest validation accuracy:", best_val_acc)
        print("Test stage skipped because save_best_model=False.")

    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "class_names": class_names,
        "checkpoint_path": checkpoint_path,
        "tested_on_test_set": save_best_model,
    }
    save_json(summary, os.path.join(config.output_dir, "summary.json"))

    return {
        "config": config.to_dict(),
        "history": history,
        "summary": summary,
        "confusion_matrix": cm,
    }


def evaluate_from_run_dir(
    run_dir: str,
    batch_size: int = 256,
    output_subdir: Optional[str] = "eval"
) -> Dict[str, object]:
    config = ExperimentConfig(**load_json(os.path.join(run_dir, "config.json")))
    ckpt_path = os.path.join(run_dir, config.checkpoint_name)
    split_path = os.path.join(run_dir, "split_indices.npz")

    model, meta = MLP3.from_checkpoint(ckpt_path)
    image_size = meta["image_size"]

    X, y, class_names = load_eurosat(config.data_root, image_size)
    split_file = np.load(split_path)
    split_indices = {
        "train": split_file["train"],
        "val": split_file["val"],
        "test": split_file["test"],
    }

    splits = apply_split(X, y, split_indices)
    X_test, y_test = splits["test"]
    X_test = standardize(X_test)

    result = evaluate_split(model, X_test, y_test, batch_size=batch_size)
    y_pred = result["y_pred"]
    cm = confusion_matrix(y_test, y_pred, len(class_names))

    # 错例信息
    wrong_mask = (y_pred != y_test)
    wrong_test_indices = np.where(wrong_mask)[0]                 # 在测试集中的序号
    wrong_global_indices = split_indices["test"][wrong_test_indices]  # 在原始全数据中的序号
    wrong_true = y_test[wrong_mask]
    wrong_pred = y_pred[wrong_mask]

    if output_subdir is not None:
        out_dir = os.path.join(run_dir, output_subdir)
        plots_dir = os.path.join(out_dir, "plots")
        ensure_dir(out_dir)
        ensure_dir(plots_dir)

        np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
        np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
        plot_confusion_matrix(cm, class_names, os.path.join(plots_dir, "confusion_matrix.png"))

        # 保存错例序号与标签
        np.savez(
            os.path.join(out_dir, "misclassified_indices.npz"),
            wrong_test_indices=wrong_test_indices,
            wrong_global_indices=wrong_global_indices,
            true_labels=wrong_true,
            pred_labels=wrong_pred,
        )

        # 另存一份 csv，方便直接查看
        wrong_table = np.column_stack([
            wrong_test_indices,
            wrong_global_indices,
            wrong_true,
            wrong_pred
        ])
        np.savetxt(
            os.path.join(out_dir, "misclassified_indices.csv"),
            wrong_table,
            delimiter=",",
            fmt="%d",
            header="wrong_test_index,wrong_global_index,true_label,pred_label",
            comments=""
        )

        save_json(
            {
                "test_acc": float(result["accuracy"]),
                "num_test_samples": int(len(y_test)),
                "num_misclassified": int(len(wrong_test_indices)),
            },
            os.path.join(out_dir, "summary.json")
        )

    print("Test accuracy:", result["accuracy"])
    print("Confusion matrix:\n", cm)
    print("Number of misclassified samples:", len(wrong_test_indices))

    return {
        "test_acc": float(result["accuracy"]),
        "confusion_matrix": cm,
        "class_names": class_names,
        "wrong_test_indices": wrong_test_indices,
        "wrong_global_indices": wrong_global_indices,
        "true_labels": wrong_true,
        "pred_labels": wrong_pred,
    }


def grid_search(base_config: ExperimentConfig, search_space: Dict[str, List[object]]) -> List[Dict[str, object]]:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    base_output_dir = base_config.output_dir
    ensure_dir(base_output_dir)

    results = []
    for run_id, combo in enumerate(itertools.product(*values), start=1):
        cfg = ExperimentConfig(**base_config.to_dict())
        parts = []
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)
            parts.append(f"{k}-{str(v).replace('/', '_')}")
        cfg.hidden_dim2 = cfg.hidden_dim1 // 2
        cfg.output_dir = os.path.join(base_output_dir, f"run_{run_id:03d}__" + "__".join(parts))

        print("\n" + "=" * 70)
        print("Running config:", {k: getattr(cfg, k) for k in keys})
        print("Output dir:", cfg.output_dir)
        print("=" * 70)

        result = train_experiment(cfg,make_plots=False,save_best_model=False,save_split_indices=False)
        results.append(
            {
                "searched_params": {k: getattr(cfg, k) for k in keys},
                "best_val_acc": result["summary"]["best_val_acc"],
                "output_dir": cfg.output_dir,
            }
        )

    results.sort(key=lambda x: x["best_val_acc"], reverse=True)
    save_json(results, os.path.join(base_output_dir, "search_results.json"))
    return results
