"""Training of a transformer-based offensive speech classifier."""

from typing import Dict

import hydra
from datasets import Dataset, DatasetDict, load_metric
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
)

from .load_data import load_annotated_data
from .training_args_with_mps_support import TrainingArgumentsWithMPSSupport


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_transformer_model(config: DictConfig) -> AutoModelForSequenceClassification:
    """Training of a transformer-based offensive speech classifier.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        AutoModelForSequenceClassification:
            The trained model.
    """
    # Load the data
    data_dict = load_annotated_data(config)
    train_df = data_dict["train"]
    val_df = data_dict["val"]
    test_df = data_dict["test"]

    # Only keep the `text` and `label` columns
    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    # Convert the data to Hugging Face Dataset objects
    train = Dataset.from_pandas(train_df, split="train", preserve_index=False)
    val = Dataset.from_pandas(val_df, split="val", preserve_index=False)
    test = Dataset.from_pandas(test_df, split="test", preserve_index=False)

    # Collect the data into a DatasetDict
    dataset = DatasetDict(train=train, val=val, test=test)

    # Get model config
    model_config = config.transformer_model

    # Create the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=model_config.padding
    )

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_id,
        use_auth_token=model_config.use_auth_token,
        cache_dir=model_config.cache_dir,
        from_flax=model_config.from_flax,
        num_labels=2,
    )

    # Tokenise the data
    def tokenise(examples: dict) -> dict:
        doc = examples["text"]
        return tokenizer(doc, truncation=True, padding=True)

    dataset = dataset.map(tokenise, batched=True)

    # Initialise the metrics
    mcc_metric = load_metric("matthews_correlation")
    f1_metric = load_metric("f1")

    # Create the `compute_metrics` function
    def compute_metrics(predictions_and_labels: EvalPrediction) -> Dict[str, float]:
        """Compute the metrics for the transformer model.

        Args:
            predictions_and_labels (EvalPrediction):
                A tuple of predictions and labels.

        Returns:
            Dict[str, float]:
                The metrics.
        """
        # Extract the predictions
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)

        # Compute the metrics
        mcc = mcc_metric.compute(predictions=predictions, references=labels)[
            "matthews_correlation"
        ]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]

        return dict(mcc=mcc, f1=f1)

    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=model_config.early_stopping_patience
    )

    # Set up output directory
    if config.testing:
        output_dir = f"{config.models.dir}/test_{model_config.name}"
    else:
        output_dir = f"{config.models.dir}/{model_config.name}"

    # Create the training arguments
    training_args = TrainingArgumentsWithMPSSupport(
        output_dir=output_dir,
        evaluation_strategy=model_config.evaluation_strategy,
        logging_strategy=model_config.logging_strategy,
        save_strategy=model_config.save_strategy,
        eval_steps=model_config.eval_steps,
        logging_steps=model_config.logging_steps,
        save_steps=model_config.save_steps,
        max_steps=model_config.max_steps,
        report_to=model_config.report_to,
        save_total_limit=model_config.save_total_limit,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        learning_rate=model_config.learning_rate,
        warmup_ratio=model_config.warmup_ratio,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        optim=model_config.optim,
        seed=config.seed,
        full_determinism=model_config.full_determinism,
        lr_scheduler_type=model_config.lr_scheduler_type,
        fp16=model_config.fp16,
        metric_for_best_model="mcc",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    print(trainer.evaluate(dataset["test"]))

    # Save the model
    trainer.save_model()

    # Return the model
    return model


if __name__ == "__main__":
    train_transformer_model()
