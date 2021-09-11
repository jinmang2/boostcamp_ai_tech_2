# Section 1. 머신러닝 프로젝트 실험관리
- 출처: https://www.inflearn.com/course/머신러닝-엔지니어-실무/

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/o-stage/mlops)

## 실험관리 - Weights and Biases

### 1.1 실험 관리 문제
- 저번에 학습 잘 됐었는데... `learning rate`가 몇이었지?
- 저번에 썼던 논문 실험결과 재연이 안되네?
- 데이터 셋 버전은?
- 가장 성능 좋은 모델은?
- 어떤 Hyperparameter가 제일 중요한지?

### 1.2 Weights and Biases 소개
- 미국의 스타트업, 기업 단위는 가격 정책있음
- 어떤 기능들이 있는가?
    - 실험 관리 기능
    - Fast integration
        ```python
        import torch
        import torch.nn as nn

        import wandb
        wandb.init(project="pedestrian-detection")

        # log any metric from your training script
        wandb.log({"acc": accuracy, "val_acc": val_accuracy})
        ```
    - 어디에서나 접근 가능
    - 학습 시각화 가능
    - 모델 재현 가능 (configuration)
    - GPU 사용률 체크 가능
    - 실시간 디버깅 가능
    - https://wandb.ai/site/experiment-tracking


### 1.3 Weights and Biases 실습
- wandb에서 기본적으로 제공하는 튜토리얼
- https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb
- tensorboard의 상위 호환 버전
- log 보낼 때 `wandb.Image`로 보내면 이미지도 보내기 가능
- `wandb.init` 프로젝트 설정
    - 인자로 config 설정 가능! (실험 재현을 위해서)
- `from wandb.keras import WandbCallback` 등으로 랩업 가능


## 하이퍼 파라미터 최적화 - W&B Sweeps

### 1.4 Hyper Parameter Optimization
- 꽤 중요한 문제!
- AutoML, 엔지니어의 특징? 자동화!
- Continuous Integration
- 다양한 알고리즘이 존재함!
- hyperparameter importance도 뽑아볼 수 있다니!

### 1.5 W&B Sweeps 소개
- `wandb.init(config=default_config, magic=True)`
- `sweep_id = wandb.sweep(sweep_config)`
    - method로 어떤 HPO를 사용할지!
- `wandb.agent(sweep_id, function=train)`
- 원격으로 죽일 수도 있음
- parallel coordinate로 visualization
- report로 뽑는거 너무 좋음!

### 1.6 W&B Sweeps 실습
- wandb 공식 튜토리얼
- https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-python/notebook.ipynb

## 추가 학습
- huggingface는 `wandb.sweep`에서 래핑된 ray나 optuna를 안쓰고 직접 hp_search하는 듯
- sweep으로 결과 정리해주려면 어떻게 해야할까? 9월에 프로젝트하면서 실험해보기!
- PBT 등 다양한 hyperparameter tuning 실험해보기
- https://gitbook-docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune
- https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498
- https://wandb.ai/wandb/huggingtweets/sweeps/g5hwbevb?workspace=user-jinmang2

```python
# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None
```

```python
# from transformers.file_utils import ENV_VARS_TRUE_VALUES
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


class WandbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        assert has_wandb, "WandbCallback requires wandb to be installed. Run `pip install wandb`."
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log outputs
        self._log_model = os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training. Use along with
                `TrainingArguments.load_best_model_at_end` to upload best model.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})
```
