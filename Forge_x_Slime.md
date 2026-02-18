# Slime: Architecture, Engine, and Comparison with Forge

## Part 1: Infrastructure & Engine

### Training Backend

Slime uses Megatron-LM as its primary training engine (with an experimental FSDP2 backend). This gives it the full Megatron parallelism stack: tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), context parallelism (CP), and expert parallelism (EP).

Forge uses TorchTitan via ForgeEngine, which wraps model loading, FSDP2/TP, checkpointing, optimizers, and AMP into a single engine object. Forge currently raises `NotImplementedError` for PP (`titan.py:133`).

|             | Slime                     | Forge                     |
|-------------|---------------------------|---------------------------|
| Engine      | Megatron-LM               | TorchTitan (ForgeEngine)  |
| Parallelism | TP + PP + DP + CP + EP    | FSDP2 + TP (no PP yet)    |
| Inference   | SGLang (separate process) | vLLM (as a Monarch actor) |

### Orchestration

Slime uses Ray actors and placement groups. Each GPU gets a `TrainRayActor`, grouped into a `RayTrainGroup`. SGLang engines run as separate Ray actors. Communication between training and inference is via Ray object refs + NCCL.

Forge uses Monarch actors. `ForgeActor` is the base class for `TitanTrainer`, `ReferenceModel`, `ComputeAdvantages`, `RewardActor`, and `Generator`. Actors communicate via Monarch endpoints and `as_service()` RPCs.

|               | Slime                            | Forge                      |
|---------------|----------------------------------|----------------------------|
| Orchestration | Ray                              | Monarch                    |
| Actor base    | TrainRayActor (Ray remote)       | ForgeActor (Monarch actor) |
| RPC mechanism | `ray.get(actor.method.remote())` | `await service.method(args)` |

### Config System

Slime uses argparse with ~1500 lines of CLI flags (`arguments.py`), optionally overlaid with YAML via `--custom-config-path`. Configuration is a flat Namespace object threaded through every function.

Forge uses Pydantic models + YAML. Loss configs inherit from `BaseLossConfig(BaseModel)` with `extra="forbid"` (catches typos at construction). The trainer is a `@dataclass` with typed fields validated in `__post_init__`.

---

## Part 2: Weight Synchronization

This is the biggest architectural difference.

### Slime: CPU-Backed Weight Swapping + NCCL Broadcast to SGLang

Slime keeps multiple named weight snapshots in CPU pinned memory via `TensorBackuper` (`tensor_backper.py`):

```python
weights_backuper.backup("actor")       # GPU -> CPU after training step
weights_backuper.backup("ref")         # snapshot reference model
weights_backuper.restore("ref")        # CPU -> GPU for ref forward pass
weights_backuper.restore("actor")      # CPU -> GPU to resume training
```

The training loop swaps between actor/ref weights on the same GPU by copying to/from CPU pinned memory. Then, to sync with SGLang (which runs in a separate process), Slime has two strategies:

- **Colocated** (`UpdateWeightFromTensor`): Same node -- Megatron weights are gathered across TP/PP/EP ranks, converted from Megatron format to HuggingFace format, serialized, then sent via Ray IPC + Gloo `gather_object`.
- **Distributed** (`UpdateWeightFromDistributed`): Different nodes -- creates a temporary NCCL process group between trainer rank 0 and all SGLang engine ranks, then broadcasts each tensor.

The weight format conversion (Megatron -> HF) is non-trivial -- there are per-architecture converters for Qwen2, Qwen3, Qwen3MoE, DeepSeekV3, LLaMA, GLM4, etc.

### Forge: Actor-Level Weight Sync via Monarch

Forge keeps the reference model as a separate Monarch actor (`ReferenceModel`) with its own GPU allocation. There is no weight swapping on the same GPU. The generator (vLLM) also runs as its own actor. Weight sync between actors happens through TorchStore: `push_weights()` exports the trainer's state dict in HF format to TorchStore, then `generator.update_weights()` pulls from TorchStore.

|                   | Slime                                       | Forge                             |
|-------------------|---------------------------------------------|-----------------------------------|
| Ref model         | Same GPU, CPU-backed swap (TensorBackuper)  | Separate actor, own GPU           |
| Generator sync    | NCCL broadcast or IPC + format conversion   | TorchStore put/get                |
| Format conversion | Megatron -> HF per-architecture converters  | Not needed (same format)          |
| Complexity        | ~2000 lines across update_weight/           | Handled by TorchStore + vLLM actor|

**Trade-off:** Slime's approach is more GPU-efficient (one GPU runs both actor and ref model by swapping), but introduces significant complexity (CPU backup management, format conversion, temporary NCCL groups). Forge's approach is simpler (each role has its own GPU) but uses more GPUs.

---

## Part 3: Fault Tolerance

### Slime: Rollout Engine Recovery Only

Slime has a `RolloutHealthMonitor` -- a daemon thread that polls SGLang engines via `/health_generate` every N seconds. If an engine times out:

1. Health monitor calls `ray.kill(engine)`, sets slot to None
2. Before next weight update, `recover_rollout_engines()` spawns a replacement
3. New NCCL groups are established, weights are pushed to the new engine

What is NOT covered:
- No training worker fault tolerance -- if any training GPU dies, the entire job crashes
- No elastic scaling
- No gradient anomaly detection (NaN/Inf)
- No automatic restart from checkpoint -- manual `--load` required

### Forge

Forge has no explicit fault tolerance mechanism in the codebase examined. It relies on the external orchestrator (Monarch/cluster manager) for process recovery.

|                           | Slime                          | Forge                   |
|---------------------------|--------------------------------|-------------------------|
| Inference engine recovery | Yes (health monitor + restart) | No explicit mechanism   |
| Training worker recovery  | No                             | No                      |
| Elastic scaling           | No                             | No                      |
| Checkpoint format         | Megatron native or DCP (FSDP)  | TorchTitan checkpointer |

---

## Part 4: SFT -- Side-by-Side

### Slime SFT

SFT in Slime is a special case of the RL pipeline with flags that disable RL-specific features:

```
--loss-type sft_loss
--disable-compute-advantages-and-returns
--debug-train-only                         # no SGLang engines
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
```

The SFT "rollout" function (`sft_rollout.py`) does no inference -- it tokenizes conversation messages and generates loss masks via `MultiTurnLossMaskGenerator`. The loss function (`loss.py:755`) is standard NLL:

```python
def sft_loss_function(args, batch, logits, sum_of_sample_mean):
    log_probs = get_log_probs_and_entropy(logits, ...)["log_probs"]
    loss = -sum_of_sample_mean(torch.cat(log_probs, dim=0))
    return loss, {"loss": loss.clone().detach()}
```

### Forge SFT

SFT in Forge is a first-class module (`apps/sft/`). The `ForgeSFTRecipe` inherits both `ForgeActor` and `ForgeEngine`, combining actor orchestration and training engine into one class. The loss uses TorchTitan's built-in `LossFunction`:

```python
loss = self.loss_fn(pred, labels)
```

Key differences:

| Aspect         | Slime SFT                                                   | Forge SFT                                         |
|----------------|-------------------------------------------------------------|----------------------------------------------------|
| Architecture   | RL pipeline with SFT flags                                  | Dedicated SFT module (`ForgeSFTRecipe`)            |
| Loss function  | Embedded in 800-line loss.py                                | TorchTitan built-in `LossFunction`                 |
| Metrics        | Loss only                                                   | Loss + accuracy + perplexity                       |
| Mask generation| MultiTurnLossMaskGenerator (tokenizer-specific logic)       | SFTOutputTransform (tokenizer-agnostic, vectorized)|
| Data pipeline  | Conversation -> Sample -> Ray object ref -> RolloutBatch    | Dataset -> dict -> collate_padded -> batch         |
| Config         | argparse flags                                              | OmegaConf/YAML with typed dataclasses              |

**Trade-off:** Slime's "SFT as special-case RL" means one pipeline to maintain, but SFT inherits RL complexity (rollout batch format, `sum_of_sample_mean` reducer, packed sequence params). Forge's dedicated SFT module is simpler and self-contained but introduces a separate code path.

---

## Part 5: GRPO -- Side-by-Side

### Slime GRPO

GRPO lives inside the monolithic `policy_loss_function` (`loss.py:476`, ~220 lines). The advantage is computed separately in `compute_advantages_and_returns` (`loss.py:263`):

```python
# Advantage = broadcast scalar reward to every token
rewards = torch.tensor(rewards)
returns = [torch.ones_like(kl[i]) * rewards[i] for i in range(len(rewards))]
```

The policy gradient loss uses `compute_policy_loss` from `ppo_utils.py`:

```python
ratio = (-ppo_kl).exp()
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

KL, entropy, OPSM, TIS, GSPO sequence-level KL, and OPD distillation are all handled within the same function via if branches.

### Forge GRPO

GRPO is a self-contained class (`rl/losses.py`), using shared primitives:

```python
class GRPOLoss(BaseLossConfig):
    def __call__(self, logits, target_ids, advantages, generator_logprobs, loss_mask, ref_logprobs=None):
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)
        ratio, log_ratio, ratio_m = compute_ratio(logprobs, generator_logprobs, loss_mask)
        pg_loss, clip_m = pg_ppo_clip(ratio, advantages, loss_mask, self.clip_low, self.clip_high)
        if self.beta > 0:
            kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask)
            pg_loss = pg_loss + self.beta * kl
        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)
        return LossOutput(loss, lp_m + ent_m + ratio_m + clip_m + kl_m + agg_m)
```

Key differences:

| Aspect            | Slime GRPO                                              | Forge GRPO                                               |
|-------------------|---------------------------------------------------------|----------------------------------------------------------|
| Loss code         | ~220 lines in monolithic function                       | Self-contained class + shared ops.py primitives          |
| Loss variants     | All in one function with if branches (GSPO, OPSM, etc) | Separate classes: GRPOLoss, DAPOLoss, etc.               |
| Aggregation       | `sum_of_sample_mean` callback                           | `aggregate()` with enum (fixed_horizon, sequence_mean)   |
| KL estimator      | `compute_approx_kl` with k1/k2/k3/low_var_kl           | `compute_kl` using k3 only                               |
| Compilation       | `@torch.compile(dynamic=True)` on `compute_approx_kl`  | `torch.compile(self.loss)` wrapping the entire loss      |
| Ratio computation | Inline: `(-ppo_kl).exp()`                               | Dedicated `compute_ratio()` with token/sequence modes    |
| Advantage compute | Inside training actor (`compute_advantages_and_returns`)| Separate `ComputeAdvantages` Monarch actor               |
| Advantage types   | GRPO, GSPO, PPO-GAE, REINFORCE++, etc.                 | Group-relative normalization in `rl/advantage.py`        |
| DR-GRPO support   | Not mentioned                                           | Default -- uses `fixed_horizon` aggregation              |

**Trade-off:** Slime packs more algorithmic variants into fewer functions -- one `policy_loss_function` handles GRPO, GSPO, PPO, OPSM, TIS, OPD. This is practical for rapid experimentation (add an if branch, not a new file) but makes each variant harder to read in isolation. Forge separates each variant into its own class with shared primitives -- each is self-contained and auditable, but adding a new variant means creating a new file and composing primitives yourself.

---

## Part 6: Training Loop Structure -- Side-by-Side

This section compares the overall training loop orchestration: how generation, training, weight sync, and checkpointing are sequenced.

### Slime: Centralized Driver Loop

Slime's training loop is a centralized Python script (`train.py:65-94`) that drives everything via Ray remote calls. The driver iterates over a fixed number of **rollouts** (not steps or epochs):

```python
# train.py (simplified)
for rollout_id in range(start_rollout_id, num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))  # BLOCKING
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))           # BLOCKING
    actor_model.update_weights()                                             # push to SGLang
```

The full per-iteration sequence is:

1. **[Optional] Eval** -- `rollout_manager.eval.remote()`
2. **Generate rollout** -- `rollout_manager.generate.remote()` dispatches to SGLang engines via async HTTP, collects responses + rewards, converts to training tensors, splits by DP rank (`train.py:69`)
3. **[Optional] Offload rollout engines** -- free GPU memory for training (`train.py:71-72`)
4. **Train actor** (and critic if enabled) -- `actor_model.async_train()` dispatches to all Ray training actors in parallel (`train.py:74-80`)
5. **Save checkpoint** -- at configurable intervals (`train.py:82-83`)
6. **Offload training model** -- free GPU memory for generation (`train.py:85`)
7. **Update weights** -- push trained weights to SGLang engines (`train.py:88`)
8. **[Optional] Eval** (`train.py:92-93`)

**Inside each training actor** (FSDP path, `fsdp_utils/actor.py`):

```
_train_core():
  1. compute_advantages_and_returns()    # reward -> per-token advantages
  2. _packed_data()                      # pack sequences into micro-batches
  3. forward(ref_model) -> ref_log_probs # if KL penalty needed
  4. forward(actor_model) -> log_probs   # forward-only for old policy probs
  5. for each micro-batch:
       _train_step():
         logits = model(**inputs)        # forward
         loss = policy_loss(logits, ...)  # GRPO/PPO/SFT loss
         loss.backward()                  # backward
         if at grad_accum boundary:
           clip_grad_norm_()
           optimizer.step()
           lr_scheduler.step()
           optimizer.zero_grad()
```

**Inside each training actor** (Megatron path, `megatron_utils/actor.py`):

```
train_actor():
  1. switch_model("ref")  -> compute_log_prob()  # ref logprobs
  2. switch_model("actor") -> compute_log_prob()  # actor logprobs
  3. compute_advantages_and_returns()
  4. train() -> for step_id in range(num_steps_per_rollout):
       train_one_step():
         forward_backward_func(...)       # Megatron pipeline schedule
         optimizer.step()
         opt_param_scheduler.step()
```

### Slime Async Variant (`train_async.py`)

Overlaps generation N+1 with training N:

```python
# train_async.py (simplified)
rollout_data_next = rollout_manager.generate.remote(start_rollout_id)  # kick off first
for rollout_id in range(start_rollout_id, num_rollout):
    rollout_data_curr = ray.get(rollout_data_next)        # wait for current
    rollout_data_next = rollout_manager.generate.remote(rollout_id + 1)  # start next
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr))      # train
    if (rollout_id + 1) % update_weights_interval == 0:
        ray.get(rollout_data_next)                        # must sync before weight update
        rollout_data_next = None
        actor_model.update_weights()
```

### Forge GRPO: Two Decoupled Async Loops

Forge's GRPO training loop (`apps/grpo/main.py:37-368`) uses a fundamentally different architecture: **two independent asyncio coroutines** connected by a replay buffer.

```python
# apps/grpo/main.py (simplified)
async def continuous_rollouts():
    while not shutdown_event.is_set():
        sample = await dataloader.sample.call_one()
        responses = await generator.generate.route(prompt)
        for response in responses:
            episode.reward = await reward_actor.evaluate_response.route(...)
        ref_logprobs = await ref_model.forward.route(input_ids)
        advantages = await compute_advantages.compute.call_one(episodes)
        for episode in episodes:
            await replay_buffer.add.call_one(episode)

async def continuous_training():
    while training_step < max_steps:
        batch = await replay_buffer.sample.call_one()
        if batch is None:
            await asyncio.sleep(0.1); continue
        await trainer.train_step.call(inputs, targets)
        await trainer.push_weights.call(training_step)
        await generator.update_weights.fanout(training_step)
        await mlogger.flush.call_one(training_step)

# Run concurrently
rollout_tasks = [asyncio.create_task(continuous_rollouts()) for _ in range(N)]
training_task = asyncio.create_task(continuous_training())
await training_task
```

**Inside `TitanTrainer.train_step()`** (`actors/trainer/titan.py:161-202`):

```
train_step(inputs, targets):
  local_inputs = inputs[self.engine.dp_rank]     # select DP shard
  loss = forward_backward(local_inputs, local_targets):
    targets["target_ids"] = create_shifted_targets(tokens, loss_mask)
    logits = model_parts[0](**inputs)             # forward
    loss_output = self.loss(logits, **targets)    # compiled GRPO loss
    loss.backward()                               # backward
  self.engine.optimizers.step()
  self.engine.lr_schedulers.step()
  self.engine.optimizers.zero_grad()
  self.engine.checkpointer.save(...)
```

### Forge SFT: Single Actor, Self-Contained Loop

Forge SFT (`apps/sft/main.py:419-458`) runs the entire training loop inside a single Monarch actor endpoint:

```python
# apps/sft/main.py (simplified)
@endpoint
async def train(self) -> None:
    dataloader = iter(self.train_dataloader)
    self.optimizers.zero_grad()
    while self.current_step < self.num_training_steps:
        batch = next(dataloader)
        self.train_step(batch)           # forward -> loss -> backward -> optim.step
        self.current_step += 1
        if self.current_step % self.eval_every_n_steps == 0:
            await self.evaluate()
        self.checkpointer.save(...)
        await self.mlogger.flush.call_one(global_step=self.current_step)
```

### Side-by-Side Comparison

| Aspect | Slime | Forge GRPO | Forge SFT |
|--------|-------|------------|-----------|
| **Loop driver** | Centralized Python script (`train.py`) | Async controller (`apps/grpo/main.py`) | Single actor endpoint (`train()`) |
| **Iteration unit** | Rollout ID (fixed count) | Training step (from replay buffer) | Training step (from dataloader) |
| **Loop structure** | `for rollout_id in range(N)` | Two concurrent `while` loops | `while step < max_steps` |
| **Gen/Train coupling** | Tightly coupled: generate then train | Decoupled via ReplayBuffer | No generation (pure SFT) |
| **Async overlap** | Sync (default) or 1-step-ahead async | Fully async (gen and train run independently) | N/A |
| **Forward pass** | Megatron `forward_backward_func` or manual `.forward()` | `model_parts[0](**inputs)` | `model_parts[0](inputs)` |
| **Backward pass** | Megatron pipeline schedule or `loss.backward()` | `loss.backward()` | `loss.backward()` |
| **Optimizer step** | At grad accumulation boundary (configurable) | Every `train_step` call | Every `train_step` call |
| **Grad accumulation** | Yes -- configurable micro-batches with `grad_accum` list | Not in GRPO path (single fwd/bwd per step) | Planned but not yet active (TODO in code) |
| **Steps per rollout** | Multiple (`num_steps_per_rollout = samples / gbs`) | 1 step per replay buffer sample | 1 step per dataloader batch |
| **Weight sync trigger** | Explicit `actor_model.update_weights()` after training | `push_weights` -> TorchStore -> `generator.update_weights` | Not needed |
| **Checkpointing** | `should_run_periodic_action()` at interval | Inside `train_step` via TorchTitan checkpointer | Inside loop via TorchTitan checkpointer |
| **Eval integration** | Separate `rollout_manager.eval.remote()` at intervals | Not in GRPO main loop (separate concern) | `self.evaluate()` at intervals |
| **Logging** | `logging_utils.log()` with manual metric dicts | `record_metric()` + metric logger flush | `record_metric()` + metric logger flush |
| **GPU memory mgmt** | Offload train/rollout models to CPU between phases | Not needed (separate GPU allocations) | Not needed (single model) |
| **Critic support** | Yes -- parallel `critic_model.async_train()` | No separate critic | No critic |
| **Concurrency model** | Ray actors + `ray.get()` blocking calls | asyncio tasks + Monarch `await` calls | Single actor, sequential |

### Data Flow Comparison

**Slime (per rollout iteration):**
```
DataSource.get_samples()
  -> SGLang HTTP generate (async tasks with semaphore)
  -> Reward model scoring
  -> _convert_samples_to_train_data() -> {tokens, rewards, loss_masks, rollout_log_probs}
  -> _split_train_data_by_dp() -> ray.put() per DP rank
  -> [each TrainRayActor] process_rollout_data() -> ray.get()
  -> compute_advantages_and_returns()
  -> pack into micro-batches
  -> forward/backward/step (multiple steps if data > global_batch_size)
```

**Forge GRPO (continuous):**
```
DatasetActor.sample()
  -> Generator.generate.route(prompt) -> [Completion objects]
  -> RewardActor.evaluate_response.route() -> reward per episode
  -> ReferenceModel.forward.route(input_ids) -> ref_logprobs
  -> ComputeAdvantages.compute() -> advantages
  -> ReplayBuffer.add() -> buffer
  ---decoupled---
  -> ReplayBuffer.sample() -> (inputs, targets) already collated
  -> TitanTrainer.train_step() -> forward/backward/step (single step)
  -> push_weights -> TorchStore -> generator.update_weights
```

### Architectural Trade-offs

**Slime's centralized driver** is easier to reason about sequentially -- the `for rollout_id` loop makes the global ordering explicit. But it forces synchronization barriers (generation must fully complete before training starts in sync mode), and all scheduling logic lives in the driver script. The async variant partially mitigates this but still requires explicit sync before weight updates.

**Forge's decoupled loops** allow generation and training to run at their natural speeds -- the replay buffer absorbs differences in throughput. This means generation doesn't block on training and vice versa. Multiple rollout threads can feed the buffer concurrently (`num_rollout_threads` config). However, this introduces replay buffer staleness: the training step may process episodes generated with an older policy version, making it inherently off-policy. Slime's async mode has the same off-policy issue but bounds it to at most 1 rollout of staleness.

**GPU memory:** Slime's approach of sharing GPUs between actor/ref/rollout (with CPU offloading) means fewer total GPUs are needed but adds offload overhead and complexity. Forge's approach of giving each role dedicated GPUs avoids offloading entirely but requires more GPUs.

**Gradient accumulation:** Slime supports configurable gradient accumulation with dynamic micro-batch sizing (`max_tokens_per_gpu`), enabling efficient training with variable-length sequences. Forge's GRPO path currently does a single forward/backward per optimizer step -- gradient accumulation support exists in the `Trainer` protocol API (`forward_backward` + `optim_step` separation) but is not used in the GRPO app.
