from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PAD = 0
BOS_TOKEN_ID = 9999
VALUE_BASE = 5000
DEFAULT_VALUE_MOD = 23
DEFAULT_MAX_DIFFICULTY = 10
DEFAULT_OPS = "asm"
OP_TO_IDS = {
    "+": (5, 1),
    "-": (6, 2),
    "*": (7, 3),
    "/": (8, 4),
}
OP_ID_TO_SYMBOL = {1: "+", 5: "+", 2: "-", 6: "-", 3: "*", 7: "*", 4: "/", 8: "/"}


@dataclass(frozen=True)
class ManoSample:
    tokens: list[int]
    difficulty: int
    answer: int
    seed: int


def vocab_size(value_mod: int = DEFAULT_VALUE_MOD, max_difficulty: int = DEFAULT_MAX_DIFFICULTY) -> int:
    del max_difficulty
    return max(BOS_TOKEN_ID + 1, VALUE_BASE + int(value_mod))


def length_weights(
    max_difficulty: int = DEFAULT_MAX_DIFFICULTY,
    *,
    mode: str = "uniform",
    alpha: float = 1.5,
    explicit: dict[int | str, float] | None = None,
) -> dict[int, float]:
    if explicit:
        raw = {int(k): float(v) for k, v in explicit.items()}
    elif mode == "powerlaw":
        raw = {l: float(l) ** (-float(alpha)) for l in range(1, int(max_difficulty) + 1)}
    elif mode == "reverse_powerlaw":
        raw = {l: float(int(max_difficulty) + 1 - l) ** (-float(alpha)) for l in range(1, int(max_difficulty) + 1)}
    elif mode == "uniform":
        raw = {l: 1.0 for l in range(1, int(max_difficulty) + 1)}
    else:
        raise ValueError(f"Unsupported length distribution mode: {mode}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Length weights must sum to a positive value.")
    return {k: v / total for k, v in sorted(raw.items())}


def _ops_from_string(ops: str) -> list[str]:
    out: list[str] = []
    if "a" in ops:
        out.append("+")
    if "s" in ops:
        out.append("-")
    if "m" in ops:
        out.append("*")
    if "d" in ops:
        out.append("/")
    if not out:
        raise ValueError(f"No supported ops in {ops!r}; use letters from a/s/m/d.")
    return out


def _gen_expr(rng: random.Random, n_ops: int, value_mod: int, ops: list[str], knowledge_augment: bool) -> tuple[list[int], int | None]:
    if n_ops == 0:
        value = rng.randint(0, int(value_mod) - 1)
        return [VALUE_BASE + value], value

    op = rng.choice(ops)
    left_ops = rng.randint(0, n_ops - 1)
    right_ops = n_ops - 1 - left_ops
    left_tokens, left_value = _gen_expr(rng, left_ops, value_mod, ops, knowledge_augment)
    right_tokens, right_value = _gen_expr(rng, right_ops, value_mod, ops, knowledge_augment)
    op_ids = OP_TO_IDS[op] if knowledge_augment else (OP_TO_IDS[op][-1],)
    op_id = rng.choice(list(op_ids))

    if left_value is None or right_value is None:
        answer = None
    elif op == "+":
        answer = (left_value + right_value) % value_mod
    elif op == "-":
        answer = (left_value - right_value + value_mod) % value_mod
    elif op == "*":
        answer = (left_value * right_value) % value_mod
    elif right_value == 0:
        answer = None
    else:
        answer = (left_value * pow(right_value, -1, value_mod)) % value_mod
    return [op_id] + left_tokens + right_tokens, answer


def generate_sample(
    seed: int,
    weights: dict[int, float],
    *,
    value_mod: int = DEFAULT_VALUE_MOD,
    ops: str = DEFAULT_OPS,
    knowledge_augment: bool = True,
    noise_ratio: float = 0.0,
    noise_weights: dict[int, float] | None = None,
) -> ManoSample:
    rng = random.Random(int(seed))
    is_noise = float(noise_ratio) > 0.0 and rng.random() < float(noise_ratio)
    length_dist = noise_weights if is_noise and noise_weights is not None else weights
    difficulties = list(length_dist.keys())
    probabilities = list(length_dist.values())
    difficulty = int(rng.choices(difficulties, weights=probabilities, k=1)[0])
    op_choices = _ops_from_string(ops)

    while True:
        # PhysicsLM's original Mano generator samples an unused `this_people`
        # array before each expression attempt. Keep the random-number stream
        # identical so a seed maps to the exact same token sequence.
        for _ in range(difficulty + 1):
            rng.randint(0, int(value_mod) - 1)
        question, answer = _gen_expr(rng, difficulty, int(value_mod), op_choices, bool(knowledge_augment))
        if answer is not None:
            break

    bos = BOS_TOKEN_ID - (rng.randint(0, 1) if knowledge_augment else 0)
    length_token = difficulty * 10 + (rng.randint(0, 1) if knowledge_augment else 0)
    answer_prefix = difficulty * 10 + 4 + (rng.randint(0, 1) if knowledge_augment else 0)
    if is_noise:
        answer = rng.randint(0, int(value_mod) - 1)
    tokens = [bos, length_token] + question + [answer_prefix, VALUE_BASE + int(answer)]
    return ManoSample(tokens=tokens, difficulty=difficulty, answer=int(answer), seed=int(seed))


def iter_seeds(
    *,
    length: int,
    base_seed: int = 0,
    repeat: int = 1,
    mode: str = "incremental",
    shuffle_seed: int = 42,
    reverse: bool = False,
) -> Iterator[int]:
    length = int(length)
    repeat = int(repeat)
    if length < 0 or repeat < 1:
        raise ValueError("length must be >=0 and repeat must be >=1.")
    if mode == "incremental":
        for offset in range(length):
            yield int(base_seed - offset if reverse else base_seed + offset)
        return
    if mode not in {"epoch_ordered", "epoch_shuffled"}:
        raise ValueError(f"Unsupported seed mode: {mode}")
    epoch = list(range(length))
    if mode == "epoch_shuffled":
        rng = random.Random(int(shuffle_seed))
        rng.shuffle(epoch)
    for _ in range(repeat):
        for offset in epoch:
            yield int(base_seed - offset if reverse else base_seed + offset)


def parse_jsonl_line(line: str, *, fallback_seed: int = 0) -> ManoSample | None:
    if not line.strip():
        return None
    row: dict[str, Any] = json.loads(line)
    tokens = [int(x) for x in row["text"]]
    if len(tokens) < 2:
        return None
    difficulty = int(row.get("difficulty", tokens[1] // 10))
    answer_token = int(tokens[-1])
    answer = int(row.get("answer", answer_token - VALUE_BASE))
    seed = int(row.get("seed", fallback_seed))
    return ManoSample(tokens=tokens, difficulty=difficulty, answer=answer, seed=seed)


def resolve_jsonl_files(path: str | Path) -> list[Path]:
    p = Path(path).resolve()
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p}")
    files = sorted(x for x in p.rglob("*.jsonl") if x.is_file())
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {p}")
    return files


def tokens_to_expression(tokens: list[int], value_mod: int = DEFAULT_VALUE_MOD, pos: int = 0) -> tuple[str, int]:
    if pos >= len(tokens):
        return "?", 0
    token = int(tokens[pos])
    if VALUE_BASE <= token < VALUE_BASE + int(value_mod):
        return str(token - VALUE_BASE), 1
    if token in OP_ID_TO_SYMBOL:
        op = OP_ID_TO_SYMBOL[token]
        left, n_left = tokens_to_expression(tokens, value_mod, pos + 1)
        right, n_right = tokens_to_expression(tokens, value_mod, pos + 1 + n_left)
        return f"({left} {op} {right})", 1 + n_left + n_right
    return "?", 1
