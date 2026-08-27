# Copyright (c) Tianjin University, Ltd. 2026. All rights reserved.
"""solve_tri 的 ATK 泛化用例生成器。

<<<<<<< HEAD
生成 400 个泛化用例：200 个 chunk_size=64 (FP32 路径) + 200 个 chunk_size=128 (Cube 路径)。
=======
生成 1000 个 ascend950 精度用例，覆盖：
- layout: bsnd, bnsd, tnd, ntd
- dtype: bf16, fp16
- chunk_size: 16, 32, 64, 128
- 定长/变长、对齐/尾块

shape 约束：不要太小（至少数个 chunk、H>=4），也不要太大（元素数上限），
避免 CPU 双标杆 / 多进程 golden 把内存打爆。
>>>>>>> b83e57e9 (fix)
"""

import json
import random
from copy import deepcopy

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

OP_NAME = "solve_tri"
<<<<<<< HEAD


def _generate_profiles(seed: int = 20260817):
    """生成 400 个泛化用例配置。
    
    - 前 200 个: chunk_size=64，走 FP32 路径
    - 后 200 个: chunk_size=128，走 Cube 低精度路径
    
    覆盖：
    - layout: bsnd, tnd
    - dtype: bf16, fp16
    - B: 1-16
    - H: 1-32 (受硬件约束 H * chunk_size * 16 + 16 <= 65536)
    - T: 64-4096
    - num_seqs: 1-8 (仅 tnd 模式)
    """
    random.seed(seed)
    profiles = []
    
    layouts = ["bsnd", "tnd"]
    dtypes = ["bf16", "fp16"]
    B_values = [1, 2, 4, 8, 16]
    H_values = [1, 2, 4, 8, 16, 32]
    T_values = [64, 128, 256, 512, 1024, 2048, 4096]
    num_seqs_values = [1, 2, 4, 8]
    
    for chunk_size in [64, 128]:
        for _ in range(200):
            layout = random.choice(layouts)
            dtype = random.choice(dtypes)
            H = random.choice(H_values)
            
            # 硬件约束: H * chunk_size * 16 + 16 <= 65536
            while H * chunk_size * 16 + 16 > 65536:
                H = random.choice(H_values)
            
            if layout == "bsnd":
                B = random.choice(B_values)
                T = random.choice(T_values)
                if T < chunk_size:
                    T = chunk_size
                num_seqs = 1
                name = f"{dtype}_{layout}_B{B}_H{H}_T{T}_C{chunk_size}"
            else:
                num_seqs = random.choice(num_seqs_values)
                min_seq_len = chunk_size
                max_seq_len = random.choice([256, 512, 1024, 2048])
                T = sum(random.randint(min_seq_len, max_seq_len) for _ in range(num_seqs))
                B = 1
                name = f"{dtype}_{layout}_NS{num_seqs}_H{H}_T{T}_C{chunk_size}"
            
            profile = {
                "name": name,
                "dtype": dtype,
                "B": B,
                "H": H,
                "T": T,
                "chunk_size": chunk_size,
                "layout": layout,
                "num_seqs": num_seqs,
            }
            profiles.append(profile)
    
    # 追加边界 case
    boundary_cases = [
        # chunk_size=64 边界: H=32 (32*64*16+16=32784, 合法)
        {"dtype": "fp16", "B": 1, "H": 32, "T": 256, "chunk_size": 64, "layout": "bsnd", "num_seqs": 1,
         "name": "fp16_bsnd_boundary_H32_C64_T256"},
        {"dtype": "bf16", "B": 1, "H": 32, "T": 512, "chunk_size": 64, "layout": "tnd", "num_seqs": 2,
         "name": "bf16_tnd_boundary_H32_C64_NS2_T512"},
        # chunk_size=128 边界: H=16 (16*128*16+16=32784, 合法最大 H)
        {"dtype": "fp16", "B": 1, "H": 16, "T": 512, "chunk_size": 128, "layout": "bsnd", "num_seqs": 1,
         "name": "fp16_bsnd_boundary_H16_C128_T512"},
        {"dtype": "bf16", "B": 1, "H": 16, "T": 1024, "chunk_size": 128, "layout": "tnd", "num_seqs": 4,
         "name": "bf16_tnd_boundary_H16_C128_NS4_T1024"},
    ]
    profiles.extend(boundary_cases)
    
    return profiles

PROFILES = _generate_profiles(seed=20260817)
=======
SOC = "ascend950"
TARGET_COUNT = 1000

# fp16/bf16 输入约 2B/elem；CPU fp32 golden 约 4B/elem；ATK 双标杆再乘几份。
# 4M elem ≈ 8MB fp16 / 16MB fp32，4 worker 同时跑仍远低于爆内存。
MIN_NUMEL = 48 * 1024
MAX_NUMEL = 4 * 1024 * 1024


def _hw_ok(H: int, chunk_size: int) -> bool:
    """910b Cube 路径约束在 950 上也作为安全上限保留。"""
    return H * chunk_size * 16 + 16 <= 65536


def _numel(layout, B, H, T, chunk_size) -> int:
    if layout in ("tnd", "ntd"):
        return int(T) * int(H) * int(chunk_size)
    return int(B) * int(T) * int(H) * int(chunk_size)


def _min_tokens(layout, chunk_size, num_seqs) -> int:
    """至少 4 个 chunk，变长再按序列数放大，避免单 chunk 小 case。"""
    per_seq = max(chunk_size * 4, 128)
    if layout in ("tnd", "ntd"):
        return per_seq * max(int(num_seqs), 1)
    return per_seq


def _clamp_T(layout, B, H, T, chunk_size, num_seqs):
    """把 T 收到 [MIN_NUMEL, MAX_NUMEL] 之间；夹不住则返回 None。"""
    T = max(int(T), _min_tokens(layout, chunk_size, num_seqs))
    step = chunk_size
    n = _numel(layout, B, H, T, chunk_size)
    while n > MAX_NUMEL and T - step >= _min_tokens(layout, chunk_size, num_seqs):
        T -= step
        n = _numel(layout, B, H, T, chunk_size)
    while n < MIN_NUMEL:
        T += step
        n = _numel(layout, B, H, T, chunk_size)
        if n > MAX_NUMEL:
            return None
    if n < MIN_NUMEL or n > MAX_NUMEL:
        return None
    if not _hw_ok(H, chunk_size):
        return None
    return T


def _make_profile(layout, dtype, chunk_size, B, H, T, num_seqs, name=None):
    if layout in ("tnd", "ntd"):
        B = 1
        num_seqs = max(2, int(num_seqs))
    else:
        num_seqs = 1
        B = max(int(B), 2)
    H = max(int(H), 4)
    T = _clamp_T(layout, B, H, T, chunk_size, num_seqs)
    if T is None:
        return None
    if name is None:
        if layout in ("tnd", "ntd"):
            name = f"{dtype}_{layout}_NS{num_seqs}_H{H}_T{T}_C{chunk_size}"
        else:
            name = f"{dtype}_{layout}_B{B}_H{H}_T{T}_C{chunk_size}"
    return {
        "name": name,
        "dtype": dtype,
        "B": B,
        "H": H,
        "T": T,
        "chunk_size": chunk_size,
        "layout": layout,
        "num_seqs": num_seqs,
    }


def _generate_profiles(seed: int = 20260826):
    random.seed(seed)
    layouts = ["bsnd", "bnsd", "tnd", "ntd"]
    dtypes = ["bf16", "fp16"]
    chunk_sizes = [16, 32, 64, 128]
    B_values = [2, 4, 8]
    H_values = [4, 8, 16]
    num_seqs_values = [2, 3, 4]
    profiles = []
    seen = set()

    def add(profile):
        if profile is None:
            return False
        key = (
            profile["dtype"],
            profile["layout"],
            profile["B"],
            profile["H"],
            profile["T"],
            profile["chunk_size"],
            profile["num_seqs"],
        )
        if key in seen:
            return False
        n = _numel(profile["layout"], profile["B"], profile["H"],
                    profile["T"], profile["chunk_size"])
        if n < MIN_NUMEL or n > MAX_NUMEL:
            return False
        if not _hw_ok(profile["H"], profile["chunk_size"]):
            return False
        seen.add(key)
        profiles.append(profile)
        return True

    # 覆盖网格：layout × dtype × chunk × (对齐 / 尾块)，中等 B/H、至少 8 个 chunk
    for layout in layouts:
        for dtype in dtypes:
            for chunk_size in chunk_sizes:
                for tail in (False, True):
                    H = 8
                    B = 4 if layout in ("bsnd", "bnsd") else 1
                    T = chunk_size * 8
                    if tail:
                        T += max(chunk_size // 2, 7)
                    num_seqs = 2 if layout in ("tnd", "ntd") else 1
                    add(_make_profile(layout, dtype, chunk_size, B, H, T, num_seqs))

    # 再铺一层更常见的中等 shape（H=4/16、B=2/8、变长 4 段）
    for layout in layouts:
        for dtype in dtypes:
            for chunk_size in chunk_sizes:
                for H in (4, 16):
                    for B in ((2, 8) if layout in ("bsnd", "bnsd") else (1,)):
                        T = chunk_size * 6 + (chunk_size // 4 if H == 16 else 0)
                        num_seqs = 4 if layout in ("tnd", "ntd") else 1
                        add(_make_profile(layout, dtype, chunk_size, B, H, T, num_seqs))

    # 补齐到 1000：随机中等 shape / 尾块 / 序列数
    attempts = 0
    while len(profiles) < TARGET_COUNT and attempts < 200000:
        attempts += 1
        layout = random.choice(layouts)
        dtype = random.choice(dtypes)
        chunk_size = random.choice(chunk_sizes)
        H = random.choice(H_values)
        if not _hw_ok(H, chunk_size):
            continue
        if layout in ("bsnd", "bnsd"):
            B = random.choice(B_values)
            T_pool = [
                chunk_size * 4,
                chunk_size * 4 + 7,
                chunk_size * 6,
                chunk_size * 6 + chunk_size // 2,
                chunk_size * 8,
                chunk_size * 8 + 13,
                chunk_size * 12,
                192,
                256,
                384,
                512,
                640,
                768,
            ]
            T = random.choice([t for t in T_pool if t >= chunk_size * 4])
            num_seqs = 1
        else:
            B = 1
            num_seqs = random.choice(num_seqs_values)
            max_seq = random.choice([
                chunk_size * 4,
                chunk_size * 6,
                chunk_size * 8,
                256,
                384,
                512,
            ])
            max_seq = max(max_seq, chunk_size * 4)
            T = sum(random.randint(chunk_size * 4, max_seq) for _ in range(num_seqs))
        add(_make_profile(layout, dtype, chunk_size, B, H, T, num_seqs))

    if len(profiles) < TARGET_COUNT:
        raise RuntimeError(f"只生成了 {len(profiles)} 条用例，不足 {TARGET_COUNT}")
    return profiles[:TARGET_COUNT]


PROFILES = _generate_profiles(seed=20260826)
>>>>>>> b83e57e9 (fix)


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({
        "op": OP_NAME,
        "case_id": index,
<<<<<<< HEAD
        "seed": 20260817 + index,
        "route": "ascendc",
        "soc": "ascend910b"
=======
        "seed": 20260826 + index,
        "route": "ascendc",
        "soc": SOC,
>>>>>>> b83e57e9 (fix)
    })
    return profile


@GENERATOR_REGISTRY.register("generator_solve_tri")
class SolveTriGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        index = max(int(self.index) - 1, 0)
        spec = _spec(index)
        case_config.id = index
        case_config.default_seed = spec["seed"]
        case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
        for item in case_config.inputs:
            cfg = item[0] if isinstance(item, list) else item
            if cfg.name == "low_precision_marker":
                cfg.dtype = _dtype(spec.get("dtype", "bf16"))
            elif cfg.name == "case_spec":
                cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
            elif cfg.name in spec:
                cfg.range_values = spec[cfg.name]
        return case_config
