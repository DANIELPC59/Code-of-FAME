"""Burst-level FRONT-inspired defense.

The original FRONT defense schedules dummy packets on a timestamp axis. This
module adapts the idea to burst sequences by treating the effective burst index
as a normalized pseudo-time axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

@dataclass(frozen=True)
class BurstFrontMetadata:
    """Per-trace accounting returned when ``return_metadata=True``."""

    effective_burst_len: np.ndarray
    sampled_client: np.ndarray
    sampled_server: np.ndarray
    inserted_client: np.ndarray
    inserted_server: np.ndarray
    dropped_client: np.ndarray
    dropped_server: np.ndarray
    original_packets: np.ndarray
    defended_packets: np.ndarray
    overhead: np.ndarray


class BurstFRONT:
    """FRONT-inspired padding for signed burst sequences.

    Parameters
    ----------
    Nc, Ns:
        Maximum dummy packet budgets for client/outgoing and server/incoming
        directions. For each trace, actual budgets are sampled from U(1, N).
        If a budget is 0, that side inserts no packets.
    Wmin, Wmax:
        Rayleigh scale range on the normalized burst-index axis. The effective
        burst sequence occupies [0, 1). Larger values spread dummy packets
        farther from the front and increase the chance of samples beyond 1.0
        being dropped.
    max_burst_len:
        Fixed output burst length. The default matches input shape [B, 2000, 1].
    client_direction, server_direction:
        Signs used for outgoing and incoming dummy packets.
    seed:
        Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        Nc: int = 500,
        Ns: int = 500,
        Wmin: float = 0.05,
        Wmax: float = 0.20,
        max_burst_len: int = 2000,
        client_direction: int = 1,
        server_direction: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        self.Nc = self._validate_budget(Nc, "Nc")
        self.Ns = self._validate_budget(Ns, "Ns")
        self.Wmin = float(Wmin)
        self.Wmax = float(Wmax)
        self.max_burst_len = self._validate_budget(max_burst_len, "max_burst_len")
        self.client_direction = self._validate_direction(client_direction, "client_direction")
        self.server_direction = self._validate_direction(server_direction, "server_direction")
        self.rng = np.random.default_rng(seed)

        if self.Wmin <= 0:
            raise ValueError("Wmin must be positive.")
        if self.Wmax < self.Wmin:
            raise ValueError("Wmax must be greater than or equal to Wmin.")

    @staticmethod
    def _validate_budget(value: int, name: str) -> int:
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    @staticmethod
    def _validate_direction(value: int, name: str) -> int:
        value = int(np.sign(value))
        if value == 0:
            raise ValueError(f"{name} must be non-zero.")
        return value

    def defend(
        self,
        data: np.ndarray,
        return_metadata: bool = False,
        show_progress: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, BurstFrontMetadata]]:
        """Apply burst-level FRONT padding.

        Parameters
        ----------
        data:
            Signed burst data with shape ``[batch_size, 2000, 1]`` or
            ``[batch_size, 2000]``. Positive bursts represent outgoing/client
            packets, negative bursts represent incoming/server packets, and 0
            marks padding after the effective trace.
        return_metadata:
            If true, return ``(defended_data, metadata)``.
        show_progress:
            If true, show a tqdm progress bar over traces in the batch.

        Returns
        -------
        np.ndarray
            Defended burst data with shape ``[batch_size, max_burst_len, 1]``.
        """

        x, had_channel = self._prepare_input(data)
        batch_size = x.shape[0]
        defended = np.zeros((batch_size, self.max_burst_len), dtype=x.dtype)

        effective_lengths = np.zeros(batch_size, dtype=np.int32)
        sampled_client = np.zeros(batch_size, dtype=np.int32)
        sampled_server = np.zeros(batch_size, dtype=np.int32)
        inserted_client = np.zeros(batch_size, dtype=np.int32)
        inserted_server = np.zeros(batch_size, dtype=np.int32)
        dropped_client = np.zeros(batch_size, dtype=np.int32)
        dropped_server = np.zeros(batch_size, dtype=np.int32)
        original_packets = np.zeros(batch_size, dtype=np.int64)
        defended_packets = np.zeros(batch_size, dtype=np.int64)
        iterable = x
        if show_progress:
            iterable = tqdm(x, desc="Applying Burst-FRONT", unit="trace")
        for i, trace in enumerate(iterable):
            bursts = self._effective_bursts(trace)
            effective_lengths[i] = len(bursts)
            original_packets[i] = int(np.sum(np.abs(bursts), dtype=np.int64))

            defended_bursts, stats = self._defend_one(bursts)
            output_len = min(len(defended_bursts), self.max_burst_len)
            if output_len:
                defended[i, :output_len] = np.asarray(defended_bursts[:output_len], dtype=x.dtype)

            sampled_client[i] = stats["sampled_client"]
            sampled_server[i] = stats["sampled_server"]
            inserted_client[i] = stats["inserted_client"]
            inserted_server[i] = stats["inserted_server"]
            dropped_client[i] = stats["dropped_client"]
            dropped_server[i] = stats["dropped_server"]
            defended_packets[i] = int(np.sum(np.abs(defended[i]), dtype=np.int64))

        defended = defended[:, :, np.newaxis] if had_channel else defended
        if not return_metadata:
            return defended

        denom = np.maximum(original_packets, 1)
        metadata = BurstFrontMetadata(
            effective_burst_len=effective_lengths,   # effective burst length after stripping zero padding
            sampled_client=sampled_client,           # sampled dummy budget for client direction, drawn from U(1, Nc)
            sampled_server=sampled_server,           # sampled dummy budget for server direction, drawn from U(1, Ns)
            inserted_client=inserted_client,         # number of dummy packets actually inserted on the client side
            inserted_server=inserted_server,         # number of dummy packets actually inserted on the server side
            dropped_client=dropped_client,           # client dummy packets dropped (sampled position beyond trace length)
            dropped_server=dropped_server,           # server dummy packets dropped (sampled position beyond trace length)
            original_packets=original_packets,       # total packet count in the original trace
            defended_packets=defended_packets,       # total packet count after defense
            overhead=(defended_packets - original_packets) / denom,  # bandwidth overhead ratio
        )
        return defended, metadata

    def _prepare_input(self, data: np.ndarray) -> Tuple[np.ndarray, bool]:
        x = np.asarray(data)
        if x.ndim == 3:
            if x.shape[2] != 1:
                raise ValueError(f"Expected data.shape[2] == 1, got {x.shape}.")
            return x[:, :, 0], True
        if x.ndim == 2:
            return x, False
        raise ValueError(f"Expected data shape [B, L, 1] or [B, L], got {x.shape}.")

    @staticmethod
    def _effective_bursts(trace: np.ndarray) -> List[int]:
        zero_positions = np.flatnonzero(trace == 0)
        end = int(zero_positions[0]) if zero_positions.size else int(trace.shape[0])
        return [int(v) for v in trace[:end] if int(v) != 0]

    def _defend_one(self, bursts: List[int]) -> Tuple[List[int], Dict[str, int]]:
        if not bursts:
            return [], {
                "sampled_client": 0,
                "sampled_server": 0,
                "inserted_client": 0,
                "inserted_server": 0,
                "dropped_client": 0,
                "dropped_server": 0,
            }

        nc = self._sample_budget(self.Nc)
        ns = self._sample_budget(self.Ns)
        wc = self._sample_window()
        ws = self._sample_window()

        client_events, dropped_client = self._sample_events(
            count=nc,
            scale=wc,
            effective_len=len(bursts),
            direction=self.client_direction,
        )
        server_events, dropped_server = self._sample_events(
            count=ns,
            scale=ws,
            effective_len=len(bursts),
            direction=self.server_direction,
        )

        events = sorted(client_events + server_events, key=lambda item: (item[0], item[1]))
        defended = self._insert_events(bursts, events)

        stats = {
            "sampled_client": nc,
            "sampled_server": ns,
            "inserted_client": len(client_events),
            "inserted_server": len(server_events),
            "dropped_client": dropped_client,
            "dropped_server": dropped_server,
        }
        return defended, stats

    def _sample_budget(self, max_budget: int) -> int:
        if max_budget <= 0:
            return 0
        return int(self.rng.integers(1, max_budget + 1))

    def _sample_window(self) -> float:
        if self.Wmax == self.Wmin:
            return self.Wmin
        return float(self.rng.uniform(self.Wmin, self.Wmax))

    def _sample_events(
        self,
        count: int,
        scale: float,
        effective_len: int,
        direction: int,
    ) -> Tuple[List[Tuple[int, float, int]], int]:
        if count <= 0:
            return [], 0

        times = self.rng.rayleigh(scale=scale, size=count)
        keep_mask = times < 1.0
        kept_times = times[keep_mask]
        positions = np.floor(kept_times * effective_len).astype(np.int32)
        positions = np.clip(positions, 0, max(effective_len - 1, 0))

        events = [
            (int(pos), float(t), int(direction))
            for pos, t in zip(positions, kept_times)
        ]
        return events, int(count - len(events))

    def _insert_events(
        self,
        bursts: List[int],
        events: List[Tuple[int, float, int]],
    ) -> List[int]:
        by_position: Dict[int, List[int]] = {}
        for pos, _, direction in events:
            by_position.setdefault(pos, []).append(direction)

        result: List[int] = []
        for pos, burst in enumerate(bursts):
            for direction in by_position.get(pos, []):
                self._append_packet(result, direction)
            self._append_burst(result, burst)
        return result

    @staticmethod
    def _append_packet(result: List[int], direction: int) -> None:
        if result and np.sign(result[-1]) == direction:
            result[-1] += direction
        else:
            result.append(direction)

    @staticmethod
    def _append_burst(result: List[int], burst: int) -> None:
        if burst == 0:
            return
        direction = int(np.sign(burst))
        if result and int(np.sign(result[-1])) == direction:
            result[-1] += int(burst)
        else:
            result.append(int(burst))


def defend_burst_front(
    data: np.ndarray,
    Nc: int = 500,
    Ns: int = 500,
    Wmin: float = 0.05,
    Wmax: float = 0.20,
    max_burst_len: int = 2000,
    seed: Optional[int] = None,
    return_metadata: bool = False,
    show_progress: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, BurstFrontMetadata]]:
    """Convenience wrapper for one-shot burst FRONT defense."""

    defender = BurstFRONT(
        Nc=Nc,
        Ns=Ns,
        Wmin=Wmin,
        Wmax=Wmax,
        max_burst_len=max_burst_len,
        seed=seed,
    )
    return defender.defend(data, return_metadata=return_metadata, show_progress=show_progress)