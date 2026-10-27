import math

import pytest

from common.schema import INVALID, LIMITS, decode_registers, encode_registers

VALUES = {"temperature_c": 63.4, "vibration_mm_s": 2.345, "current_a": 12.34, "pressure_bar": 5.67, "rpm": 2891.0}


def test_roundtrip_within_quantisation():
    ts, vals, running = decode_registers(encode_registers(1_767_225_600, VALUES))
    assert ts == 1_767_225_600 and running
    assert vals["temperature_c"] == pytest.approx(63.4, abs=0.05)
    assert vals["vibration_mm_s"] == pytest.approx(2.345, abs=0.0005)
    assert vals["current_a"] == pytest.approx(12.34, abs=0.005)
    assert vals["pressure_bar"] == pytest.approx(5.67, abs=0.005)
    assert vals["rpm"] == 2891.0


def test_registers_are_uint16():
    regs = encode_registers(1_767_225_600, VALUES)
    assert all(0 <= r <= 0xFFFF for r in regs)


@pytest.mark.parametrize("missing", [None, math.nan])
def test_missing_becomes_sentinel_and_back(missing):
    regs = encode_registers(1_767_225_600, {**VALUES, "pressure_bar": missing})
    assert INVALID in regs
    assert decode_registers(regs)[1]["pressure_bar"] is None


def test_overflow_is_clamped_not_sentinel():
    regs = encode_registers(1_767_225_600, {**VALUES, "current_a": 1e6})
    assert decode_registers(regs)[1]["current_a"] == pytest.approx(655.34)
    assert decode_registers(regs)[1]["current_a"] > LIMITS["current_a"][1]  # downstream validation will null it
