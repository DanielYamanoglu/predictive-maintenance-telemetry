"""Modbus TCP server exposing the simulated plant as PLC-style holding registers."""
from __future__ import annotations

import asyncio

from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import ModbusTcpServer

from common.schema import REGISTERS_PER_MACHINE, encode_registers
from simulator.plant import DT_S, Machine


def machine_reading(m: Machine, i: int) -> tuple[int, dict]:
    row = m.data.iloc[i % len(m)]
    return int(m.data.index[i % len(m)].timestamp()), {k: row[k] for k in m.data.columns if k != "fault_active"}


async def serve_modbus(machines: list[Machine], host: str = "0.0.0.0", port: int = 5020, speed: float = 60.0,
                       stop: asyncio.Event | None = None) -> None:
    """Serve registers; machine k occupies addresses [k*8, k*8+8). One step = DT_S simulated seconds,
    advanced every DT_S/speed wall seconds."""
    # pymodbus looks datablock cells up at (wire address + 1), hence the base of 1 for wire address 0
    block = ModbusSequentialDataBlock(1, [0] * (REGISTERS_PER_MACHINE * len(machines)))
    server = ModbusTcpServer(ModbusServerContext(devices=ModbusDeviceContext(hr=block), single=True), address=(host, port))
    task = asyncio.create_task(server.serve_forever())
    stop = stop or asyncio.Event()
    step = 0
    try:
        while not stop.is_set():
            for k, m in enumerate(machines):
                ts, vals = machine_reading(m, step)
                block.setValues(1 + k * REGISTERS_PER_MACHINE, encode_registers(ts, vals))
            step += 1
            await asyncio.sleep(DT_S / speed)
    finally:
        await server.shutdown()
        task.cancel()
