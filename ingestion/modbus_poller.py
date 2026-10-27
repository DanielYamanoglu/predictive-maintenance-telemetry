"""Poll the PLC's holding registers over Modbus TCP."""
from __future__ import annotations

import asyncio
import logging

from pymodbus.client import AsyncModbusTcpClient

from common.schema import REGISTERS_PER_MACHINE
from ingestion.pipeline import Ingestor
from simulator.plant import layout

log = logging.getLogger("modbus")
MAX_REGS_PER_READ = 120  # Modbus limit is 125 registers per request


async def poll_modbus(ing: Ingestor, host: str, port: int, n_machines: int, interval_s: float,
                      duration_s: float | None = None) -> None:
    """Each cycle, read all machine blocks in as few contiguous range requests as the protocol allows."""
    machines = layout(n_machines)
    client = AsyncModbusTcpClient(host, port=port)
    while not await client.connect():  # the PLC / simulator may still be starting
        log.info("waiting for modbus %s:%s", host, port)
        await asyncio.sleep(2)
    loop = asyncio.get_running_loop()
    t_end = None if duration_s is None else loop.time() + duration_s
    try:
        while t_end is None or loop.time() < t_end:
            per_req = MAX_REGS_PER_READ // REGISTERS_PER_MACHINE  # machines per request
            for first in range(0, n_machines, per_req):
                chunk = machines[first:first + per_req]
                rr = await client.read_holding_registers(first * REGISTERS_PER_MACHINE,
                                                         count=len(chunk) * REGISTERS_PER_MACHINE)
                if rr.isError():
                    ing.stats["modbus_errors"] += 1
                    log.warning("modbus read error: %s", rr)
                    continue
                for k, (line, mid, _kind) in enumerate(chunk):
                    ing.from_registers(line, mid, rr.registers[k * REGISTERS_PER_MACHINE:(k + 1) * REGISTERS_PER_MACHINE])
            await asyncio.sleep(interval_s)
    finally:
        client.close()
