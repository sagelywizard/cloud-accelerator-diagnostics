# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines command line interface for `tpu-info` tool.

Top-level functions should be added to `project.scripts` in `pyproject.toml`.
"""

from absl import app
from absl import flags

from tpu_info import device
from tpu_info import metrics
import grpc
import rich.console
import rich.table

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Print helpful debugging info.')

def _bytes_to_gib(size: int) -> float:
  return size / (1 << 30)


def print_chip_info(debug: bool):
  """Print local TPU devices and libtpu runtime metrics."""

  # TODO(wcromar): Merge all of this info into one table
  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.")
    return

  console = rich.console.Console()

  table = rich.table.Table(title="TPU Chips", title_justify="left")
  table.add_column("Device")
  table.add_column("Type")
  table.add_column("Cores")
  # TODO(wcromar): this may not match the libtpu runtime metrics
  # table.add_column("HBM (per core)")
  table.add_column("PID")

  chip_paths = [device.chip_path(chip_type, index) for index in range(count)]
  chip_owners = device.get_chip_owners()

  for chip in chip_paths:
    owner = chip_owners.get(chip)

    table.add_row(
        chip,
        str(chip_type),
        str(chip_type.value.accelerators_per_chip),
        str(owner),
    )

  console.print(table)

  table = rich.table.Table(title="TPU Chip Utilization", title_justify="left")
  table.add_column("Core ID")
  table.add_column("Memory usage")
  table.add_column("Duty cycle", justify="right")

  try:
    chip_usage = metrics.get_chip_usage(chip_type)
  except grpc.RpcError as e:
    # TODO(wcromar): libtpu should start this server automatically
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      if debug:
        print(
            "Libtpu metrics unavailable. Did you start a workload with "
            "`TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?"
        )
      # TODO(wcromar): Point to public documentation once released
      return
    else:
      raise e

  if debug:
    # TODO(wcromar): take alternative ports as a flag
    print("Connected to libtpu at grpc://localhost:8431...")
  for chip in chip_usage:
    duty_cycle_str = f"{chip.duty_cycle_pct:.2f}%"
    for core in chip.core_usage:
      table.add_row(
          str(core.core_id),
          f"{_bytes_to_gib(core.memory_usage):.2f} GiB /"
          f" {_bytes_to_gib(core.total_memory):.2f} GiB",
          duty_cycle_str,
      )
      # Only print duty cycle on the first core. Why? Duty cycle is only
      # reported per-chip, so unfortunately we don't have the per-core data.
      duty_cycle_str = ""

  console.print(table)

def print_chip_info_with_flags(argv):
  """print_chip_info_with_flags passes flags to print_chip_info.

  Why separate print_chip_info_with_flags and print_chip_info? This is useful
  for testing. We can test the main logic of print_chip_info without requiring
  absl handle any flag parsing.
  """
  del argv
  print_chip_info(FLAGS.debug)

def run_absl_app():
  """run_absl_app is the main entrypoint for the tpu-info CLI."""
  app.run(print_chip_info_with_flags)