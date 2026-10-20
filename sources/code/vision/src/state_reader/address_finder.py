"""
Helper to find and validate memory addresses for Getting Over It camera Y.

This module provides utilities to:
1. Scan the memory address space to find candidate values
2. Validate found addresses against known game states
3. Save discovered addresses to the config file

Primary workflow (manual, outside this script):
1. Launch Getting Over It
2. Open Cheat Engine, attach to GettingOverIt.exe
3. Enable Mono → Dissect Mono
4. Find CameraController or GameCamera instance
5. Navigate to Transform → position → y
6. Use "Find out what accesses this address" to build pointer chain
7. Save offsets to config/memory_addresses.json
"""

import json
import os
import time
from typing import Optional


def save_offsets(
    offsets: list[int],
    config_path: Optional[str] = None,
    description: str = "",
):
    """Save discovered pointer offsets to the config file.

    Args:
        offsets: List of offsets forming a pointer chain.
                 [base_offset, ptr1_offset, ptr2_offset, ..., final_offset]
                 e.g. [0x1A2B3C, 0x10, 0x20, 0x4] means:
                 *(*(*(base + 0x1A2B3C) + 0x10) + 0x20) + 0x4
        config_path: Path to the config file.
        description: Optional description of how the address was found.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "memory_addresses.json",
        )

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    cfg = {
        "pointer_offsets": offsets,
        "description": description,
        "game_version": "1.7",  # from save data
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Found using Cheat Engine Mono Dissect on Assembly-CSharp.dll",
    }

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[address_finder] Saved offsets to {config_path}")


# ---- Cheat Engine Guide ----
# This is included as a docstring for reference.

CHEAT_ENGINE_GUIDE = """
=== Finding Camera Y Address with Cheat Engine ===

Step 1: Launch and Attach
  - Launch Getting Over It (windowed mode recommended)
  - Start a game so the camera is moving (climb a bit)
  - Open Cheat Engine, click the computer icon, select GettingOverIt.exe

Step 2: Enable Mono Features
  - Menu: Mono → Activate Mono Features
  - Menu: Mono → Dissect Mono

Step 3: Find Camera Controller
  - In the Mono Dissect window, expand Assembly-CSharp.dll
  - Look for one of these classes:
    * CameraController
    * GameCamera
    * GameCameraPosition
    * ProgressMeter
  - Click on the class to see its instances (if any)
  - Select an instance to see its fields

Step 4: Navigate to Camera Y
  - In the instance fields, find the Transform reference
  - Expand Transform → position → y
  - You should see a float value that matches the camera height

  Alternative: Find the hingePos directly
  - Look for a class that stores the hinge position (around -105,000)
  - This is the game's primary progress metric

Step 5: Get the Pointer Chain
  - Right-click the y value → "Find out what accesses this address"
  - In the game, move the camera (climb or fall)
  - Look for instructions that read the address
  - Click "Show disassembler" → "More Information"
  - Note the base address (should be mono-2.0-bdwgc.dll + offset)
  - Continue following pointer paths until you have a static chain

Step 6: Save Offsets
  - Use the save_offsets() function from this module
  - Example:
    from src.state_reader.address_finder import save_offsets
    save_offsets([0x1A2B3C, 0x10, 0x8, 0x4], description="CameraController.transform.position.y")

Step 7: Verify
  - Run scripts/test_memory.py to verify the reads match in-game height
  - Move up and down in the game and check the values change correctly
"""


if __name__ == "__main__":
    print(CHEAT_ENGINE_GUIDE)
