#!/usr/bin/env python
from stage_cli import run_stage

from dfc_sam.engine.stages import Stage

if __name__ == "__main__":
    run_stage(Stage.JOINT)
