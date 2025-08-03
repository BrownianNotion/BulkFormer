#!/bin/bash
source .venv/bin/activate
deepspeed fine_tune_bioaid.py --deepspeed ds_config.json