#!/usr/bin/env python3
"""Derive performance metrics from supplied counts/timings; no GPU or network use."""
import argparse
import json
import math
from pathlib import Path
import sys


def number(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be numeric')
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{field} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def ceiling(value, field, unit):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{field} must include {unit}, basis and source')
    result = number(value.get(unit), f'{field}.{unit}', positive=True)
    if value.get('basis') not in ('theoretical', 'measured'):
        raise ValueError(f'{field}.basis must be theoretical or measured')
    if not isinstance(value.get('source'), str) or not value['source'].strip():
        raise ValueError(f'{field}.source is required')
    return result


def calculate(row):
    if not isinstance(row, dict) or not isinstance(row.get('case'), str) or not row['case'].strip():
        raise ValueError('each case requires a nonempty case name')
    seconds = number(row.get('latency_us'), 'latency_us', positive=True) * 1e-6
    counts = {key: number(row[key], key) if row.get(key) is not None else None
              for key in ('useful_flops', 'executed_flops', 'logical_bytes', 'dram_bytes', 'l2_bytes')}
    compute = ceiling(row.get('compute_ceiling'), 'compute_ceiling', 'flops_s')
    dram = ceiling(row.get('dram_ceiling_bytes_s'), 'dram_ceiling_bytes_s', 'bytes_s')
    l2 = ceiling(row.get('l2_ceiling_bytes_s'), 'l2_ceiling_bytes_s', 'bytes_s')
    notes = []
    if compute is not None and (not isinstance(row.get('work_mode'), str)
                                or not row['work_mode'].strip()
                                or row['compute_ceiling'].get('mode') != row['work_mode']):
        notes.append('Compute ceiling mode missing or mismatched; compute utilization unavailable.')
        compute = None

    def rate(key, scale=1):
        return counts[key] / seconds / scale if counts[key] is not None else None

    def utilization(key, peak):
        return 100 * rate(key) / peak if counts[key] is not None and peak else None

    def intensity(key):
        return counts['useful_flops'] / counts[key] if counts['useful_flops'] is not None and counts[key] else None

    metrics = {
        'useful_tflops_s': rate('useful_flops', 1e12),
        'executed_tflops_s': rate('executed_flops', 1e12),
        'mfu_pct': utilization('useful_flops', compute),
        'hardware_compute_utilization_pct': utilization('executed_flops', compute),
        'logical_effective_bandwidth_gb_s': rate('logical_bytes', 1e9),
        'dram_bandwidth_gb_s': rate('dram_bytes', 1e9),
        'l2_bandwidth_gb_s': rate('l2_bytes', 1e9),
        'dram_bandwidth_utilization_pct': utilization('dram_bytes', dram),
        'l2_bandwidth_utilization_pct': utilization('l2_bytes', l2),
        'logical_arithmetic_intensity_flops_byte': intensity('logical_bytes'),
        'dram_arithmetic_intensity_flops_byte': intensity('dram_bytes'),
        'l2_arithmetic_intensity_flops_byte': intensity('l2_bytes'),
    }
    for key, value in metrics.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f'{row["case"]}: derived {key} is nonfinite; check units and magnitude')
        if key.endswith('_pct') and value is not None and value > 100:
            notes.append(f'{key} exceeds 100%; verify counts, units, ceiling and measurement regime.')
    return {'input': row, 'metrics': metrics, 'notes': notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('cases'), list) or not data['cases']:
            raise ValueError('input must contain a nonempty cases list')
        rows = [calculate(row) for row in data['cases']]
        names = [row['input']['case'] for row in rows]
        if len(set(names)) != len(names):
            raise ValueError('case names must be unique within this report')
        print(json.dumps({'schema': 'kernel-performance-metrics-v1',
                          'qualification': 'Derived arithmetic only; no correctness or bottleneck certification.',
                          'cases': rows}, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, OverflowError) as error:
        print(f'metrics: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
