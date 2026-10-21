"""Published correctness requirement shared by evaluation entry points.

This report summarizes actual correctness checks. It is not a substitute for the
research evaluator's source-bound, isolated confirmation receipt.
"""
PUBLISHED_PROBLEMS = (
    'p00_scalar_add', 'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul',
    'r01_array_sum', 'r02_big_sum', 'r03_mean', 'r04_row_sum',
    'm01_matmul_512', 'm02_matmul_rect', 'f01_rmsnorm', 'f02_row_softmax',
    'f03_l2norm', 'a01_attention', 'a02_attention_128',
)


def published_status(rows, *, source_unchanged=True):
    """Fail closed on missing, failed or duplicated published-case records."""
    by_name = {}
    for row in rows:
        if row.get('problem') in PUBLISHED_PROBLEMS:
            by_name.setdefault(row['problem'], []).append(row)
    missing = [name for name in PUBLISHED_PROBLEMS if name not in by_name]
    failed = [name for name, records in by_name.items()
              if len(records) != 1 or not (
                  records[0].get('status') == 'pass' or
                  ('status' not in records[0] and records[0].get('exit_code') == 0))]
    return {
        'required': list(PUBLISHED_PROBLEMS), 'missing': missing, 'failed': failed,
        'source_unchanged': bool(source_unchanged),
        'all15_passed': not missing and not failed and bool(source_unchanged),
        'scope': 'Published correctness only; hidden-set correctness is unverified',
    }
