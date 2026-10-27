# Scaled-score mutable alias regression

Byte-exact export of the existing campaign case `e41dbf0c789e8ae5d9e2cc4b31f2e5b4ee1f04cc66551c937d44454dadf54c53`; no new case was generated during consolidation.

A dead scaled-score Assign followed by Accum exposes accidental aliasing of a cached physical register. The original 65 GPU cases had no such observer, and the integer-copy CPU case does not activate this optimization. A prescale candidate failed 1023 outputs; the baseline and repaired candidate passed. Inputs, launch/ABI and published tolerances remain unchanged. The independent NumPy reference is retained byte-for-byte.

The combined experiment gate includes this case alongside all original 65 GPU witnesses. Historical campaign evidence is referenced in `docs/evidence/campaign-20261024-speed.json`; exporting this payload does not create a new execution receipt.
