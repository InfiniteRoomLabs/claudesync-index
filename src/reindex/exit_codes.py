"""
Cron-friendly exit codes (sysexits.h convention).

Cron's contract: stderr is mailed on non-zero. Wrapper scripts /
healthchecks can decide retry-vs-alert based on the code:

  0   OK              success; cron silent
  64  USAGE           bad args; do not retry, fix invocation
  65  DATAERR         input data invalid; do not retry, fix data
  69  UNAVAILABLE     remote service unavailable; cron retry
  70  SOFTWARE        internal/unhandled error; retry then alert
  74  IOERR           local I/O error; cron retry
  75  TEMPFAIL        already running OR partial failure; cron retry
  78  CONFIG          missing/invalid config (e.g. ANTHROPIC_API_KEY); fix config
  127 NOT_FOUND       required tool missing on PATH; fix install
"""

from __future__ import annotations

OK = 0
USAGE = 64
DATAERR = 65
UNAVAILABLE = 69
SOFTWARE = 70
IOERR = 74
TEMPFAIL = 75
CONFIG = 78
NOT_FOUND = 127
