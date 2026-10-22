"""Best-effort trace emission; diagnostics must not cancel physical execution."""
import json
import sys

import numpy as np


def numeric_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported trace value: ' + type(value).__name__)


def append_event(path, record):
    """Return a visible diagnostic on failure, never fabricate a successful event."""
    diagnostic = None
    try:
        line = json.dumps(record, default=numeric_json, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        diagnostic = dict(reason='trace_serialization_failed', type=type(error).__name__,
                          detail=str(error))
        line = json.dumps(dict(channel=record.get('channel'), sequence=record['sequence'],
                               steps=record['steps'], trace_error=diagnostic))
    try:
        with path.open('a') as stream:
            stream.write(line + '\n')
    except OSError as error:
        diagnostic = dict(reason='trace_write_failed', type=type(error).__name__,
                          detail=str(error), original_error=diagnostic)
        # Service stderr has separate storage from the per-episode trace.
        try:
            print(json.dumps(dict(trace_error=diagnostic, event=line)), file=sys.stderr, flush=True)
        except OSError:
            pass
    return diagnostic
