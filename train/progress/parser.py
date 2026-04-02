"""Parse SimpleTuner stdout into normalized progress JSON lines."""

import json
import re
import time


# SimpleTuner progress patterns
_STEP_PATTERN = re.compile(
    r"[Ss]tep\s+(\d+)\s*/\s*(\d+).*?[Ll]oss:\s*([\d.eE+-]+).*?[Ll][Rr]:\s*([\d.eE+-]+)"
)
_STEP_SIMPLE = re.compile(r"[Ss]tep\s+(\d+)\s*/\s*(\d+)")
_LOSS_PATTERN = re.compile(r"[Ll]oss:\s*([\d.eE+-]+)")
_LR_PATTERN = re.compile(r"[Ll][Rr]:\s*([\d.eE+-]+)")
_CHECKPOINT_PATTERN = re.compile(r"[Ss]aving\s+checkpoint|[Cc]heckpoint\s+saved", re.I)
_VALIDATION_PATTERN = re.compile(r"[Vv]alidation\s+image.*?saved.*?(\S+\.png)", re.I)
_ERROR_PATTERN = re.compile(r"error|traceback|exception|oom|out of memory", re.I)


class ProgressParser:
    """Parses SimpleTuner stdout lines into normalized progress events."""

    def __init__(self):
        self.start_time = time.time()
        self.last_step = 0
        self.last_step_time = self.start_time

    def parse_line(self, line):
        """Parse a single stdout line. Returns a dict or None.

        Returns:
            dict with 'type' key: 'progress', 'log', 'validation', 'checkpoint', 'error'
            None if line is not meaningful
        """
        line = line.strip()
        if not line:
            return None

        # Try full step+loss+lr pattern first
        m = _STEP_PATTERN.search(line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            loss = float(m.group(3))
            lr = float(m.group(4))

            now = time.time()
            elapsed = now - self.start_time

            if step > self.last_step:
                step_delta = step - self.last_step
                time_delta = now - self.last_step_time
                steps_per_sec = step_delta / time_delta if time_delta > 0 else 0
                self.last_step = step
                self.last_step_time = now
            else:
                steps_per_sec = 0

            remaining_steps = total - step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            return {
                "type": "progress",
                "step": step,
                "total": total,
                "loss": loss,
                "lr": lr,
                "pct": round((step / total) * 100, 2) if total > 0 else 0,
                "elapsed": round(elapsed),
                "steps_per_sec": round(steps_per_sec, 3),
                "eta": round(eta),
            }

        # Partial step match (no loss/lr in same line)
        m = _STEP_SIMPLE.search(line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            loss_m = _LOSS_PATTERN.search(line)
            lr_m = _LR_PATTERN.search(line)
            result = {
                "type": "progress",
                "step": step,
                "total": total,
                "pct": round((step / total) * 100, 2) if total > 0 else 0,
            }
            if loss_m:
                result["loss"] = float(loss_m.group(1))
            if lr_m:
                result["lr"] = float(lr_m.group(1))
            self.last_step = step
            self.last_step_time = time.time()
            return result

        # Validation image saved
        m = _VALIDATION_PATTERN.search(line)
        if m:
            return {
                "type": "validation",
                "image": m.group(1),
                "message": line,
            }

        # Checkpoint saved
        if _CHECKPOINT_PATTERN.search(line):
            return {
                "type": "checkpoint",
                "message": line,
            }

        # Error detection
        if _ERROR_PATTERN.search(line):
            return {
                "type": "error",
                "message": line,
            }

        # Generic log line
        return {
            "type": "log",
            "message": line,
        }


def parse_stream(stream, callback):
    """Parse a stream of lines, calling callback(event_dict) for each.

    Args:
        stream: Iterable of strings (e.g., process.stdout)
        callback: Function called with each parsed event dict
    """
    parser = ProgressParser()
    for line in stream:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        event = parser.parse_line(line)
        if event:
            callback(event)
