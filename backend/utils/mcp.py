# backend/utils/mcp.py

import json
import time

def make_message(sender, receiver, msg_type, trace_id, payload):
    return {
        "timestamp": time.time(),
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id,
        "payload": payload
    }

def log_message(msg):
    # simple console log; swap to a file or structured logger as needed
    print(json.dumps(msg, indent=2))
