"""Compare native CHMT owner payloads without requiring an external cases tree."""
import argparse
import json
from pathlib import Path
import struct


def payload(case):
    files = list((Path(case) / "chmtOutput").glob("checkpoint-*/state.chmt"))
    if not files:
        raise ValueError("no native checkpoint in " + str(case))
    path = max(files, key=lambda p: int(p.parent.name.split("-")[-1]))
    raw = path.read_bytes()
    if len(raw) < 72 or raw[:8] != b"CHMTCP2\0":
        raise ValueError("invalid checkpoint header: " + str(path))
    config_size, state_size, checksum = struct.unpack_from("<QQQ", raw, 48)
    if len(raw) != 72 + config_size + state_size:
        raise ValueError("invalid checkpoint payload length: " + str(path))
    digest = 14695981039346656037
    for byte in raw[72:]:
        digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    if digest != checksum:
        raise ValueError("checkpoint checksum mismatch: " + str(path))
    return raw[:48], raw[72:72+config_size], raw[72+config_size:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full")
    parser.add_argument("resumed")
    args = parser.parse_args()
    try:
        left, right = payload(args.full), payload(args.resumed)
        if left[0] != right[0]:
            raise ValueError("checkpoint schemas differ")
        if left[1] != right[1]:
            raise ValueError("checkpoint model configuration differs")
        if left[2] != right[2]:
            raise ValueError("HostState payload differs")
        print(json.dumps({"state_payload_bitwise_equal": True, "state_bytes": len(left[2])}))
    except ValueError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
