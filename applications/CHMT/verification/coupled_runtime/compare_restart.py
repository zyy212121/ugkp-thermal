import argparse
import hashlib
import json
from pathlib import Path
import struct


HEADER = struct.Struct('<8sQIIIIIIQQQQ')


def state_payload(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError('truncated checkpoint header')
    fields = HEADER.unpack_from(data)
    config_size,state_size = fields[-3:-1]
    begin = HEADER.size+config_size
    if begin+state_size != len(data):
        raise ValueError('checkpoint payload length mismatch')
    return data[begin:],fields[2:8]


def compare_checkpoints(reference,resumed):
    left,left_counts = state_payload(reference)
    right,right_counts = state_payload(resumed)
    if left_counts != right_counts:
        raise ValueError('checkpoint physical field dimensions differ')
    if left != right:
        offset = next((n for n,(a,b) in enumerate(zip(left,right)) if a != b),min(len(left),len(right)))
        raise ValueError('HostState payload differs at byte '+str(offset))
    temperature_count = 0
    if len(left) >= 16 and left[-8:] == b'CHMTNUM1':
        size = struct.unpack_from('<Q',left,len(left)-16)[0]
        if size > len(left)-16 or size < 8:
            raise ValueError('invalid numerical state extension size')
        temperature_count = struct.unpack_from('<Q',left,len(left)-16-size)[0]
        if size != 8+8*temperature_count:
            raise ValueError('invalid numerical state extension count')
    return dict(state_payload_bitwise_equal=True,state_bytes=len(left),
                state_sha256=hashlib.sha256(left).hexdigest(),temperature_count=temperature_count)


def terminal(case):
    out = Path(case)/'chmtOutput'
    files = list(out.glob('checkpoint-*/state.chmt'))
    if not files:
        raise ValueError('missing native checkpoint output')
    selected = max(files,key=lambda p:int(p.parent.name.split('-')[-1]))
    return out,int(selected.parent.name.split('-')[-1]),selected


def compare_cases(reference,resumed):
    left,n,left_file = terminal(reference)
    right,m,right_file = terminal(resumed)
    if n != m:
        raise ValueError('restart terminal accepted step differs')
    result = compare_checkpoints(left_file,right_file)
    output = []
    for name in ('accepted','gas','material','exchange'):
        file = name+'-'+str(n)+'.csv'
        a,b = (left/file).read_bytes(),(right/file).read_bytes()
        if a != b:
            raise ValueError('restart terminal output differs: '+file)
        output.append(dict(path=file,bytes=len(a),sha256=hashlib.sha256(a).hexdigest()))
    return dict(result,accepted_step=n,terminal_outputs=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('resumed',type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_cases(args.reference,args.resumed),indent=2))


if __name__ == '__main__':
    main()
