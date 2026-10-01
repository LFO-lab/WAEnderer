"""Version 1 wire contract. Kept identical to runtime/erae_protocol.py."""
import math
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

PREFIX = '/waenderer/v1/'
MAX_PACKET = 1200
MAX_SEQ = 2**31 - 1
# Session and client tokens accompany every response to disambiguate restarts.
SCHEMAS = {
    'subscribe': 'si', 'snapshot': 's', 'unsubscribe': 's',
    'select': 'sisi', 'window': 'sisi',
    'result': 'ssiiis',  # engine, client, seq, accepted, requested value, reason
    'capabilities': 'sssiii',  # engine, client, corpus, sequence, ready, N; then T integers
    'state': 'sssiiisiiiiisss',
    # engine client corpus seq ready running mode valid frame generation requestedT activeT transition windowMode error
}


def validate(kind, args):
    if kind not in SCHEMAS:
        raise ValueError('unknown OSC message')
    schema = SCHEMAS[kind]
    if len(args) != len(schema) and not (kind == 'capabilities' and len(args) >= len(schema)):
        raise ValueError('wrong OSC argument count')
    types = schema + ('i' * (len(args) - len(schema)))
    for typ, value in zip(types, args):
        if typ == 's':
            if not isinstance(value, str) or len(value.encode('utf-8')) > 256 or '\0' in value:
                raise ValueError('invalid OSC string')
        elif type(value) is not int or not -2**31 <= value <= MAX_SEQ:
            raise ValueError('expected int32')
    if kind in ('select', 'window') and args[1] < 0:
        raise ValueError('negative command sequence')
    if kind == 'subscribe' and not 1 <= args[1] <= 65535:
        raise ValueError('invalid reply port')
    if kind in ('subscribe', 'snapshot', 'unsubscribe', 'select', 'window') and not args[0]:
        raise ValueError('empty client session')
    if kind == 'state':
        if args[3] < 0 or any(args[i] not in (0, 1) for i in (4, 5, 7)):
            raise ValueError('invalid state flags/sequence')
    if kind == 'capabilities':
        if args[3] < 0 or args[4] not in (0, 1) or args[5] < 0 or any(t <= 0 for t in args[6:]):
            raise ValueError('invalid capabilities')
    if kind == 'result' and (args[2] < 0 or args[3] not in (0, 1)):
        raise ValueError('invalid result')
    return tuple(args)


def encode(kind, *args):
    validate(kind, args)
    builder = OscMessageBuilder(PREFIX + kind)
    for arg in args:
        builder.add_arg(arg, 'i' if type(arg) is int else 's')
    packet = builder.build().dgram
    if len(packet) > MAX_PACKET:
        raise ValueError('OSC packet too large')
    return packet


def decode(packet):
    if len(packet) > MAX_PACKET or not packet.startswith(PREFIX.encode()):
        raise ValueError('invalid packet size/address (bundles are not accepted)')
    try:
        message = OscMessage(packet)
        kind = message.address[len(PREFIX):]
        args = validate(kind, message.params)
        # Reject noncanonical type tags, trailing bytes and implicit conversions.
        if encode(kind, *args) != packet:
            raise ValueError('noncanonical OSC packet')
        return kind, args
    except Exception as exc:
        raise ValueError('invalid OSC packet') from exc
