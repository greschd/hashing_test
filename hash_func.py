#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>

import hashlib
import numbers
try: # Python3
    from functools import singledispatch
    from collections.abc import Iterable
except ImportError: # Python2
    from singledispatch import singledispatch
    from collections import Iterable

import numpy as np

@singledispatch
def create_hash(obj):
    raise ValueError('Cannot hash object of type {}'.format(type(obj)))

@create_hash.register(type(hashlib.md5()))
def _(obj):
    return obj

@create_hash.register(bytes)
def _(obj):
    return hashlib.md5(obj)

@create_hash.register(unicode)
def _(obj):
    return create_hash(obj.encode())

@create_hash.register(np.int64)
@create_hash.register(int)
def _(obj):
    return create_hash(str(obj).encode('utf8'))

@create_hash.register(np.float64)
@create_hash.register(float)
def _(obj):
    return create_hash(truncate_float(obj).tobytes())

def truncate_float(x, num_bits=12):
    mask = ~(2**num_bits - 1)
    int_repr = np.float64(x).view(np.int64)
    masked_int = int_repr & mask
    return masked_int.view(np.float64)

@create_hash.register(Iterable)
def _(obj):
    # print(obj)
    return create_hash(np.array([create_hash(o).hexdigest() for o in obj]))

@create_hash.register(np.ndarray)
def _(obj):
    if obj.dtype == np.float64:
        return create_hash(truncate_array(obj).tobytes())
    elif obj.dtype == np.complex128:
        return create_hash((truncate_array(obj.real) + 1j * truncate_array(obj.imag)).tobytes())
    else:
        return create_hash(obj.tobytes())

def truncate_array(x, num_bits=12):
    mask = ~(2**num_bits - 1)
    int_array = np.array(x, dtype=np.float64).view(np.int64)
    masked_array = int_array & mask
    return masked_array.view(np.float64)

@create_hash.register(dict)
def _(obj):
    hash_dict = {create_hash(k).hexdigest(): create_hash(v).hexdigest() for k, v in obj.items()}
    return create_hash(sorted(hash_dict.items()))

if __name__ == '__main__':
    def print_hash(obj):
        print(create_hash(obj).hexdigest())

    print('string:')
    print_hash(u'test string')
    print_hash('test string')
    print('float:')
    print_hash(1.)
    print_hash(1. + 1e-13)
    print('dict:')
    print_hash(dict(a=12., b=[1, 2, [3, 4]]))
    print_hash(dict(a=12. + 1e-14, b=[1, 2, [3, 4]]))
    print('complex array:')
    print_hash(np.array([1j]))
    print_hash(np.array([1j + 1e-13j]))
