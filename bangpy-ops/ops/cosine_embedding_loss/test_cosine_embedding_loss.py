# Copyright (C) [2022] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Test cosineEmbeddingLoss operator with multi-platform code link"""
import numpy as np
import pytest
from cosine_embedding_loss import DTYPES, KERNEL_NAME, TARGET_LIST
import bangpy
from evaluate_functions import evaluate
from bangpy.common import load_op_by_type


np.set_printoptions(threshold=np.inf)


@pytest.mark.parametrize(
    "data_amount", [2 ** 20 * 10, 2 ** 30, 2 ** 30 * 2, 2 ** 30 * 4, 2 ** 30 * 8]
)
@pytest.mark.parametrize(
    "data_width",
    [
        2 ** 5,
        2 ** 5 + 1,
        2 ** 5 - 1,
        2 ** 6,
        2 ** 7,
        2 ** 8,
        2 ** 9,
        2 ** 10,
        2 ** 11 + 1,
        2 ** 11 - 1,
        2 ** 12,
        2 ** 13,
        2 ** 14,
        2 ** 15,
        2 ** 16,
        2 ** 17,
        2 ** 18,
        2 ** 19,
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_cosine_embedding_loss(target, data_amount, data_width, dtype):
    """
    pytest main function
    """

    if target not in TARGET_LIST:
        return
    # float16 is 64 aligned
    if data_width == 32 and dtype == bangpy.float16:
        return
    f = load_op_by_type(KERNEL_NAME, dtype.name)

    evaluate(f, dtype, data_amount, data_width)
