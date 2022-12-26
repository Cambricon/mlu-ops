# Copyright (C) [2021] by Cambricon, Inc.
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
# pylint: disable=invalid-name, missing-function-docstring, useless-object-inheritance
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, attribute-defined-outside-init
"""Non-maximum suppression operator implementation using BANGPy TCP API."""
import pytest
import numpy as np
import bangpy as bp
from bangpy.common import load_op_by_type
from nms import KERNEL_NAME, TARGET_LIST


def _py_nms(output, iou_threshold=0.5, score_threshold=0.5, valid_num=1):
    def _iou(boxA, boxB):
        """Compute the Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangle
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    assert len(output) != 0, "box num must be valid!"
    output = output[np.argsort(-output[:, 0], kind="stable")]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j, _ in enumerate(bboxes):
            if _iou(bbox[1:5], bboxes[j][1:5]) >= iou_threshold:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, float)
    score_mask = bboxes[:, 0] > score_threshold
    bboxes = bboxes[score_mask, :]

    if len(bboxes) >= valid_num:
        return_value = bboxes[:valid_num]

    else:
        zeros = np.zeros((valid_num - len(bboxes), 5))
        return_value = np.vstack([bboxes, zeros])

    return return_value


def verify_operator(
    box_num, max_output_size=None, iou_threshold=0.5, score_threshold=0.5, dtype=None
):
    """Check the NMS's result."""
    max_output_size = box_num if not max_output_size else max_output_size
    score_list = list(np.linspace(0, 1.0, box_num + 1000))
    score_array = np.random.choice(score_list, size=box_num, replace=False)
    box_array = np.random.uniform(size=[4, box_num], low=1.0, high=100.0)
    x1 = np.random.uniform(size=[box_num], low=1.0, high=100.0)
    y1 = np.random.uniform(size=[box_num], low=1.0, high=100.0)
    x2 = np.random.uniform(size=[box_num], low=10.0, high=100.0)
    y2 = np.random.uniform(size=[box_num], low=10.0, high=100.0)
    x2 += x1
    y2 += y1
    dev = bp.device(0)
    box_array = np.vstack([x1, y1, x2, y2])
    score_temp = bp.Array(score_array.astype(dtype.as_numpy_dtype), dev)
    score = bp.Array(np.zeros([box_num], dtype=dtype.as_numpy_dtype), dev)
    box = bp.Array(box_array.astype(dtype.as_numpy_dtype), dev)
    output = bp.Array(np.zeros([max_output_size, 5], dtype=dtype.as_numpy_dtype), dev)
    f = load_op_by_type(KERNEL_NAME, dtype.name)
    evaluator = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
    print(
        "nms : %f ms"
        % (
            evaluator(
                score,
                score_temp,
                box,
                output,
                box_num,
                max_output_size,
                iou_threshold,
                score_threshold,
            ).mean
            * 1e3
        )
    )
    box_array = np.vstack([score_array, box_array])
    cpu_output = _py_nms(box_array.transpose(), valid_num=max_output_size)
    bp.assert_allclose(
        output.numpy(), cpu_output.astype(dtype.as_numpy_dtype), rtol=1e-2
    )


@pytest.mark.parametrize(
    "box_num",
    [16, 15, 300],
)
@pytest.mark.parametrize(
    "max_output_size",
    [1, 4],
)
@pytest.mark.parametrize(
    "iou_threshold",
    [0.5],
)
@pytest.mark.parametrize(
    "score_threshold",
    [0.5],
)
@pytest.mark.parametrize(
    "dtype",
    [bp.float32, bp.float16],
)
def test_nms(target, box_num, max_output_size, iou_threshold, score_threshold, dtype):
    """Test nms operator by giving multiple sets of parameters."""
    if target not in TARGET_LIST:
        return
    verify_operator(box_num, max_output_size, iou_threshold, score_threshold, dtype)
