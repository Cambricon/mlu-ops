# Copyright (C) [2023] by Cambricon, Inc.
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

""" Pytest config file"""
import pytest


def pytest_addoption(parser):
    """pytest hook function, only used for pytest command line parameter."""
    parser.addoption(
        "--target",
        action="store",
        default="mlu270",
        help="Test target tag: mlu220-m2, mlu270, mlu290, mlu370-s4.",
    )


@pytest.fixture
def target(request):
    """Fixture function for test get target tag."""
    target_tag = request.config.getoption("--target")
    assert target_tag in ("mlu220-m2", "mlu270", "mlu290", "mlu370-s4")
    return target_tag
