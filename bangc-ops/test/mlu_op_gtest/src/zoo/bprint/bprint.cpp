/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "bprint.h"

namespace mluoptest {

void BprintExecutor::paramCheck() {
    GTEST_CHECK(parser_->inputs().size() == 1,
                "bprint tensor input number is wrong.");
}

void BprintExecutor::compute() {
    VLOG(4) << "BprintExecutor compute ";
    auto tensor_x = tensor_desc_[0].tensor;
    auto dev_x = data_vector_[0].device_ptr;
    VLOG(4) << "call mluOpBprint()";
    interface_timer_.start();
    MLUOP_CHECK(mluOpBprint(handle_, tensor_x, dev_x));
    interface_timer_.stop();
}

void BprintExecutor::cpuCompute() { printf("mluOpBprint Print Success.\n"); }

int64_t BprintExecutor::getTheoryOps() {
    int64_t theory_ops = parser_->input(0)->total_count;
    VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
    return theory_ops;
}

} // namespace mluoptest
