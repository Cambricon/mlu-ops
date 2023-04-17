/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_GTEST_ACCURACY_TEST_H_
#define TEST_MLUOP_GTEST_SRC_GTEST_ACCURACY_TEST_H_
#include <string>
#include <vector>

std::string getCaseName(std::string str);

bool getAccuracyThreshold(std::string op_name, double *threshold);

bool checkAccuracyBaselineStrategy(std::string case_name,
                                   std::vector<double> &base_errors,
                                   std::vector<double> &errors,
                                   double threshold);

#endif  // TEST_MLUOP_GTEST_SRC_GTEST_ACCURACY_TEST_H_
