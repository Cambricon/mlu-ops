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
#ifndef TEST_MLU_OP_GTEST_SRC_GTEST_PERF_TEST_H_
#define TEST_MLU_OP_GTEST_SRC_GTEST_PERF_TEST_H_

#include <libxml/xpath.h>
#include <string>

xmlXPathObjectPtr getNodeSet(xmlDocPtr doc, const xmlChar *xpath);

std::string getTestCaseName(std::string str);

bool getXmlData(std::string case_name, double *xml_time, double *workspace_size);

bool updateBaselineStrategy(double hw_time_mean,
                            double scale_bound,
                            double threshold_absolute,
                            double threshold_relative,
                            double *hw_time_base);

#endif  // TEST_MLU_OP_GTEST_SRC_GTEST_PERF_TEST_H_
