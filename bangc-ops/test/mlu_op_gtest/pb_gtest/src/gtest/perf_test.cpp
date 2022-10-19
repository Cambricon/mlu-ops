/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <string>
#include <cstring>
#include <unordered_set>
#include "perf_test.h"
#include "core/logging.h"

// baseline default threshold
#define DEFAULT_SCALE_BOUND (100)
#define DEFAULT_THRESHOLD_ABSOLUTE (5)
#define DEFAULT_THRESHOLD_RELATIVE (0.04f)

xmlXPathObjectPtr getNodeSet(xmlDocPtr doc, const xmlChar *xpath) {
  xmlXPathContextPtr context = NULL;
  xmlXPathObjectPtr result = NULL;
  context = xmlXPathNewContext(doc);
  if (context == NULL) {
    LOG(ERROR) << "getNodeSet:Get context of xml file failed.";
    return NULL;
  }
  result = xmlXPathEvalExpression(xpath, context);
  xmlXPathFreeContext(context);
  if (result == NULL) {
    LOG(INFO) << "getNodeSet:Get XPath expression of xml file failed.";
    return NULL;
  }
  if (xmlXPathNodeSetIsEmpty(result->nodesetval)) {
    xmlXPathFreeObject(result);
    LOG(INFO) << "getNodeSet:XPath node set is empty.";
    return NULL;
  }
  return result;
}

// get hardware_time_base in xml file
bool getXmlData(std::string case_name, double *xml_time,
                double *workspace_size) {
  std::string xml_file;
  if (getenv("MLUOP_BASELINE_XML_FILE") != NULL) {
    xml_file = getenv("MLUOP_BASELINE_XML_FILE");
  } else {
    LOG(ERROR) << "getXmlData:The env of MLUOP_BASELINE_XML_FILE is NULL.";
    return false;
  }

  *xml_time = 0;
  xmlDocPtr doc = NULL;
  xmlKeepBlanksDefault(0);
  doc = xmlReadFile(xml_file.c_str(), "UTF-8", XML_PARSE_RECOVER);
  if (doc == NULL) {
    LOG(INFO) << "open " << xml_file << " failed.";
    return false;
  }
  std::string xpath_string = "///testcase//*[@name='" + case_name + "']";
  xmlChar *xpath = BAD_CAST(xpath_string.c_str());
  xmlXPathObjectPtr search_result = getNodeSet(doc, xpath);
  if (search_result == NULL) {
    LOG(INFO) << "search " << case_name
              << " data in mlu_op_base_data.xml failed.";
    xmlFreeDoc(doc);
    return false;
  }
  std::string search_data = case_name;
  if (search_result) {
    xmlNodeSetPtr nodeset = NULL;
    xmlNodePtr property = NULL;
    xmlChar *name = NULL;
    nodeset = search_result->nodesetval;
    property = nodeset->nodeTab[0];

    xmlNodePtr property_next = NULL;
    property_next = property->next;
    // get hw_time_base
    if (xmlHasProp(property, BAD_CAST("name")) &&
        xmlHasProp(property, BAD_CAST("value"))) {
      name = xmlGetProp(property, BAD_CAST "name");
      if (!xmlStrcmp(name, BAD_CAST(search_data.c_str()))) {
        xmlChar *value = xmlGetProp(property, BAD_CAST "value");
        *xml_time = atof((const char *)value);
        xmlFree(value);
      }
      xmlFree(name);
    } else {
      LOG(ERROR) << "getXmlData:search the name or value of property failed "
                    "when getting hw_time_base.";
      xmlXPathFreeObject(search_result);
      xmlFreeDoc(doc);
      return false;
    }
    // get workspace_size_mlu
    if (xmlHasProp(property_next, BAD_CAST("name")) &&
        xmlHasProp(property_next, BAD_CAST("value"))) {
      name = xmlGetProp(property_next, BAD_CAST "name");
      if (!xmlStrcmp(name, BAD_CAST("workspace_size_mlu"))) {
        xmlChar *value = xmlGetProp(property_next, BAD_CAST "value");
        *workspace_size = atof((const char *)value);
        xmlFree(value);
      }
      xmlFree(name);
    } else {
      LOG(ERROR) << "getXmlData:search the name or value of property failed "
                    "when getting workspace size.";
      xmlXPathFreeObject(search_result);
      xmlFreeDoc(doc);
      return false;
    }

    xmlXPathFreeObject(search_result);
    xmlFreeDoc(doc);
    return true;
  }
  return false;
}

// get pb or prototxt file name
std::string getTestCaseName(std::string str) {
  std::string::size_type start = str.find_last_of("/");
  std::string result;
  if (start != std::string::npos) {
    start += 1;
    result = str.substr(start);
  } else {
    result = str;
  }
  return result;
}

// update baseline hardware_time_base
bool updateBaselineStrategy(double hw_time_mean, double scale_bound,
                            double threshold_absolute,
                            double threshold_relative, double *hw_time_base) {
  if (*hw_time_base <= 0) {
    LOG(ERROR) << "updateBaselineStrategy:baseline time is error, it must be "
                  "greater than zero";
    return false;
  }
  if (scale_bound <= 0 || threshold_absolute <= 0 || threshold_relative <= 0) {
    LOG(ERROR) << "updateBaselineStrategy:threshold is wrong, it must be "
                  "greater than zero";
    return false;
  }
  double time_base = *hw_time_base;
  double time_diff = hw_time_mean - time_base;
  bool is_baseline_pass = true;
  if (time_base <= scale_bound) {  // small scale
    if (time_diff > threshold_absolute) {
      is_baseline_pass = false;
    } else if (time_diff <= threshold_absolute) {
      is_baseline_pass = true;
      *hw_time_base = (time_base + hw_time_mean) / 2;
    }
  } else {  // big scale
    if (time_diff / time_base > threshold_relative) {
      is_baseline_pass = false;
    } else if (time_diff / time_base <= threshold_relative) {
      is_baseline_pass = true;
      *hw_time_base = (time_base + hw_time_mean) / 2;
    }
  }
  return is_baseline_pass;
}
