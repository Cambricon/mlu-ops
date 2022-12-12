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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_SRC_GTEST_MLU_OP_TEST_RESULT_PRINTER_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_SRC_GTEST_MLU_OP_TEST_RESULT_PRINTER_H_

#include <limits>
#include <ostream>
#include <vector>
#include <string>

#include "gtest/internal/gtest-internal.h"
#include "gtest/internal/gtest-string.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest-message.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest-printers.h"
#include "gtest/gtest_prod.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest-typed-test.h"
#include "gtest/gtest_pred_impl.h"

class xmlPrinter : public testing::EmptyTestEventListener {
 public:
  explicit xmlPrinter(const char *output_file);

  virtual void OnTestIterationEnd(const testing::UnitTest &unit_test,
                                  int iteration);

 private:
  static bool IsNormalizableWhitespace(char c) {
    return c == 0x9 || c == 0xA || c == 0xD;
  }

  // May c appear in a well-formed XML document?
  static bool IsValidXmlCharacter(char c) {
    return IsNormalizableWhitespace(c) || c >= 0x20;
  }

  // Returns an XML-escaped copy of the input string str.  If
  // is_attribute is true, the text is meant to appear as an attribute
  // value, and normalizable whitespace is preserved by replacing it
  // with character references.
  static std::string EscapeXml(const std::string &str, bool is_attribute);

  // Returns the given string with all characters invalid in XML removed.
  static std::string RemoveInvalidXmlCharacters(const std::string &str);

  // Convenience wrapper around EscapeXml when str is an attribute value.
  static std::string EscapeXmlAttribute(const std::string &str) {
    return EscapeXml(str, true);
  }

  // Convenience wrapper around EscapeXml when str is not an attribute value.
  static std::string EscapeXmlText(const char *str) {
    return EscapeXml(str, false);
  }

  // Verifies that the given attribute belongs to the given element and
  // streams the attribute as XML.
  static void OutputXmlAttribute(std::ostream *stream,
                                 const std::string &element_name,
                                 const std::string &name,
                                 const std::string &value);

  // Streams an XML CDATA section, escaping invalid CDATA sequences as needed.
  static void OutputXmlCDataSection(std::ostream *stream, const char *data);

  // Streams an XML representation of a TestInfo object.
  static void OutputXmlTestInfo(std::ostream *stream,
                                const char *test_case_name,
                                const testing::TestInfo &test_info);

  // Prints an XML representation of a TestCase object
  static void PrintXmlTestCase(std::ostream *stream,
                               const testing::TestCase &test_case);

  // Prints an XML summary of unit_test to output stream out.
  static void PrintXmlUnitTest(std::ostream *stream,
                               const testing::UnitTest &unit_test);

  // Produces a string representing the test properties in a result as space
  // delimited XML attributes based on the property key="value" pairs.
  // When the std::string is not empty, it includes a space at the beginning,
  // to delimit this attribute from prior attributes.
  static std::string TestPropertiesAsXmlAttributes(
      const testing::TestResult &result);

  // Streams an XML representation of the test properties of a TestResult
  // object.
  static void OutputXmlTestProperties(std::ostream *stream,
                                      const testing::TestResult &result);

  // The output file.
  const std::string output_file_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(xmlPrinter);
};

class JsonPrinter : public testing::EmptyTestEventListener {
 public:
  explicit JsonPrinter(const char *output_file);

  virtual void OnTestIterationEnd(const testing::UnitTest &unit_test,
                                  int iteration);

 private:
  // Returns an JSON-escaped copy of the input string str.
  static std::string EscapeJson(const std::string &str);

  //// Verifies that the given attribute belongs to the given element and
  //// streams the attribute as JSON.
  static void OutputJsonKey(std::ostream *stream,
                            const std::string &element_name,
                            const std::string &name, const std::string &value,
                            const std::string &indent, bool comma = true);
  static void OutputJsonKey(std::ostream *stream,
                            const std::string &element_name,
                            const std::string &name, int value,
                            const std::string &indent, bool comma = true);

  // Streams a JSON representation of a TestInfo object.
  static void OutputJsonTestInfo(::std::ostream *stream,
                                 const char *test_case_name,
                                 const testing::TestInfo &test_info);

  // Prints a JSON representation of a TestCase object
  static void PrintJsonTestCase(::std::ostream *stream,
                                const testing::TestCase &test_case);

  // Prints a JSON summary of unit_test to output stream out.
  static void PrintJsonUnitTest(::std::ostream *stream,
                                const testing::UnitTest &unit_test);

  // Produces a string representing the test properties in a result as
  // a JSON dictionary.
  static std::string TestPropertiesAsJson(const testing::TestResult &result,
                                          const std::string &indent);

  // The output file.
  const std::string output_file_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(JsonPrinter);
};

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_SRC_GTEST_MLU_OP_TEST_RESULT_PRINTER_H_
