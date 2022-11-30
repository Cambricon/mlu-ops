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
#include "gtest/internal/custom/gtest.h"
#include "gtest/gtest-spi.h"

#include <ctype.h>   // NOLINT
#include <math.h>    // NOLINT
#include <stdarg.h>  // NOLINT
#include <stdlib.h>  // NOLINT
#include <wchar.h>   // NOLINT
#include <wctype.h>  // NOLINT

#include <algorithm> // NOLINT
#include <iomanip>   // NOLINT
#include <sstream>   // NOLINT
#include <string>    // NOLINT

#if GTEST_OS_LINUX

// TODO(kenton@google.com): Use autoconf to detect availability of
// gettimeofday().
#define GTEST_HAS_GETTIMEOFDAY_ 1

#include <fcntl.h>   // NOLINT
#include <limits.h>  // NOLINT
#include <sched.h>   // NOLINT
// Declares vsnprintf().  This header is not available on Windows.
#include <strings.h>   // NOLINT
#include <sys/mman.h>  // NOLINT
#include <sys/time.h>  // NOLINT
#include <unistd.h>    // NOLINT

#elif GTEST_OS_SYMBIAN
#define GTEST_HAS_GETTIMEOFDAY_ 1
#include <sys/time.h>  // NOLINT

#elif GTEST_OS_ZOS
#define GTEST_HAS_GETTIMEOFDAY_ 1
#include <sys/time.h>  // NOLINT

// On z/OS we additionally need strings.h for strcasecmp.
#include <strings.h>   // NOLINT

#elif GTEST_OS_WINDOWS_MOBILE  // We are on Windows CE.

#include <windows.h>  // NOLINT
#undef min

#elif GTEST_OS_WINDOWS  // We are on Windows proper.

#include <io.h>         // NOLINT
#include <sys/timeb.h>  // NOLINT
#include <sys/types.h>  // NOLINT
#include <sys/stat.h>   // NOLINT

#if GTEST_OS_WINDOWS_MINGW
// MinGW has gettimeofday() but not _ftime64().
// TODO(kenton@google.com): Use autoconf to detect availability of
//   gettimeofday().
// TODO(kenton@google.com): There are other ways to get the time on
//   Windows, like GetTickCount() or GetSystemTimeAsFileTime().  MinGW
//   supports these.  consider using them instead.
#define GTEST_HAS_GETTIMEOFDAY_ 1
#include <sys/time.h>  // NOLINT
#endif                 // GTEST_OS_WINDOWS_MINGW

// cpplint thinks that the header is already included, so we want to
// silence it.
#include <windows.h>   // NOLINT
#undef min

#else

// Assume other platforms have gettimeofday().
// TODO(kenton@google.com): Use autoconf to detect availability of
//   gettimeofday().
#define GTEST_HAS_GETTIMEOFDAY_ 1

// cpplint thinks that the header is already included, so we want to
// silence it.
#include <sys/time.h>  // NOLINT
#include <unistd.h>    // NOLINT

#endif  // GTEST_OS_LINUX

#if GTEST_HAS_EXCEPTIONS
#include <stdexcept>
#endif

#if GTEST_CAN_STREAM_RESULTS_
#include <arpa/inet.h>   // NOLINT
#include <netdb.h>       // NOLINT
#include <sys/socket.h>  // NOLINT
#include <sys/types.h>   // NOLINT
#endif

#include "src/gtest-internal-inl.h"

#if GTEST_OS_WINDOWS
#define vsnprintf _vsnprintf
#endif  // GTEST_OS_WINDOWS

#include "mlu_op_test_result_printer.h"
#include "gtest/internal/gtest-filepath.h"
static const char kUniversalFilter[] = "*";
static const char *GetDefaultFilter() {
#ifdef GTEST_TEST_FILTER_ENV_VAR_
  const char *const testbridge_test_only = getenv(GTEST_TEST_FILTER_ENV_VAR_);
  if (testbridge_test_only != NULL) {
    return testbridge_test_only;
  }
#endif  // GTEST_TEST_FILTER_ENV_VAR_
  return kUniversalFilter;
}
// The list of reserved attributes used in the <testsuites> element of XML
// output.
static const char *const kReservedTestSuitesAttributes[] = {
    "disabled",    "errors", "failures", "name",
    "random_seed", "tests",  "time",     "timestamp"};

// The list of reserved attributes used in the <testsuite> element of XML
// output.
static const char *const kReservedTestSuiteAttributes[] = {
    "disabled", "errors", "failures", "name", "tests", "time"};

// The list of reserved attributes used in the <testcase> element of XML output.
static const char *const kReservedTestCaseAttributes[] = {
    "classname", "name", "status", "time", "type_param", "value_param"};
template <int kSize>
std::vector<std::string> ArrayAsVector(const char *const (&array)[kSize]) {
  return std::vector<std::string>(array, array + kSize);
}

static std::vector<std::string> GetReservedAttributesForElement(
    const std::string &xml_element) {
  if (xml_element == "testsuites") {
    return ArrayAsVector(kReservedTestSuitesAttributes);
  } else if (xml_element == "testsuite") {
    return ArrayAsVector(kReservedTestSuiteAttributes);
  } else if (xml_element == "testcase") {
    return ArrayAsVector(kReservedTestCaseAttributes);
  } else {
    GTEST_CHECK_(false) << "Unrecognized xml_element provided: " << xml_element;
  }
  // This code is unreachable but some compilers may not realizes that.
  return std::vector<std::string>();
}

using testing::internal::StreamableToString;

void xmlPrinter::OnTestIterationEnd(const testing::UnitTest &unit_test,
                                    int iteration) {
  FILE *xmlout = NULL;
  std::string output_file_name = output_file_;
  if (testing::GTEST_FLAG(repeat) > 1) {
    std::string append_string = "_" + std::to_string(iteration);
    output_file_name.insert(output_file_name.find("."), append_string);
  }
  testing::internal::FilePath output_file(output_file_name);
  testing::internal::FilePath output_dir(output_file.RemoveFileName());
  if (output_dir.CreateDirectoriesRecursively()) {
    xmlout = testing::internal::posix::FOpen(output_file_name.c_str(), "w");
  }
  if (xmlout == NULL) {
    GTEST_LOG_(FATAL) << "Unable to open file \"" << output_file_name << "\"";
  }
  std::stringstream stream;
  PrintXmlUnitTest(&stream, unit_test);
  fprintf(xmlout, "%s",
          testing::internal::StringStreamToString(&stream).c_str());
  fclose(xmlout);
}

void xmlPrinter::OutputXmlAttribute(std::ostream *stream,
                                    const std::string &element_name,
                                    const std::string &name,
                                    const std::string &value) {
  const std::vector<std::string> &allowed_names =
      GetReservedAttributesForElement(element_name);

  GTEST_CHECK_(std::find(allowed_names.begin(), allowed_names.end(), name) !=
               allowed_names.end())
      << "Attribute " << name << " is not allowed for element <" << element_name
      << ">.";

  *stream << " " << name << "=\"" << EscapeXmlAttribute(value) << "\"";
}
// Creates a new XmlUnitTestResultPrinter.
xmlPrinter::xmlPrinter(const char *output_file) : output_file_(output_file) {
  if (output_file_.c_str() == NULL || output_file_.empty()) {
    GTEST_LOG_(FATAL) << "XML output file may not be null";
  }
}
// Returns an XML-escaped copy of the input string str.  If is_attribute
// is true, the text is meant to appear as an attribute value, and
// normalizable whitespace is preserved by replacing it with character
// references.
//
// Invalid XML characters in str, if any, are stripped from the output.
// It is expected that most, if not all, of the text processed by this
// module will consist of ordinary English text.
// If this module is ever modified to produce version 1.1 XML output,
// most invalid characters can be retained using character references.
// TODO(wan): It might be nice to have a minimally invasive, human-readable
// escaping scheme for invalid characters, rather than dropping them.
std::string xmlPrinter::EscapeXml(const std::string &str, bool is_attribute) {
  testing::Message m;

  for (size_t i = 0; i < str.size(); ++i) {
    const char ch = str[i];
    switch (ch) {
      case '<':
        m << "&lt;";
        break;
      case '>':
        m << "&gt;";
        break;
      case '&':
        m << "&amp;";
        break;
      case '\'':
        if (is_attribute)
          m << "&apos;";
        else
          m << '\'';
        break;
      case '"':
        if (is_attribute)
          m << "&quot;";
        else
          m << '"';
        break;
      default:
        if (IsValidXmlCharacter(ch)) {
          if (is_attribute && IsNormalizableWhitespace(ch))
            m << "&#x"
              << testing::internal::String::FormatByte(
                     static_cast<unsigned char>(ch))
              << ";";
          else
            m << ch;
        }
        break;
    }
  }

  return m.GetString();
}

// Returns the given string with all characters invalid in XML removed.
// Currently invalid characters are dropped from the string. An
// alternative is to replace them with certain characters such as . or ?.
std::string xmlPrinter::RemoveInvalidXmlCharacters(const std::string &str) {
  std::string output;
  output.reserve(str.size());
  for (std::string::const_iterator it = str.begin(); it != str.end(); ++it)
    if (IsValidXmlCharacter(*it)) output.push_back(*it);
  return output;
}
// Formats the given time in milliseconds as seconds.
std::string FormatTimeInMillisAsSeconds(testing::internal::TimeInMillis ms) {
  ::std::stringstream ss;
  ss << (static_cast<double>(ms) * 1e-3);
  return ss.str();
}

static bool PortableLocaltime(time_t seconds, struct tm *out) {
#if defined(_MSC_VER)
  return localtime_s(out, &seconds) == 0;
#elif defined(__MINGW32__) || defined(__MINGW64__)
  // MINGW <time.h> provides neither localtime_r nor localtime_s, but uses
  // Windows' localtime(), which has a thread-local tm buffer.
  struct tm *tm_ptr = localtime(&seconds);  // NOLINT
  if (tm_ptr == NULL) return false;
  *out = *tm_ptr;
  return true;
#else
  return localtime_r(&seconds, out) != NULL;
#endif
}
// Converts the given epoch time in milliseconds to a date string in the ISO
// 8601 format, without the timezone information.
std::string FormatEpochTimeInMillisAsIso8601(
    testing::internal::TimeInMillis ms) {
  struct tm time_struct;
  if (!PortableLocaltime(static_cast<time_t>(ms / 1000), &time_struct))
    return "";
  // YYYY-MM-DDThh:mm:ss
  return StreamableToString(time_struct.tm_year + 1900) + "-" +
         testing::internal::String::FormatIntWidth2(time_struct.tm_mon + 1) +
         "-" + testing::internal::String::FormatIntWidth2(time_struct.tm_mday) +
         "T" + testing::internal::String::FormatIntWidth2(time_struct.tm_hour) +
         ":" + testing::internal::String::FormatIntWidth2(time_struct.tm_min) +
         ":" + testing::internal::String::FormatIntWidth2(time_struct.tm_sec);
}

// Streams an XML CDATA section, escaping invalid CDATA sequences as needed.
void xmlPrinter::OutputXmlCDataSection(::std::ostream *stream,
                                       const char *data) {
  const char *segment = data;
  *stream << "<![CDATA[";
  for (;;) {
    const char *const next_segment = strstr(segment, "]]>");
    if (next_segment != NULL) {
      stream->write(segment,
                    static_cast<std::streamsize>(next_segment - segment));
      *stream << "]]>]]&gt;<![CDATA[";
      segment = next_segment + strlen("]]>");
    } else {
      *stream << segment;
      break;
    }
  }
  *stream << "]]>";
}

// Prints an XML representation of a TestInfo object.
// TODO(wan): There is also value in printing properties with the plain printer.
void xmlPrinter::OutputXmlTestInfo(::std::ostream *stream,
                                   const char *test_case_name,
                                   const testing::TestInfo &test_info) {
  const testing::TestResult &result = *test_info.result();
  const std::string kTestcase = "testcase";

  if (test_info.is_in_another_shard()) {
    return;
  }

  *stream << "    <testcase";
  OutputXmlAttribute(stream, kTestcase, "name", test_info.name());

  if (test_info.value_param() != NULL) {
    int error_count = 0;
    if (getenv("MLUOP_XML_VALUE_PARAM_PROCESS") != NULL &&
        strcmp(getenv("MLUOP_XML_VALUE_PARAM_PROCESS"), "ON") == 0) {
      std::string str = test_info.value_param();
      testing::internal::string str_pb(".pb");
      testing::internal::string str_prototxt(".prototxt");
      testing::internal::string::size_type start = str.find_last_of("/");
      if (start != testing::internal::string::npos) {
        start += 1;
      } else {
        error_count++;
      }

      testing::internal::string::size_type end = str.find(".pb", 0);
      if (end != testing::internal::string::npos) {
        end += str_pb.size();
      } else {
        end = str.find(".prototxt", 0);
        if (end != testing::internal::string::npos) {
          end += str_prototxt.size();
        } else {
          error_count++;
        }
      }
      if (error_count == 0) {
        testing::internal::string result = str.substr(start, end - start);
        OutputXmlAttribute(stream, kTestcase, "value_param", result);
      } else {
        OutputXmlAttribute(stream, kTestcase, "value_param",
                           test_info.value_param());
      }
    } else {
      OutputXmlAttribute(stream, kTestcase, "value_param",
                         test_info.value_param());
    }
  }

  if (test_info.type_param() != NULL) {
    OutputXmlAttribute(stream, kTestcase, "type_param", test_info.type_param());
  }

  OutputXmlAttribute(stream, kTestcase, "status",
                     test_info.should_run() ? "run" : "notrun");
  OutputXmlAttribute(stream, kTestcase, "time",
                     FormatTimeInMillisAsSeconds(result.elapsed_time()));
  OutputXmlAttribute(stream, kTestcase, "classname", test_case_name);

  int failures = 0;
  for (int i = 0; i < result.total_part_count(); ++i) {
    const testing::TestPartResult &part = result.GetTestPartResult(i);
    if (part.failed()) {
      if (++failures == 1) {
        *stream << ">\n";
      }
      const std::string location =
          testing::internal::FormatCompilerIndependentFileLocation(
              part.file_name(), part.line_number());
      const std::string summary = location + "\n" + part.summary();
      *stream << "      <failure message=\""
              << EscapeXmlAttribute(summary.c_str()) << "\" type=\"\">";
      const std::string detail = location + "\n" + part.message();
      OutputXmlCDataSection(stream, RemoveInvalidXmlCharacters(detail).c_str());
      *stream << "</failure>\n";
    }
  }

  if (failures == 0 && result.test_property_count() == 0) {
    *stream << " />\n";
  } else {
    if (failures == 0) {
      *stream << ">\n";
    }
    OutputXmlTestProperties(stream, result);
    *stream << "    </testcase>\n";
  }
}

// Prints an XML representation of a TestCase object
void xmlPrinter::PrintXmlTestCase(std::ostream *stream,
                                  const testing::TestCase &test_case) {
  const std::string kTestsuite = "testsuite";
  *stream << "  <" << kTestsuite;
  OutputXmlAttribute(stream, kTestsuite, "name", test_case.name());
  OutputXmlAttribute(stream, kTestsuite, "tests",
                     StreamableToString(test_case.reportable_test_count()));
  OutputXmlAttribute(stream, kTestsuite, "failures",
                     StreamableToString(test_case.failed_test_count()));
  OutputXmlAttribute(
      stream, kTestsuite, "disabled",
      StreamableToString(test_case.reportable_disabled_test_count()));
  OutputXmlAttribute(stream, kTestsuite, "errors", "0");
  OutputXmlAttribute(stream, kTestsuite, "time",
                     FormatTimeInMillisAsSeconds(test_case.elapsed_time()));
  *stream << TestPropertiesAsXmlAttributes(test_case.ad_hoc_test_result())
          << ">\n";

  // provide bug information for AuotoJira to create Jira Issues
  // if (test_case.failed_test_count()) {
  *stream << "    <properties>\n";
  *stream << "      <property name=\"project\" value=\"MLUOPCORE\" />\n";
  *stream << "      <property name=\"component\" value=\"mluOps\" />\n";
  *stream << "      <property name=\"bug_level\" value=\"一般\" />\n";
  *stream << "      <property name=\"bug_label\" value=\"master\" />\n";
  *stream << "      <property name=\"product_component\" value=\"mluOps-Core\" "
             "/>\n";
  *stream << "      <property name=\"bug_source\" value=\"集成测试\" />\n";
  *stream << "      <property name=\"bug_category\" value=\"功能\" />\n";
  *stream << "      <property name=\"bug_frequency\" value=\"必现\" />\n";
  *stream << "      <property name=\"bug_boardform\" value=\" \" />\n";
  *stream << "      <property name=\"bug_scenarios\" value=\"通用\" />\n";
  *stream << "    </properties>\n";
  //}

  for (int i = 0; i < test_case.total_test_count(); ++i) {
    if (test_case.GetTestInfo(i)->is_reportable())
      OutputXmlTestInfo(stream, test_case.name(), *test_case.GetTestInfo(i));
  }
  *stream << "  </" << kTestsuite << ">\n";
}

void xmlPrinter::PrintXmlUnitTest(std::ostream *stream,
                                  const testing::UnitTest &unit_test) {
  const std::string kTestsuites = "testsuites";
  *stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  *stream << "<" << kTestsuites;
  OutputXmlAttribute(stream, kTestsuites, "tests",
                     StreamableToString(unit_test.reportable_test_count()));
  OutputXmlAttribute(stream, kTestsuites, "failures",
                     StreamableToString(unit_test.failed_test_count()));
  OutputXmlAttribute(
      stream, kTestsuites, "disabled",
      StreamableToString(unit_test.reportable_disabled_test_count()));
  OutputXmlAttribute(stream, kTestsuites, "errors", "0");
  OutputXmlAttribute(
      stream, kTestsuites, "timestamp",
      FormatEpochTimeInMillisAsIso8601(unit_test.start_timestamp()));
  OutputXmlAttribute(stream, kTestsuites, "time",
                     FormatTimeInMillisAsSeconds(unit_test.elapsed_time()));

  if (testing::GTEST_FLAG(shuffle)) {
    OutputXmlAttribute(stream, kTestsuites, "random_seed",
                       StreamableToString(unit_test.random_seed()));
  }
  *stream << TestPropertiesAsXmlAttributes(unit_test.ad_hoc_test_result());

  OutputXmlAttribute(stream, kTestsuites, "name", "AllTests");
  *stream << ">\n";

  for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
    if (unit_test.GetTestCase(i)->reportable_test_count() > 0)
      PrintXmlTestCase(stream, *unit_test.GetTestCase(i));
  }
  *stream << "</" << kTestsuites << ">\n";
}
// Produces a string representing the test properties in a result as space
// delimited XML attributes based on the property key="value" pairs.
std::string xmlPrinter::TestPropertiesAsXmlAttributes(
    const testing::TestResult &result) {
  testing::Message attributes;
  for (int i = 0; i < result.test_property_count(); ++i) {
    const testing::TestProperty &property = result.GetTestProperty(i);
    attributes << " " << property.key() << "="
               << "\"" << EscapeXmlAttribute(property.value()) << "\"";
  }
  return attributes.GetString();
}

void xmlPrinter::OutputXmlTestProperties(std::ostream *stream,
                                         const testing::TestResult &result) {
  const std::string kProperties = "properties";
  const std::string kProperty = "property";

  if (result.test_property_count() <= 0) {
    return;
  }

  *stream << "<" << kProperties << ">\n";
  for (int i = 0; i < result.test_property_count(); ++i) {
    const testing::TestProperty &property = result.GetTestProperty(i);
    *stream << "<" << kProperty;
    *stream << " name=\"" << EscapeXmlAttribute(property.key()) << "\"";
    *stream << " value=\"" << EscapeXmlAttribute(property.value()) << "\"";
    *stream << "/>\n";
  }
  *stream << "</" << kProperties << ">\n";
}

// End XmlUnitTestResultPrinter

// Creates a new JsonUnitTestResultPrinter.
JsonPrinter::JsonPrinter(const char *output_file) : output_file_(output_file) {
  if (output_file_.empty()) {
    GTEST_LOG_(FATAL) << "JSON output file may not be null";
  }
}

void JsonPrinter::OnTestIterationEnd(const testing::UnitTest &unit_test,
                                     int iteration) {
  FILE *jsonout = NULL;
  std::string output_file_name = output_file_;
  if (testing::GTEST_FLAG(repeat) > 1) {
    std::string append_string = "_" + std::to_string(iteration);
    output_file_name.insert(output_file_name.find("."), append_string);
  }
  testing::internal::FilePath output_file(output_file_name);
  testing::internal::FilePath output_dir(output_file.RemoveFileName());
  if (output_dir.CreateDirectoriesRecursively()) {
    jsonout = testing::internal::posix::FOpen(output_file_name.c_str(), "w");
  }
  if (jsonout == NULL) {
    GTEST_LOG_(FATAL) << "Unable to open file \"" << output_file_name << "\"";
  }
  std::stringstream stream;
  PrintJsonUnitTest(&stream, unit_test);
  fprintf(jsonout, "%s",
          testing::internal::StringStreamToString(&stream).c_str());
  fclose(jsonout);
}

// Returns an JSON-escaped copy of the input string str.
std::string JsonPrinter::EscapeJson(const std::string &str) {
  testing::Message m;

  for (size_t i = 0; i < str.size(); ++i) {
    const char ch = str[i];
    switch (ch) {
      case '\\':
      case '"':
      case '/':
        m << '\\' << ch;
        break;
      case '\b':
        m << "\\b";
        break;
      case '\t':
        m << "\\t";
        break;
      case '\n':
        m << "\\n";
        break;
      case '\f':
        m << "\\f";
        break;
      case '\r':
        m << "\\r";
        break;
      default:
        if (ch < ' ') {
          m << "\\u00"
            << testing::internal::String::FormatByte(
                   static_cast<unsigned char>(ch));
        } else {
          m << ch;
        }
        break;
    }
  }

  return m.GetString();
}

// The following routines generate an JSON representation of a UnitTest
// object.

// Formats the given time in milliseconds as seconds.
static std::string FormatTimeInMillisAsDuration(
    testing::internal::TimeInMillis ms) {
  ::std::stringstream ss;
  ss << (static_cast<double>(ms) * 1e-3) << "s";
  return ss.str();
}

// Converts the given epoch time in milliseconds to a date string in the
// RFC3339 format, without the timezone information.
static std::string FormatEpochTimeInMillisAsRFC3339(
    testing::internal::TimeInMillis ms) {
  struct tm time_struct;
  if (!PortableLocaltime(static_cast<time_t>(ms / 1000), &time_struct))
    return "";
  // YYYY-MM-DDThh:mm:ss
  return StreamableToString(time_struct.tm_year + 1900) + "-" +
         testing::internal::String::FormatIntWidth2(time_struct.tm_mon + 1) +
         "-" + testing::internal::String::FormatIntWidth2(time_struct.tm_mday) +
         "T" + testing::internal::String::FormatIntWidth2(time_struct.tm_hour) +
         ":" + testing::internal::String::FormatIntWidth2(time_struct.tm_min) +
         ":" + testing::internal::String::FormatIntWidth2(time_struct.tm_sec) +
         "Z";
}

static inline std::string Indent(int width) { return std::string(width, ' '); }

void JsonPrinter::OutputJsonKey(std::ostream *stream,
                                const std::string &element_name,
                                const std::string &name,
                                const std::string &value,
                                const std::string &indent, bool comma) {
  const std::vector<std::string> &allowed_names =
      GetReservedAttributesForElement(element_name);

  GTEST_CHECK_(std::find(allowed_names.begin(), allowed_names.end(), name) !=
               allowed_names.end())
      << "Key \"" << name << "\" is not allowed for value \"" << element_name
      << "\".";

  *stream << indent << "\"" << name << "\": \"" << EscapeJson(value) << "\"";
  if (comma) *stream << ",\n";
}

void JsonPrinter::OutputJsonKey(std::ostream *stream,
                                const std::string &element_name,
                                const std::string &name, int value,
                                const std::string &indent, bool comma) {
  const std::vector<std::string> &allowed_names =
      GetReservedAttributesForElement(element_name);

  GTEST_CHECK_(std::find(allowed_names.begin(), allowed_names.end(), name) !=
               allowed_names.end())
      << "Key \"" << name << "\" is not allowed for value \"" << element_name
      << "\".";

  *stream << indent << "\"" << name << "\": " << StreamableToString(value);
  if (comma) *stream << ",\n";
}

// Prints a JSON representation of a TestInfo object.
void JsonPrinter::OutputJsonTestInfo(::std::ostream *stream,
                                     const char *test_case_name,
                                     const testing::TestInfo &test_info) {
  const testing::TestResult &result = *test_info.result();
  const std::string kTestcase = "testcase";
  const std::string kIndent = Indent(10);

  *stream << Indent(8) << "{\n";
  OutputJsonKey(stream, kTestcase, "name", test_info.name(), kIndent);

  if (test_info.value_param() != NULL) {
    OutputJsonKey(stream, kTestcase, "value_param", test_info.value_param(),
                  kIndent);
  }
  if (test_info.type_param() != NULL) {
    OutputJsonKey(stream, kTestcase, "type_param", test_info.type_param(),
                  kIndent);
  }

  OutputJsonKey(stream, kTestcase, "status",
                test_info.should_run() ? "RUN" : "NOTRUN", kIndent);
  OutputJsonKey(stream, kTestcase, "time",
                FormatTimeInMillisAsDuration(result.elapsed_time()), kIndent);
  OutputJsonKey(stream, kTestcase, "classname", test_case_name, kIndent, false);
  *stream << TestPropertiesAsJson(result, kIndent);

  int failures = 0;
  for (int i = 0; i < result.total_part_count(); ++i) {
    const testing::TestPartResult &part = result.GetTestPartResult(i);
    if (part.failed()) {
      *stream << ",\n";
      if (++failures == 1) {
        *stream << kIndent << "\""
                << "failures"
                << "\": [\n";
      }
      const std::string location =
          testing::internal::FormatCompilerIndependentFileLocation(
              part.file_name(), part.line_number());
      const std::string message = EscapeJson(location + "\n" + part.message());
      *stream << kIndent << "  {\n"
              << kIndent << "    \"failure\": \"" << message << "\",\n"
              << kIndent << "    \"type\": \"\"\n"
              << kIndent << "  }";
    }
  }

  if (failures > 0) *stream << "\n" << kIndent << "]";
  *stream << "\n" << Indent(8) << "}";
}

// Prints an JSON representation of a TestCase object
void JsonPrinter::PrintJsonTestCase(std::ostream *stream,
                                    const testing::TestCase &test_case) {
  const std::string kTestsuite = "testsuite";
  const std::string kIndent = Indent(6);

  *stream << Indent(4) << "{\n";
  OutputJsonKey(stream, kTestsuite, "name", test_case.name(), kIndent);
  OutputJsonKey(stream, kTestsuite, "tests", test_case.reportable_test_count(),
                kIndent);
  OutputJsonKey(stream, kTestsuite, "failures", test_case.failed_test_count(),
                kIndent);
  OutputJsonKey(stream, kTestsuite, "disabled",
                test_case.reportable_disabled_test_count(), kIndent);
  OutputJsonKey(stream, kTestsuite, "errors", 0, kIndent);
  OutputJsonKey(stream, kTestsuite, "time",
                FormatTimeInMillisAsDuration(test_case.elapsed_time()), kIndent,
                false);
  *stream << TestPropertiesAsJson(test_case.ad_hoc_test_result(), kIndent)
          << ",\n";

  *stream << kIndent << "\"" << kTestsuite << "\": [\n";

  bool comma = false;
  for (int i = 0; i < test_case.total_test_count(); ++i) {
    if (test_case.GetTestInfo(i)->is_reportable()) {
      if (comma) {
        *stream << ",\n";
      } else {
        comma = true;
      }
      OutputJsonTestInfo(stream, test_case.name(), *test_case.GetTestInfo(i));
    }
  }
  *stream << "\n" << kIndent << "]\n" << Indent(4) << "}";
}

// Prints a JSON summary of unit_test to output stream out.
void JsonPrinter::PrintJsonUnitTest(std::ostream *stream,
                                    const testing::UnitTest &unit_test) {
  const std::string kTestsuites = "testsuites";
  const std::string kIndent = Indent(2);
  *stream << "{\n";

  OutputJsonKey(stream, kTestsuites, "tests", unit_test.reportable_test_count(),
                kIndent);
  OutputJsonKey(stream, kTestsuites, "failures", unit_test.failed_test_count(),
                kIndent);
  OutputJsonKey(stream, kTestsuites, "disabled",
                unit_test.reportable_disabled_test_count(), kIndent);
  OutputJsonKey(stream, kTestsuites, "errors", 0, kIndent);
  if (testing::GTEST_FLAG(shuffle)) {
    OutputJsonKey(stream, kTestsuites, "random_seed", unit_test.random_seed(),
                  kIndent);
  }
  OutputJsonKey(stream, kTestsuites, "timestamp",
                FormatEpochTimeInMillisAsRFC3339(unit_test.start_timestamp()),
                kIndent);
  OutputJsonKey(stream, kTestsuites, "time",
                FormatTimeInMillisAsDuration(unit_test.elapsed_time()), kIndent,
                false);

  *stream << TestPropertiesAsJson(unit_test.ad_hoc_test_result(), kIndent)
          << ",\n";

  OutputJsonKey(stream, kTestsuites, "name", "AllTests", kIndent);
  *stream << kIndent << "\"" << kTestsuites << "\": [\n";

  bool comma = false;
  for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
    if (unit_test.GetTestCase(i)->reportable_test_count() > 0) {
      if (comma) {
        *stream << ",\n";
      } else {
        comma = true;
      }
      PrintJsonTestCase(stream, *unit_test.GetTestCase(i));
    }
  }

  *stream << "\n"
          << kIndent << "]\n"
          << "}\n";
}

// Produces a string representing the test properties in a result as
// a JSON dictionary.
std::string JsonPrinter::TestPropertiesAsJson(const testing::TestResult &result,
                                              const std::string &indent) {
  testing::Message attributes;
  for (int i = 0; i < result.test_property_count(); ++i) {
    const testing::TestProperty &property = result.GetTestProperty(i);
    attributes << ",\n"
               << indent << "\"" << property.key() << "\": "
               << "\"" << EscapeJson(property.value()) << "\"";
  }
  return attributes.GetString();
}

// End JsonPrinter
