#coding=utf-8

import template, config
from HeaderParser import CppHeaderParser
import os, re

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def enum_to_rst_str(enum_template, name, enum, horizontal_type, doxygen, link_name, bare_name = "", doxygen_with_bare_name = ""):
    content = ""
    for e in enum:
        item = e["name"]
        if "raw_value" in e:
            item += " = " + e["raw_value"]
        content += template.enum_property_template % {'item': item}
    horizontal = horizontal_type * (len(name) + 6)
    if doxygen_with_bare_name:
        return enum_template % {
            'enumName': name,
            'bareName': bare_name,
            'linkName': link_name.strip("_"),
            'enumContent': content,
            'doxygenWithName': doxygen,
            'doxygenWithBareName': doxygen_with_bare_name,
            'horizontal': horizontal
        }

    return enum_template % {
        'enumName': name,
        'linkName': link_name.strip("_"),
        'enumContent': content,
        'doxygen': doxygen,
        'horizontal': horizontal
    }

def union_to_rst_str(union_template, name, union, horizontal_type, doxygen, link_name, bare_name = "", doxygen_with_bare_name = ""):
    content = ""
    for e in union:
        item = e["type"] + " " + e["name"]
        content += template.union_property_template % {'item': item}
    horizontal = horizontal_type * (len(name) + 7)
    if doxygen_with_bare_name:
        return union_template % {
            'unionName': name,
            'bareName': bare_name,
            'linkName': link_name.strip("_"),
            'unionContent': content,
            'doxygenWithName': doxygen,
            'doxygenWithBareName': doxygen_with_bare_name,
            'horizontal': horizontal
        }

    return union_template % {
        'unionName': name,
        'linkName': link_name.strip("_"),
        'unionContent': content,
        'doxygen': doxygen,
        'horizontal': horizontal
    }

def typedef_to_rst_str(name, doxygen, link_name, horizontal_type):
    horizontal = horizontal_type * (len(name) + 9)
    return template.typedef_content_template % {
        'name': name,
        'linkName': link_name.strip("_"),
        'horizontal': horizontal,
        'doxygen': doxygen
    }

def using_to_rst_str(name, horizontal_type, content, doxygen, link_name):
    horizontal = horizontal_type * (len(name) + 9)
    return template.using_content_template % {
        'typedefName': name,
        'linkName': link_name.strip("_"),
        'horizontal': horizontal,
        'content': content,
        'doxygen': doxygen
    }

def variable_to_rst_str(name, horizontal_type, doxygen, link_name):
    horizontal = horizontal_type * (len(name) + 2)
    return template.variable_content_template % {
        'name': name,
        'horizontal': horizontal,
        'doxygen': doxygen,
        'linkName': link_name
    }

def struct_to_rst_str(struct_template, name, horizontal_type, properties, nested_classes, doxygen, link_name, bare_name = "", doxygenWithBareName = ""):
    horizontal = horizontal_type * (len(name) + 8)
    content = properties_to_rst_str(name, properties, nested_classes, 0)
    if doxygenWithBareName:
        return struct_template % {
            'structName': name,
            'linkName': link_name.strip("_"),
            'horizontal': horizontal,
            'structContent': content,
            'doxygenWithName': doxygen,
            'bareName': bare_name,
            'doxygenWithBareName': doxygenWithBareName
        }
    return struct_template % {
        'structName': name,
        'linkName': link_name.strip("_"),
        'horizontal': horizontal,
        'structContent': content,
        'doxygen': doxygen,
    }

def properties_to_rst_str(parent_name, properties, nested_classes, level=0):
    content = ""
    for p in properties:
        cls = None
        if p['raw_type'] == parent_name + "::" + p['type']:
            for c in nested_classes:
                if c['name'] == p['type']:
                    cls = c
                    break
            if cls:
                if cls['declaration_method'] == 'union':
                    content += struct_union_to_rst_str(p["name"], cls, level)
                elif cls['declaration_method'] == 'struct':
                    content += property_struct_to_rst_str(p["name"], cls, level)
                else:
                    cls = None
        if not cls:
            p_name = p["name"]
            if 'enum_type' in p:
                # enum
                content += struct_enum_to_rst_str(p, level)
            else:
                if p["array"]:
                    p_name += p["primitive_array"]
                p_str = p["type"] + " " + p_name + ((" = %s" % p['defaultValue']) if 'defaultValue' in p else "")
                content += template.struct_property_template % {'content': p_str, 'space': '    ' * level}
    return content

def struct_union_to_rst_str(name, cls, level=0):
    properties = cls['properties']['public']
    content = properties_to_rst_str(cls['name'], properties, cls['nested_classes'], level + 1)
    return template.union_in_struct_content_template % {
        'varName': name,
        'unionName': '' if cls['name'].startswith('<anon-union') else cls['name'],
        'content': content,
        'space': '    ' * level
    }

def struct_enum_to_rst_str(enum, level=0):
    content = ''
    for value in enum['enum_type']['values']:
        item = value['name']
        if "raw_value" in value:
            item += " = " + value["raw_value"]
        content += template.enum_in_struct_property_template % {
            'item': item,
            'space': '    ' * (level + 1)
        }
    enumName = ''
    if 'name' in enum['enum_type']:
        enumName = enum['enum_type']['name']
    if enum['enum_type']['isclass']:
        enumName = 'class ' + enumName
    return template.enum_in_struct_content_template % {
        'varName': enum['name'] if 'name' in enum else '',
        'enumName': enumName,
        'content': content,
        'space': '    ' * level
    }

def property_struct_to_rst_str(name, cls, level=0):
    properties = cls['properties']['public']
    content = properties_to_rst_str(cls['name'], properties, cls['nested_classes'], level + 1)
    return template.property_struct_content_template % {
        'varName': name,
        'structName': '' if cls['name'].startswith('<anon-struct') else cls['name'],
        'content': content,
        'space': '    ' * level
    }


def function_to_rst_str(func_name, funcs, horizontal_type, link_name, prefix=""):
    horizontal = horizontal_type * (len(func_name) + 2)
    doxygen = ""
    for func in funcs:
        # 特殊处理 <type *>
        func_debug = func["debug"].replace(" *>", "*>")

        m = re.search("\([^;]*", func_debug)
        if m == None:
            continue
        if not prefix and "namespace" in func:
            prefix = func["namespace"]
        func_str = prefix + func_name + m.group()
        func_str = func_str.strip()
        if func_str.endswith("{"):
            func_str = func_str[:-1]
            func_str = func_str.strip()
        doxygen += template.function_doxygen_template % {
            'content': func_str
        }
    if not doxygen:
        return ""

    return template.function_content_template % {
        'name': func_name,
        'linkName': link_name.strip("_"),
        'horizontal': horizontal,
        'doxygen': doxygen
    }

def define_to_rst_str(value, horizontal_type):
    if not value:
        return ""
    value = re.sub(r"\s*\\\s*", " ", value).strip()
    name = ""
    has_value = False
    left = 0
    for i in range(len(value)):
        c = value[i]
        if c == "(":
            left += 1
            if not name:
                name = value[:value.index("(")]
        elif c == ")":
            left -= 1
            if left == 0 and not has_value and i < (len(value)-1):
                has_value = True
        elif c == " ":
            if not name:
                splited = value.split()
                name = splited[0]
                if len(splited) > 1:
                    has_value = True
    if not has_value:
        return ""
    horizontal = horizontal_type * (len(name) + 2)
    return template.define_content_template % {
        'name': name,
        'linkName': name.strip("_"),
        'horizontal': horizontal,
        'doxygen': name
    }

def data_types_to_rst(datatypes, rst_file_path, datatypes_sort):
    content = ""
    # enums
    enums = filter(lambda v: "name" in v, datatypes["enums"])
    if datatypes_sort:
        enums = sorted(enums, key=lambda v: v["name"])
    for enum in enums:
        tem = template.enum_content_template
        bare_name = ""
        doxygen_with_bare_name = ""
        if enum["typedef"]:
            if "bare_name" in enum and enum["bare_name"]:
                tem = template.typedef_enum_with_bareName_content_template
                bare_name = enum["bare_name"]
                doxygen_with_bare_name = enum["namespace"] + bare_name
            else:
                tem = template.typedef_enum_content_template
        content += enum_to_rst_str(tem, enum["name"], enum["values"], "-",
                                   enum["namespace"] + enum["name"], enum["name"], bare_name, doxygen_with_bare_name)

    # union
    unions = filter(lambda v: "name" in v, datatypes["unions"])
    if datatypes_sort:
        unions = sorted(unions, key=lambda v: v["name"])
    for union in unions:
        bare_name = ""
        doxygen_with_bare_name = ""
        if "bare_name" in union and union["bare_name"] and not union["bare_name"].startswith('<anon-union'):
            tem = template.typedef_union_with_bareName_content_template
            bare_name = union["bare_name"]
            doxygen_with_bare_name = union["namespace"] + bare_name
        else:
            tem = template.typedef_union_content_template
        content += union_to_rst_str(tem, union["name"], union["properties"]["public"], "-",
                                    union["namespace"] + union["name"], union["name"], bare_name, doxygen_with_bare_name)

    # struct
    structs = datatypes["structs"]
    if datatypes_sort:
        structs = sorted(structs, key=lambda v: v["name"])
    for v in structs:
        tem = template.struct_content_template
        bare_name = v["bare_name"]
        doxygen_with_bare_name = ""
        if "typedef" in v and v["typedef"]:
            if bare_name.startswith("<anon-struct"):
                bare_name = ""
            if bare_name:
                tem = template.typedef_struct_with_bareName_content_template
                doxygen_with_bare_name = ((v["namespace"] + "::") if v["namespace"] else "") + bare_name
            else:
                tem = template.typedef_struct_content_template
        content += struct_to_rst_str(tem, v["name"], "-", v["properties"]["public"], v['nested_classes'],
                                     ((v["namespace"] + "::") if v["namespace"] else "") + v["name"],
                                     v["name"], bare_name, doxygen_with_bare_name)

    # typedef
    typedefs = datatypes["typedefs"]
    if datatypes_sort:
        typedefs = sorted(datatypes["typedefs"], key=lambda v: v if isinstance(v, str) else (v["name"][v["name"].rindex("::") + 2:] if "::" in v["name"] else v["name"]))
    for v in typedefs:
        if isinstance(v, str):
            doxygen = v
            if doxygen.find("[") > 0:
                doxygen = doxygen[:doxygen.find("[")]
            content += typedef_to_rst_str(v, doxygen, v, "-")
        else:
            name = v["name"]
            typedef_name = name[name.rindex("::") + 2:] if "::" in name else name
            typedef_content = v["raw_type"] + " " + name
            content += using_to_rst_str(typedef_name, "-", typedef_content, name, typedef_name)

    if content:
        content = template.data_types_title_template + content
        write_file(rst_file_path, content)
        return True
    return False

def group_functions(funcs):
    groups = {}
    for func in funcs:
        if "group" in func and func["group"]:
            if func["group"] in groups:
                groups[func["group"]].append(func)
            else:
                groups[func["group"]] = [func]
        else:
            if "others" in groups:
                groups["others"].append(func)
            else:
                groups["others"] = [func]
    return groups

def functions_to_rst(funcs, group, rst_file_path, api_sort, is_group=False):
    content = (template.api_title_with_link_template % {'title': group, 'linkName': group}) if config.api_title_link_label \
        else (template.api_title_template % {'title': group})
    if api_sort:
        funcs = sorted(funcs, key=lambda v: v["name"])
    funcs_by_name = []
    func_name_indexs = {}
    funcs = filter(lambda v: not v["name"].startswith("function<"), funcs)
    for func in funcs:
        if func["name"] not in func_name_indexs:
            funcs_by_name.append([func])
            func_name_indexs[func["name"]] = len(funcs_by_name) - 1
        else:
            funcs_by_name[func_name_indexs[func["name"]]].append(func)
    for v in funcs_by_name:
        link_label = v[0]["name"]
        if is_group and group != "others":
            # group 去除空格，字母转小写
            link_label += "_" + group.replace(" ", "").lower()
        content += function_to_rst_str(v[0]["name"], v, "-", link_label)
    write_file(rst_file_path, content)

def defines_to_rst(defines, rst_file_path):
    content = template.define_title_template
    for define in defines:
        content += define_to_rst_str(define, "-")
    write_file(rst_file_path, content)

def cls_data_types_to_rst_str(cls, horizontal_type, datatypes_sort):
    # enums
    content = ""
    doxygen_prefix = ((cls["namespace"] + "::") if cls["namespace"] else "") + cls["name"] + "::"
    enums = list(filter(lambda v: "name" in v, cls["enums"]["public"]))
    if config.get_private:
        enums.extend(filter(lambda v: "name" in v, cls["enums"]["private"]))
    if config.get_protected:
        enums.extend(filter(lambda v: "name" in v, cls["enums"]["protected"]))
    if datatypes_sort:
        enums = sorted(enums, key=lambda v: v["name"])
    for enum in enums:
        tem = template.enum_content_template
        bare_name = ""
        doxygen_with_bare_name = ""
        if enum["typedef"]:
            if "bare_name" in enum and enum["bare_name"]:
                tem = template.typedef_enum_with_bareName_content_template
                bare_name = enum["bare_name"]
                doxygen_with_bare_name = doxygen_prefix + bare_name
            else:
                tem = template.typedef_enum_content_template
        doxygen = doxygen_prefix + enum["name"]
        content += enum_to_rst_str(tem, enum["name"], enum["values"], horizontal_type, doxygen, "%s_%s" % (cls["name"], enum["name"]), bare_name, doxygen_with_bare_name)

    # union
    unions = []
    for union in filter(lambda v: v["declaration_method"] == "union", cls["nested_classes"]):
        if union["access_in_parent"] == "public":
            unions.append(union)
        elif union["access_in_parent"] == "private":
            if config.get_private:
                unions.append(union)
        elif union["access_in_parent"] == "protected":
            if config.get_protected:
                unions.append(union)
    if datatypes_sort:
        unions = sorted(unions, key=lambda v: v["name"])
    for union in unions:
        bare_name = ""
        doxygen_with_bare_name = ""
        if "bare_name" in union and union["bare_name"] and not union["bare_name"].startswith('<anon-union'):
            tem = template.typedef_union_with_bareName_content_template
            bare_name = union["bare_name"]
            doxygen_with_bare_name = doxygen_prefix + bare_name
        else:
            tem = template.typedef_union_content_template
        doxygen = doxygen_prefix + union["name"]
        content += union_to_rst_str(tem, union["name"], union["properties"]["public"], horizontal_type, doxygen, "%s_%s" % (cls["name"], union["name"]), bare_name, doxygen_with_bare_name)

    # struct
    structs = []
    for struct in filter(lambda v: v["declaration_method"] == "struct", cls["nested_classes"]):
        if struct["access_in_parent"] == "public":
            structs.append(struct)
        elif struct["access_in_parent"] == "private":
            if config.get_private:
                structs.append(struct)
        elif struct["access_in_parent"] == "protected":
            if config.get_protected:
                structs.append(struct)
    if datatypes_sort:
        structs = sorted(structs, key=lambda v: v["name"])
    for v in structs:
        tem = template.struct_content_template
        bare_name = v["bare_name"]
        doxygen_with_bare_name = ""
        if "typedef" in v and v["typedef"]:
            if bare_name.startswith("<anon-struct"):
                bare_name = ""
            if bare_name:
                tem = template.typedef_struct_with_bareName_content_template
                doxygen_with_bare_name = doxygen_prefix + bare_name
            else:
                tem = template.typedef_struct_content_template
        content += struct_to_rst_str(tem, v["name"], horizontal_type, v["properties"]["public"], v['nested_classes'],
                                         doxygen_prefix + v["name"],
                                     "%s_%s" % (cls["name"], v["name"]), bare_name, doxygen_with_bare_name)

    # typedef
    typedefs = cls["typedefs"]["public"]
    if config.get_private:
        typedefs.extend(cls["typedefs"]["private"])
    if config.get_protected:
        typedefs.extend(cls["typedefs"]["protected"])
    for k, v in cls["using"].items():
        if v["using_type"] == "typealias":
            v["name"] = k
            typedefs.append(v)
    if datatypes_sort:
        typedefs = sorted(typedefs, key=lambda v: v if isinstance(v, str) else (v["name"][v["name"].rindex("::") + 2:] if "::" in v["name"] else v["name"]))
    for typedef in typedefs:
        if isinstance(typedef, str):
            content += typedef_to_rst_str(typedef, doxygen_prefix + typedef, cls["name"] + "_" + typedef, horizontal_type)
        else:
            typedef_content = typedef["raw_type"] + " " + ((cls["namespace"] + "::") if cls["namespace"] else "") + typedef["name"]
            doxygen = doxygen_prefix + typedef["name"]
            content += using_to_rst_str(typedef["name"], horizontal_type, typedef_content, doxygen, "%s_%s" % (cls["name"], typedef["name"]))

    if content:
        content = template.cls_data_types_title_template + content

    return content

def cls_to_rst(cls, rst_file_path, api_sort, datatypes_sort):
    horizontal = "=" * (len(cls["name"]) + 8)
    content = template.class_title_template % {
        "name": cls["name"],
        "horizontal": horizontal,
        "doxygen": ((cls["namespace"] + "::") if cls["namespace"] else "") + cls["name"]
    }

    # data_types
    content += cls_data_types_to_rst_str(cls, ">", datatypes_sort)

    # variable
    if config.get_class_variable:
        variables = cls["properties"]["public"]
        if config.get_private:
            variables.extend(cls["properties"]["private"])
        if config.get_protected:
            variables.extend(cls["properties"]["protected"])
        variables = list(filter(lambda v: 'enum_type' not in v, variables))
        if len(variables) > 0:
            if api_sort:
                variables = sorted(variables, key=lambda v: v["name"])
            content += template.cls_variable_title_template
            for variable in variables:
                var_name = variable['name']
                var_link_name = cls["name"] + "_" + var_name
                var_doxygen = ((cls["namespace"] + "::") if cls["namespace"] else "") + cls["name"] + "::" + var_name
                content += variable_to_rst_str(var_name, ">", var_doxygen, var_link_name)

    # function
    methods = cls["methods"]["public"]
    if config.get_private:
        methods.extend(cls["methods"]["private"])
    if config.get_protected:
        methods.extend(cls["methods"]["protected"])
    if len(methods) > 0:
        if api_sort:
            methods = sorted(methods, key=lambda v: v["name"])
        funcs_by_name = []
        func_name_indexs = {}
        # 排除类似  std::function<void(std::shared_ptr<CNFrameInfo>)> frame_done_callback_ = nullptr; 解析出的函数
        funcs = filter(lambda v: not v["name"].startswith("function<"), methods)
        for func in funcs:
            func["name"] = ("~" + func["name"]) if func["destructor"] else func["name"]
            if func["name"] not in func_name_indexs:
                funcs_by_name.append([func])
                func_name_indexs[func["name"]] = len(funcs_by_name) - 1
            else:
                funcs_by_name[func_name_indexs[func["name"]]].append(func)
        content += template.cls_functions_title_template
        for v in funcs_by_name:
            prefix = ((cls["namespace"] + "::") if cls["namespace"] else "") + cls["name"] + "::"
            content += function_to_rst_str(v[0]["name"], v, ">",  cls["name"] + "_" + v[0]["name"], prefix)
    write_file(rst_file_path, content)

def cls_index_to_rst(classes, rst_file_path):
    content = ""
    for name in classes:
        content += template.class_index_item_template % {'name': name}
    content = template.class_index_template % {'content': content}
    write_file(rst_file_path, content)

def api_index_to_rst(groups, rst_file_path):
    content = ""
    if config.api_index_sort:
        groups.sort()
    # 将others移到最后
    if "others" in groups:
        groups.remove("others")
        groups.append("others")
    for name in groups:
        content += template.api_group_index_item_template % {'name': name}
    tmp = template.api_group_index_template
    if config.api_title_link_label:
        tmp = template.api_group_index_with_link_template
    content = tmp % {
        'content': content,
        'title': config.api_group_index_title,
        'link': config.api_group_index_title.replace(" ", "")
    }
    write_file(rst_file_path, content)

def hpp_to_rst(file_paths, output_dir="", get_define=False, api_sort=False, datatypes_sort=False):
    datatype_rst_file_path = os.path.join(output_dir, "datatype.rst")
    define_rst_file_path = os.path.join(output_dir, "define.rst")
    cls_index_file_path = os.path.join(output_dir, "class_index.rst")
    api_index_file_path = os.path.join(output_dir, "api.rst")

    rst_paths = []
    cls_rst_paths = []
    functions = []
    datatypes = {
        "enums": [],
        "structs": [],
        "typedefs": [],
        "unions": []
    }
    defines = []
    classes = []
    for file_path in file_paths:
        header = CppHeaderParser.CppHeader(file_path, encoding="utf-8")
        path_basename = os.path.basename(file_path)
        file_name_without_suffix = path_basename[:path_basename.index(".")] if "." in path_basename else path_basename
        functions.extend(header.functions)
        datatypes["enums"].extend(header.enums)
        datatypes["structs"].extend(filter(lambda v: v["declaration_method"] == "struct" and not v["parent"], header.classes.values()))
        datatypes["typedefs"].extend(header.typedefs.keys())
        for k, v in header.using.items():
            if v["using_type"] == "typealias":
                v["name"] = k
                datatypes["typedefs"].append(v)
        datatypes["unions"].extend(
            filter(lambda v: v["declaration_method"] == "union" and not v["parent"], header.classes.values()))
        defines.extend(header.defines)

        # class
        for k, v in header.classes.items():
            if v["declaration_method"] == "class":
                classes.append(file_name_without_suffix + "_" + v["name"])
                cls_rst_file_path = os.path.join(output_dir, "%s_%s.rst" % (file_name_without_suffix, v["name"]))
                cls_to_rst(v, cls_rst_file_path, api_sort, datatypes_sort)
                cls_rst_paths.append(cls_rst_file_path)

    # data types
    if data_types_to_rst(datatypes, datatype_rst_file_path, datatypes_sort):
        rst_paths.append(datatype_rst_file_path)

    # function
    if len(functions) > 0:
        # 分组
        if config.api_group:
            func_groups = group_functions(functions)
            groups = []
            for group, group_funcs in func_groups.items():
                # 过滤需要隐藏的group
                if group in config.hidden_groups:
                    continue
                group_file_name = group.replace(" ", "_").lower()
                api_rst_file_path = os.path.join(output_dir, group_file_name + ".rst")
                functions_to_rst(group_funcs, group, api_rst_file_path, api_sort, True)
                rst_paths.append(api_rst_file_path)
                groups.append(group_file_name)

            # index.rst
            api_index_to_rst(groups, api_index_file_path)
            rst_paths.append(api_index_file_path)
        # 不分组
        else:
            # 过滤需要隐藏的api
            functions = filter(lambda x: "group" not in x or x["group"] not in config.hidden_groups, functions)

            api_rst_file_path = os.path.join(output_dir, "api.rst")
            functions_to_rst(functions, config.api_title, api_rst_file_path, api_sort)
            rst_paths.append(api_rst_file_path)

    # define
    if get_define:
        defines_to_rst(defines, define_rst_file_path)
        rst_paths.append(define_rst_file_path)

    # class_index
    if len(classes) > 0:
        cls_index_to_rst(classes, cls_index_file_path)
        rst_paths.append(cls_index_file_path)

    return "\n".join(rst_paths + cls_rst_paths)
