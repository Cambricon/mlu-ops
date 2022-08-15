#coding=utf-8

import config

link_label_template = ('''.. _%(linkName)s:

''' if config.link_label else "")

# api标题模板
api_title_template = '''%(title)s
==============================================================================

'''

api_title_with_link_template = '''.. _%(linkName)s:

%(title)s
==============================================================================

'''

# datatype标题模板
datatype_title_template = '''%(title)s Data Type Reference
%(horizontal)s

'''

# enum内容模板
enum_content_template = link_label_template + '''enum %(enumName)s
%(horizontal)s

**enum %(enumName)s {**
    %(enumContent)s
**};**

.. doxygenenum:: %(doxygen)s

'''

typedef_enum_content_template = link_label_template + '''enum %(enumName)s
%(horizontal)s

**typedef enum {**
    %(enumContent)s
**} %(enumName)s;**

.. doxygenenum:: %(doxygen)s

'''

typedef_enum_with_bareName_content_template = link_label_template + '''enum %(enumName)s
%(horizontal)s

**typedef enum %(bareName)s {**
    %(enumContent)s
**} %(enumName)s;**

.. doxygenenum:: %(doxygenWithBareName)s
.. doxygentypedef:: %(doxygenWithName)s

'''

# enum property
enum_property_template = '''
    **%(item)s,**
'''

typedef_union_content_template = link_label_template + '''union %(unionName)s
%(horizontal)s

**typedef union {**
    %(unionContent)s
**} %(unionName)s;**

.. doxygenunion:: %(doxygen)s

'''

typedef_union_with_bareName_content_template = link_label_template + '''union %(unionName)s
%(horizontal)s

**typedef union %(bareName)s {**
    %(unionContent)s
**} %(unionName)s;**

.. doxygenunion:: %(doxygenWithBareName)s
.. doxygentypedef:: %(doxygenWithName)s

'''

# union property
union_property_template = '''
    **%(item)s;**
'''

# struct 内容模板
struct_content_template = link_label_template + '''struct %(structName)s
%(horizontal)s

**struct %(structName)s {**
    %(structContent)s
**};**

.. doxygenstruct:: %(doxygen)s
   :members:

'''
# struct内容模板
typedef_struct_content_template = link_label_template + '''struct %(structName)s
%(horizontal)s

**typedef struct {**
    %(structContent)s
**} %(structName)s;**

.. doxygenstruct:: %(doxygen)s
   :members:

'''

typedef_struct_with_bareName_content_template = link_label_template + '''struct %(structName)s
%(horizontal)s

**typedef struct %(bareName)s {**
    %(structContent)s
**} %(structName)s;**

.. doxygenstruct:: %(doxygenWithBareName)s
    :members:
.. doxygentypedef:: %(doxygenWithName)s

'''

struct_property_template = '''
    %(space)s**%(content)s;**
'''

# union
union_in_struct_content_template = '''
    %(space)s**union %(unionName)s{**
        %(content)s
    %(space)s**} %(varName)s;**
'''

# struct enum
enum_in_struct_content_template = '''
    %(space)s**enum %(enumName)s{**
        %(content)s
    %(space)s**} %(varName)s;**
'''

# struct enum property
enum_in_struct_property_template = '''
    %(space)s**%(item)s,**
'''

property_struct_content_template = '''
    %(space)s**struct %(structName)s{**
        %(content)s
    %(space)s**} %(varName)s;**
'''

# c typedef
typedef_content_template = link_label_template + '''typedef %(name)s
%(horizontal)s

.. doxygentypedef:: %(doxygen)s

'''

# namespace模板
namespace_template = '''Namespace %(namespace)s
============================================================================

.. doxygennamespace:: %(doxygen)s

'''

data_types_title_template = '''Data Type Reference
===========================

'''
cls_data_types_title_template = '''Data Type Reference
---------------------------

'''

define_title_template = '''Define Reference
===========================

'''

define_content_template = link_label_template + '''%(name)s
%(horizontal)s

.. doxygendefine:: %(doxygen)s

'''

cls_functions_title_template = '''API Reference
------------------------------

'''

function_content_template = link_label_template + '''%(name)s
%(horizontal)s

%(doxygen)s

'''

function_doxygen_template = '''.. doxygenfunction:: %(content)s
'''

class_title_template = '''Class %(name)s
%(horizontal)s

.. doxygenclass:: %(doxygen)s

'''



# typedef内容模板
using_content_template = link_label_template + '''typedef %(typedefName)s
%(horizontal)s

**typedef %(content)s;**

.. doxygentypedef:: %(doxygen)s

'''

class_index_template = '''Classes
=============================

.. toctree::
    :maxdepth: 5
    %(content)s
'''

class_index_item_template = '''
    %(name)s'''

api_group_index_template = '''%(title)s
==========================================================================

.. toctree::
    :maxdepth: 5
    %(content)s
'''

api_group_index_with_link_template = '''.. _%(link)s:

%(title)s
=============================================================================

.. toctree::
    :maxdepth: 5
    %(content)s
'''

api_group_index_item_template = '''
    %(name)s'''

# variable

cls_variable_title_template = '''Variables
------------------------

'''

variable_content_template = link_label_template + '''%(name)s
%(horizontal)s

.. doxygenvariable:: %(doxygen)s

'''
