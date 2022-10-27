cmake_policy(SET CMP0057 NEW) # Use `IN_LIST` operator
# B \ A = {x in B and x not in A}
macro(SET_RELATIVE_COMPLEMENT B A RET)
  set(_RELATIVE_COMP_RET "")
  foreach (element ${${B}})
    if (NOT element IN_LIST ${A})
      set(_RELATIVE_COMP_RET ${_RELATIVE_COMP_RET} ${element})
    endif()
  endforeach()
  set(${RET} ${_RELATIVE_COMP_RET})
endmacro()

# A + B = {x in A OR x in B}
macro(SET_UNION A B RET)
  set(_SET_UNION_RET ${${A}} ${${B}})
  list(REMOVE_DUPLICATES _SET_UNION_RET)
  set(${RET} ${_SET_UNION_RET})
endmacro()

## handle kernels dependency
if (NOT MLUOP_KERNEL_CONFIG)
  set(MLUOP_KERNEL_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/kernel_depends.toml")
endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if (NOT dependencies)
  message("#parse ${MLUOP_KERNEL_CONFIG} to populate dependencies")
  # get all kernels defined in ${MLUOP_KERNEL_CONFIG}
  execute_process(
    COMMAND grep -v -P "[[:space:]]*#" ${MLUOP_KERNEL_CONFIG}
    COMMAND tr -d "[:blank:]"
    COMMAND grep -P "^\\w\\S*?=\\[\\S*\\]"
    OUTPUT_VARIABLE depedencies
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND echo -n "${depedencies}"
    COMMAND cut -d "=" -f1
    COMMAND sort
    COMMAND uniq # remove duplicate key
    COMMAND xargs
    OUTPUT_VARIABLE pre_kernels
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  separate_arguments(pre_kernels)

  set(parsed_kernels)

  foreach(kernel ${pre_kernels})
    execute_process(
      COMMAND echo -n "${depedencies}"
      COMMAND grep -E "^${kernel}="
      COMMAND cut -d= -f 2
      COMMAND cut -d [ -f 2
      COMMAND cut -d ] -f 1
      COMMAND tr "," "\n"
      COMMAND sort
      COMMAND uniq # remove duplicate
      COMMAND tr "\n" " "
      COMMAND xargs echo -n # remove '"'
      OUTPUT_VARIABLE _kernel_depends
    )
    separate_arguments(_kernel_depends)
    list(REMOVE_ITEM _kernel_depends ${kernel})
    # <kernel> depends on <_kernel_depends;>
    set(kernel_depend_${kernel} ${_kernel_depends})
    foreach(dep ${_kernel_depends})
      # we have a -> {b}, at present c -> {a}, so we got c -> {a, b}
      SET_UNION(kernel_depend_${kernel} kernel_depend_${dep} kernel_depend_${kernel})
    endforeach()
    #foreach(dep ${_kernel_depends})
    #  # reverse relation <dep> needed by <kernel_support_dep>
    #  set(kernel_support_${dep} ${kernel_support_${dep}} ${kernel})
    #  foreach(support ${kernel_support_${kernel}})
    #    set(kernel_support_${dep} ${kernel_support_${dep}} ${support})
    #  endforeach()
    #endforeach()

    if (parsed_kernels)
      foreach (_kernel ${parsed_kernels})
        # we have a -> {b}, at present b -> {c}, so we got a -> {b, c}
        if (kernel IN_LIST kernel_depend_${_kernel})
          SET_UNION(kernel_depend_${_kernel} kernel_depend_${kernel} kernel_depend_${_kernel})
        endif()
      endforeach()
    endif()
    set(parsed_kernels ${parsed_kernels} ${kernel})

  endforeach()

  if (hasParent)
    set(pre_kernels ${pre_kernels} PARENT_SCOPE)
    set(dependencies ${dependencies} PARENT_SCOPE)
  endif()

  foreach(kernel ${pre_kernels})
    if (kernel IN_LIST kernel_depend_${kernel})
      message(WARNING "cyclic dependencies under ${kernel}")
      list(REMOVE_ITEM kernel_depend_${kernel} ${kernel})
    endif()
    message(STATUS "populated kernel ${kernel}, depend_on: ${kernel_depend_${kernel}}")
    if (hasParent)
      set(kernel_depend_${kernel} ${kernel_depend_${kernel}} PARENT_SCOPE)
    endif()
  endforeach()

endif()

macro(populate_op ret)
  set(${${ret}})
  cmake_parse_arguments(POPULATE_OP "" "" "SPECIFIC_OP" ${ARGN})
  foreach(kernel ${POPULATE_OP_SPECIFIC_OP})
    SET_UNION(${ret} kernel_depend_${kernel} ${ret})
  endforeach()
endmacro()


