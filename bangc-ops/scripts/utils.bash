#!/usr/bin/env bash

COLOR_FATAL=$'\033[41;38m'
COLOR_ERROR=$'\033[1;31m'
COLOR_WARN=$'\033[1;33m'
COLOR_NOTE=$'\033[1;32m'
COLOR_INFO=$'\033[1;34m'
COLOR_TRACE=$'\033[0;37m'
COLOR_RESET=$'\033[0m'

log_colored () {
  if [ -t 1 ]; then
    echo -n $1
  fi
  shift
  echo -e "[$PROG_NAME] $*"
  if [ -t 1 ]; then
    echo -e -n $COLOR_RESET
  fi
}

log_fatal () {
  log_colored $COLOR_FATAL $*
  exit -1
}

log_error () {
  log_colored $COLOR_ERROR $*
}

log_warn () {
  log_colored $COLOR_WARN $*
}

log_note () {
  log_colored $COLOR_NOTE $*
}

log_info () {
  log_colored $COLOR_INFO $*
}

log_trace () {
  log_colored $COLOR_TRACE $*
}

prog_log_fatal () {
  log_fatal  "|FATAL| $*"
}

prog_log_error () {
  log_error "|ERROR| $*"
}

prog_log_warn () {
  log_warn "|WARNING| $*"
}

prog_log_note () {
  log_note "|NOTE| $*"
}

prog_log_info () {
  log_info "|INFO| $*"
}

prog_log_trace () {
  log_trace "|TRACE| $*"
}

join_by () {
  # bash method like python ",".join([a, b, c,])
  # @ref Stack Overflow 1527049 how-can-i-join-elements-of-an-array-in-bash
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

version_le () {
  # compare v1:a.b.c and v2:x.y.z
  # check v1 <= v2, return 0 for true, 1 for false
  local v1=$1
  local v2=$2
  [ "$v1" = "$(echo -e "$v1\n$v2" | sort -V | head -n1)" ]
}

download_pkg () {
  local _url=$1
  local _filename=$(basename $_url)
  local _download_dir=$2
  local _package_verify=$(wget -c ${_url}.md5sum --timeout=10 -qO-)
  # ugly workaround for cntoolkit md5sum naming bug
  local _package_verify_fixname=$(paste <(awk '{print $1}' <<< "$_package_verify") <(echo "$_filename"))

  pushd ${_download_dir} >/dev/null
    ## Download package in either condition:
    #### 1. File not found
    #### 2. File.md5sum not found
    #### 3. File and md5sum does not match
    if [ ! -f "${_filename}" ] \
      || [ -z "${_package_verify}" ] \
      || [ -n "$(echo ${_package_verify_fixname} | md5sum --check --quiet || :)" ]; then
      rm -rf ${_filename}
      wget --progress=bar:force:noscroll -c ${_url} --timeout=100
    else
      prog_log_trace "download_pkg hit cache for ${_filename}"
    fi
  popd >/dev/null
}

get_json_val_from_str () {
  # def get_json_val (json_file, key, default_value=None)
  # parse json file and fetch value by key name
  # XXX Does not support nested object yet
  json_val='' # return value
  local input=$1
  local key=$2
  local match_pattern="\"${key}\".*"
  # json may contains newline and spaces
  # TODO avoid deleting spaces inside value string
  local matched_str=`echo ${input} | sed -z 's/\n//g' | sed -r 's/ +//g' | grep -E -o ${match_pattern} | sed -r "s/^\"${key}\" *://"`

  # parse formats below:
  #  | "(?key)":{(?object)}
  #  | "(?key)":[(?array)]
  #  | "(?key)":(true | false | null)
  #  | "(?key)":(?number)
  #  | "(?key)":"(?string)"
  local next_char=${matched_str:0:1}
  case $next_char in
    '{')
      # value may be object
      # TODO match nested braces
      json_val=$(cut -d '{' -f 2 <<< ${matched_str} | cut -d '}' -f 1)
      ;;
    '[')
      # value may be array
      # TODO match nested brackets
      json_val=$(cut -d '[' -f 2 <<< ${matched_str} | cut -d ']' -f 1)
      ;;
    '"')
      json_val=$(sed -r 's/"([^"]+)".*/\1/' <<< ${matched_str})
      ;;
    't' | 'f' | 'n')
      json_val=$(sed -r 's/^(true|false|null).*/\1/' <<< ${matched_str})
      ;;
    [-.0-9])
      # TODO validate number
      json_val=$(sed -r 's/([-.0-9]+).*/\1/' <<< ${matched_str})
      ;;
    *)
      ;;
  esac
  # | cut -d '{' -f 2 | cut -d '}' -f 1`
  if [ -z "${json_val}" ] && [ -n "$3" ]; then
    # return default value (if set)
    json_val=$3
  fi
  echo ${json_val};
}

get_json_val () {
  get_json_val_from_str "`cat $1`" $2 $3
}


common_extract () {
  local file_ext=${1##*.}
  local overwrite=$3
  local write_flag=
  mkdir -vp $2
  pushd $2 >/dev/null
    case ${file_ext} in
      rpm)
        [ -n "${overwrite}" ] && write_flag="-u"
        [ -n "${BUILD_VERBOSE}" ] && local ex_flag="-v"
        rpm2cpio $1 | cpio ${write_flag} -di $ex_flag
        ;;
      deb)
        [ -n "${overwrite}" ] && write_flag="--force-overwrite"
        [ -n "${BUILD_VERBOSE}" ] && local dpkg_ex="-X"
        dpkg ${dpkg_ex:--x} ${write_flag} $1 ./
        ;;
      gz)
        write_flag="--keep-newer-files"
        [ -n "${overwrite}" ] && write_flag="--overwrite"
        [ -n "${BUILD_VERBOSE}" ] && local ex_flag="-v"
        #  --no-same-owner for docker rootless
        tar ${write_flag} --no-same-owner -zxf $ex_flag $1
        ;;
      xz)
        write_flag="--keep-newer-files"
        [ -n "${overwrite}" ] && write_flag="--overwrite"
        [ -n "${BUILD_VERBOSE}" ] && local ex_flag="-v"
        tar ${write_flag} --no-same-owner -Jxf $ex_flag $1
        ;;
      bz2)
        write_flag="--keep-newer-files"
        [ -n "${overwrite}" ] && write_flag="--overwrite"
        [ -n "${BUILD_VERBOSE}" ] && local ex_flag="-v"
        tar ${write_flag} --no-same-owner -jxf $ex_flag $1
        ;;
      *)
        prog_log_fatal "Unknown file extension ${file_ext}, cannot extract this"
    esac
  popd >/dev/null
}

select_arch () {
  local arch=$1
  arch_name=''
  case $arch in
    x86_64)
      arch_name="amd64"
      ;;
    aarch64)
      arch_name="arm64"
      ;;
    *)
      echo 'unknown arch name'
  esac
  echo ${arch_name}
}

workaround_cmake_3_5_pkg_name () {
  # workaround for cmake 3.5, which does not support `CPACK_DEBIAN_<COMPONENT>_FILE_NAME` well
  local build_dir=$1
  local component=$2
  local component_lower=$(tr '[:upper:]' '[:lower:]' <<< $component)
  cmake3 --version 2>/dev/null && CMAKE=${CMAKE:-cmake3} || CMAKE=${CMAKE:-cmake}
  local cmake_minor_version=`${CMAKE} --version | head -n1 | cut -d '.' -f2`
  local cpack_debian_package_name=`sed -n "s@SET(CPACK_DEBIAN_${component}_FILE_NAME *\"\(.*\)\")@\1@p" $build_dir/CPackConfig.cmake`
  local cpack_rpm_package_name=`sed -n "s@SET(CPACK_RPM_${component}_FILE_NAME *\"\(.*\)\")@\1@p" $build_dir/CPackConfig.cmake`
  local package_name=`sed -n "s@SET(CPACK_PACKAGE_NAME *\"\(.*\)\")@\1@p" $build_dir/CPackConfig.cmake`
  local package_vesion=`sed -n "s@SET(CPACK_PACKAGE_VERSION *\"\(.*\)\")@\1@p" $build_dir/CPackConfig.cmake`
  echo $cmake_minor_version
  echo $cpack_debian_package_name
  if [ "${cmake_minor_version}" -lt "6" ]; then
    if [ -d $build_dir/_CPack_Packages/Linux/DEB/ ]; then
      for files in $build_dir/_CPack_Packages/Linux/DEB/$package_name-$package_vesion-Linux-$component_lower.deb; do
        if [ ! -f "$files" ]; then break; fi
        mv $files `dirname $files`/${cpack_debian_package_name}
      done
      for files in $build_dir/$package_name-$package_vesion-Linux-$component_lower.deb; do
        if [ ! -f "$files" ]; then break; fi
        mv $files `dirname $files`/${cpack_debian_package_name}
      done
    fi
    if [ -d $build_dir/_CPack_Packages/Linux/RPM/RPMS ]; then
      for files in $build_dir/_CPack_Packages/Linux/RPM/RPMS/$package_name-$package_vesion-Linux-$component_lower.rpm; do
        if [ ! -f "$files" ]; then break; fi
        mv $files `dirname $files`/${cpack_rpm_package_name}
      done
      for files in $build_dir/$package_name-$package_vesion-Linux-$component_lower.rpm; do
        if [ ! -f "$files" ]; then break; fi
        mv $files `dirname $files`/${cpack_rpm_package_name}
      done
    fi
  fi
}

