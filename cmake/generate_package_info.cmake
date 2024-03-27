################################################################################
# Generate package info
################################################################################
message("generating package configuration")
set(PKG_NAME "mluops")
set(PKG_VERSION "${BUILD_VERSION}")
set(PKG_RELEASE "1")
set(PKG_PROVIDES "mlu-ops")
execute_process(COMMAND sh -c "LANG=en date '+%a, %d %b %G %X %z'" OUTPUT_VARIABLE PKG_DATE)
execute_process(COMMAND sh -c "LANG=en date '+%a %b %d %Y'" OUTPUT_VARIABLE PKG_DATE2 OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND grep "^ID=" "/etc/os-release"
  COMMAND cut -d = -f2
  COMMAND tr '[[:upper:]]' '[[:lower:]]'
  COMMAND xargs
  OUTPUT_VARIABLE HOST_DISTRO
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND grep "^VERSION_ID=" "/etc/os-release"
  COMMAND cut -d = -f2
  COMMAND xargs
  OUTPUT_VARIABLE HOST_DISTRO_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("HOST_DISTRO: ${HOST_DISTRO}")
if ( "${HOST_DISTRO}" MATCHES "ubuntu|debian|linuxmint" )
  # we could also check ID_LIKE and /etc/debian_version
  set(HINT_DEB_DISTRO "${HOST_DISTRO}${HOST_DISTRO_VERSION}")
else()
  set(HINT_DEB_DISTRO "")
endif()
if( "${HOST_DISTRO}" MATCHES "centos|scientific" )
  # have /etc/redhat-release
  set(HINT_RPM_DISTRO "el${HOST_DISTRO_VERSION}")
elseif("${HOST_DISTRO}" MATCHES "fedora")
  set(HINT_RPM_DISTRO "f${HOST_DISTRO_VERSION}")
else()
  set(HINT_RPM_DISTRO "")
endif()
set(DEB_DISTRO "${HINT_DEB_DISTRO}")
set(RPM_DISTRO "${HINT_RPM_DISTRO}")
# get codename of Linux distribution, we could read VERSION_CODENAME in /etc/os-release
# but at present, I just save these codenames in cmake itself
set(MAP_ubuntu16.04 "xenial")
set(MAP_ubuntu18.04 "bionic")
set(MAP_ubuntu20.04 "focal")
set(MAP_ubuntu20.10 "groovy")
set(MAP_ubuntu22.04 "jammy")
set(MAP_debian9 "stretch")
set(MAP_debian10 "buster")
set(MAP_debian11 "bullseye")
set(PKG_DISTRIBUTION "${MAP_${DEB_DISTRO}}")

execute_process(
  COMMAND head -n 1 "${CMAKE_SOURCE_DIR}/installer/independent/debian/changelog"
  COMMAND cut -c 9-20
  COMMAND cut -d - -f1
  COMMAND xargs
  OUTPUT_VARIABLE MLUOP_RELEASE_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message("current version: ${MLUOP_RELEASE_VERSION}")
if("${PKG_VERSION}" MATCHES "${MLUOP_RELEASE_VERSION}")
  message("build with mluops changelog")
else()
  configure_file(
    "${CMAKE_SOURCE_DIR}/installer/independent/debian/changelog.in"
    "${CMAKE_SOURCE_DIR}/installer/independent/debian/changelog"
    @ONLY
  )
endif()
