%define __spec_install_post /usr/lib/rpm/brp-compress || :
%define debug_package %{nil}
%define neuware_dir /usr/local/neuware
%define build_dir bangc-ops/build

Name: mluops
Summary: The Machine Lerning Unit OPerators
Version: 0.5.0
Release: 1%{?dist}
License: Cambricon Release License
Vendor: Cambricon Inc.
URL: http://www.cambricon.com
BuildRoot: %{_tmppath}/%{name}-%{version}-root
BuildRequires: bash >= 2.03, sh-utils, tar
BuildRequires: gcc >= 4.8.2, libgcc >= 4.8.2
BuildRequires: gcc gcc-c++
BuildRequires: libstdc++ >= 4.8.2, glibc >= 2.17
BuildRequires: glibc-devel
BuildRequires: binutils >= 2.27
BuildRequires: readline-devel >= 6.2-4
BuildRequires: rpm-devel
#BuildRequires: python-devel
BuildRequires: texinfo-tex
BuildRequires: /usr/bin/pod2man
BuildRequires: texlive-ec texlive-cm-super
BuildRequires: systemtap-sdt-devel
BuildRequires: zlib-devel zlib-devel
BuildRequires: valgrind >= 3.13.0
BuildRequires: xz
BuildRequires: doxygen
BuildRequires: texlive-latex
#BuildRequires: python >= 2.7.0
#BuildRequires: cncc >= 2.6.0
#BuildRequires: cnas >= 2.6.0
Requires(post): /sbin/install-info
Requires(preun): /sbin/install-info
Requires: cndrv >= 0.2.0
Requires: cnrt >= 4.4.0
Autoreq: no

Source0: %{name}-%{version}.tar.gz
Source1: neuware-env.conf

%description
The Machine Lerning Unit OPerators.

%prep
%setup -q

%build
bash independent_build.sh -t %{_packagetype}

%install
install -d $RPM_BUILD_ROOT%{neuware_dir}/lib64
install -d $RPM_BUILD_ROOT%{neuware_dir}/include
install -d $RPM_BUILD_ROOT/etc/ld.so.conf.d
strip %{build_dir}/lib/libmluops.so*
cp %{build_dir}/lib/libmluops.so* $RPM_BUILD_ROOT%{neuware_dir}/lib64/
cp bangc-ops/mlu_op.h bangc-ops/mlu_op_kernel.h $RPM_BUILD_ROOT%{neuware_dir}/include/
cp -r samples/ $RPM_BUILD_ROOT%{neuware_dir}/
cp $RPM_SOURCE_DIR/neuware-env.conf $RPM_BUILD_ROOT/etc/ld.so.conf.d/

%clean
#rm -rf $RPM_BUILD_ROOT
#rm -rf $RPM_BUILD_DIR

%files
%defattr (-, root, root)
%{neuware_dir}/include/mlu_op.h
%{neuware_dir}/include/mlu_op_kernel.h
%{neuware_dir}/lib64/libmluops.so*
%{neuware_dir}/samples/bangc-ops
/etc/ld.so.conf.d/neuware-env.conf

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%changelog
* Mon Feb 20 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.5.0
* Mon Dec 19 2022 Cambricon Software Team <service@cambricon.com>
- release mluops v0.4.1
* Mon Dec 12 2022 Cambricon Software Team <service@cambricon.com>
- release mluops v0.4.0
* Fri Oct 14 2022 Cambricon Software Team <service@cambricon.com>
- release mluops v0.3.0
* Tue Sep 22 2022 Cambricon Software Team <service@cambricon.com>
- release mluops v0.2.0
* Wed Aug 31 2022 Cambricon Software Team <service@cambricon.com>
- release mluops v0.1.1
