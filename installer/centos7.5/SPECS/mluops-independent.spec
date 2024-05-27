%define __spec_install_post /usr/lib/rpm/brp-compress || :
%define debug_package %{nil}
%define neuware_dir /usr/local/neuware
%define build_dir package

Name: mluops
Summary: The Machine Lerning Unit OPerators
Version: 1.1.1
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
BuildRequires: texinfo-tex
BuildRequires: /usr/bin/pod2man
BuildRequires: texlive-ec texlive-cm-super
BuildRequires: systemtap-sdt-devel
BuildRequires: zlib-devel zlib-devel
BuildRequires: valgrind >= 3.13.0
BuildRequires: xz
BuildRequires: doxygen
BuildRequires: texlive-latex
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
strip %{build_dir}%{neuware_dir}/lib64/libmluops.so*
cp -rf %{build_dir}/* $RPM_BUILD_ROOT
install -d $RPM_BUILD_ROOT/etc/ld.so.conf.d
cp $RPM_SOURCE_DIR/neuware-env.conf $RPM_BUILD_ROOT/etc/ld.so.conf.d/

%clean
#rm -rf $RPM_BUILD_ROOT
#rm -rf $RPM_BUILD_DIR

%files
%defattr (-, root, root)
%{neuware_dir}/*
/etc/ld.so.conf.d/neuware-env.conf

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%changelog
* Thu May 27 2024 Cambricon Software Team <service@cambricon.com>
- release mluops v1.2.0
* Thu Apr 12 2024 Cambricon Software Team <service@cambricon.com>
- release mluops v1.1.1
* Thu Mar 28 2024 Cambricon Software Team <service@cambricon.com>
- release mluops v1.1.0
* Tue Feb 6 2024 Cambricon Software Team <service@cambricon.com>
- release mluops v1.0.0
* Mon Dec 18 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.11.0
* Fri Nov 24 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.10.0
* Mon Oct 16 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.9.0
* Tue Aug 31 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.8.1
* Tue Aug 09 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.8.0
* Tue Jun 16 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.7.1
* Tue Jun 02 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.7.0
* Tue Apr 14 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.6.0
* Tue Mar 20 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.5.1
* Mon Mar 06 2023 Cambricon Software Team <service@cambricon.com>
- release mluops v0.4.2
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
