#!/bin/bash
# 此脚本用于上传当前目录中的文档到docview网站，并且将会在“yellow.hub.cambricon.com/toolrd/makedocs:latest”的Docker容器内运行。

# 此函数接受一个参数：docview网站的文档网址URL（建议使用从根目录开始的部分URL，以备未来docview网站换网址，例如：/website_guideline/source）
function build_upload_docview(){
	if [ -f "./Makefile" ]; then
		make clean && make html > makedocs.log
		cat makedocs.log
		found=`grep -c -i -w "ERROR" makedocs.log`
		if [ $found -ne "0" ]; then
			echo "ERROR: making html failed"
			return 1
		else
			echo "SUCCESS: build ok"
		fi
		# 仅在MR被合并后才上传文档到docview网站，其他rebuild、push都只是检查文档是否被正确编译
		if [ -d "./_build/html" -a 's_'$gitlabMergeRequestState = 's_merged' ]; then
			docview-upload ./_build/html $1 >> makedocs.log
			cat makedocs.log
			found=`grep -c -i -w "ERROR" makedocs.log`
			if [ $found -ne "0" ]; then
				echo "ERROR: uploading html failed"
				return 2
			else
				echo "SUCCESS: upload ok"
				return 0
			fi
		 fi
	fi
	return 0
}

# 不同的文档通常只要修改下面的docview网址即可
echo "start http://docview.cambricon.com/software4/cnnl/index.html"
build_upload_docview /software4/cnnl/docs/user_guide/zh
exit $?
