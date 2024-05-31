# /bin/bash

make_directory() {
    if [ ! -d $1 ];then
        mkdir $1
    fi
}

new_file() {
    if [ ! -f  $1 ];then
    	touch $1
    fi
}

get_timestamp() {
    current=`date "+%Y-%m-%d %H:%M:%S"`
    timeStamp=`date -d "$current" +%s`
    currentTimeStamp=$((timeStamp*1000+10#`date "+%N"`/1000000))
    echo $currentTimeStamp
}

get_pr_id() {
    PR_string=$(echo $GITHUB_REF | grep -Eo "/[0-9]*/")
    pr_id=(${PR_string//// })
    echo $pr_id
}

# default repo name
repo_name="mlu-ops"
request_postfix="rqt"

# get PR id
get_pr_id
pr_id=$?

# generate time stamp
get_timestamp
timestamp=$?

# initialize test directory and files in directory
repo_root="/home/user/${repo_name}_ci/"
requests_path="$repo_root/requests"
request_name="${repo_name}_${pr_id}_${timestamp}.rqt"
request_root="$repo_root/$request_name/"
sub_logs_path="$request_root/sub_logs/"

make_directory $repo_root
make_directory $requests_path
make_directory $request_root
make_directory $sub_logs_path

echo "working" > "$request_root/status"
new_file "$request_root/log"

# generate test file
echo "repo:${repo_name}" > "$requests_path/${request_name}"
echo "pr_id:${pr_id}" >> "$requests_path/${request_name}"
echo "timestamp:${timestamp}" >> "$requests_path/${request_name}"

# start script
python3 .github/ci_script/file_guard.py "$request_root/status" "$request_root/log" &
wait

status=$( head -n +1 ${request_root}/status )

if [ "$status" != "success" ];then
    cat ${request_root}/status
    exit -1
else
    cat ${request_root}/status
    exit 0
fi
