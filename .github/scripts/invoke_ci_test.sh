# /bin/bash
# get PR id
github_ref=$(echo $GITHUB_REF | grep -Eo "/[0-9]*/")
pr_id=(${github_ref//// })

# generate time stamp
current_time=`date "+%Y-%m-%d %H:%M:%S"`
timestamp_string=`date -d "${current_time}" +%s` 
current_timestamp=$((timestamp_string*1000+10#`date "+%N"`/1000000))
 
# temporally set to mlu370
card_type="MLU370-S4"

# default repo name
repo_name="mlu-ops"

github_user=${GITHUB_ACTOR}

# start script
python3 .github/scripts/run_ci_test.py ${repo_name} ${github_user} ${pr_id} ${current_timestamp}