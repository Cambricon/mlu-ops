import requests
import json
import time
import sys

local_communication_port = 12547

params = sys.argv
try:
    repo = params[1]
    user = params[2]
    pr_id = params[3]
    timestamp = params[4]
except Exception as e:
    print("Got some wrong with input params. Test fail.")
    exit(-1)

json_obj = {
    "timestamp": timestamp,
    "repo": repo,
    "pr_id": pr_id,
    "trigger_type": "ci",
    "trigger_id": user,
    "repeat_times": "3",
    "status": "running"
}
local_test_server = "http://localhost:" + str(local_communication_port)

# invoke test
response = requests.post(local_test_server, json=json_obj)
# get internal id
task_obj = json.loads(response.text)

try:
    while 1:
        response = requests.get(local_test_server + "/aiming=get_status&id=" + task_obj["id"])
        result = json.loads(response.text)
        if "success" in result["status"] or "fail" in result["status"] or "error" in result["status"] or "stable" in result["status"]:
            print(result["log"])
            print(result["status"])
            response = requests.get(local_test_server + "/aiming=end_job&id=" + task_obj["id"])
            if "success" in result["status"]:
                exit(0)
            else:
                exit(-1)
            break
        time.sleep(10)
except Exception as e:
    print(e)
    print("Got internal error while invoking test. Since we can not reboot this test, you should rerun this test in github.")
    exit(-1)
