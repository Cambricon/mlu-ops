import time
import sys
import os
guard_status_file = sys.argv[1]
guard_log_file = sys.argv[2]


def print_log(log_path):
    os.system("cat " + log_path)

if __name__ == '__main__':
    while True:
        # check status, end process when read "success" or "fail"
        status_file = open(guard_status_file, "r")
        line = status_file.readline().strip()
        status_file.close()
        if "success" in line.lower():
            print("Task success.")
            break
        elif "fail" in line.lower():
            print("Task Fail. The reason of failure shows as below:")
            print("--------------------------------------------------------------------------------")
            print_log(guard_log_file)
            exit(-1)
        # sleep for a while
        time.sleep(2)
