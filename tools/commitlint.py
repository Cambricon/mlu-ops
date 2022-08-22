#!/usr/bin/python3

import sys
import re

types = ('[Feature]', '[Fix]', '[Docs]', '[WIP]')
scopes = ('(bangc-ops)', '(bangpy-ops)','(mlu-ops)')

#get header
def get_commit_msg(msg):
    res = re.search("\n", msg)
    if(res == None):
        return msg
    else:
        return msg[0:res.span()[0]]

def valid_commit_msg(commit_msg):
    res = re.match(r'(?P<type>\[\w+\])(?P<scope>\(\w+-ops\))(?P<colon>:)(?P<subject> *\w+)', commit_msg)
    if res == None:
        print("\033[0;35m-- please input standard format for commit: {[type](scope): <subject> }\033[0m")
        print("\033[0;35m-- the type should be one of: 'Feature', 'Fix', 'Docs', 'WIP' \033[0m")
        print("\033[0;35m-- the msg_scope should be one of: 'bangc-ops', 'bangpy-ops', 'mlu-ops' \033[0m")
        return False
    else:
        msg_type = res.groupdict()['type'].lstrip().rstrip()
        if msg_type not in types:
            print("\033[0;35m-- type not match, the type should be one of: 'Feature', 'Fix', 'Docs', 'WIP' \033[0m")
            print("\033[0;35m-- please input standard format for commit: {[type](scope): <subject> }\033[0m")
            return False

        msg_scope = res.groupdict()['scope'].lstrip().rstrip()
        if msg_scope not in scopes:
            print("\033[0;35m-- msg_scope not match, the msg_scope should be one of: 'bangc-ops', 'bangpy-ops'', 'mlu-ops'' \033[0m")
            print("\033[0;35m-- please input standard format for commit: {[type](scope): <subject> }\033[0m")
            return False
    return True

def main():
    print('-- [commit-msg] Checking git-commit-format')
    message_file = sys.argv[1]
    txt_file = open(message_file, 'r')
    msg = txt_file.read()
    commit_msg = get_commit_msg(msg)
    if(valid_commit_msg(commit_msg)):
        print('-- commit format is correct')
        sys.exit(0)
    sys.exit(1)

if __name__ == "__main__":
    main()
