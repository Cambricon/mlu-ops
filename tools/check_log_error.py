#!/usr/bin/python3

#检查LOG(ERROR)后是否有返回非success状态，仅作提示，不强制限制
import sys
import re

lines = ''
# LOG(ERROR) << "[cnnlWeightNormBackward] Only support axis == 0 or axis assign to"
#     #            << " last dimension currently. But now axis is " << axis << ".";
#     # return MLUOP_STATUS_SUCCESS;
sucess_regx = 'LOG\(ERROR\)[^}]+return MLUOP_STATUS_SUCCESS;'

#void func_name(.....) {
void_reg = 'void [*&\w]+\([^()]+\) \{'

# if (....) {
if_reg = 'if \([\S ]+\) \{'

#else {
else_reg = 'else \{'

#default: {
default_reg = 'default: \{'

#case xxxx : {
case_reg = 'case \w+: \{'

#func_name(...){}
func_reg = '\w+\([^\(\)]+\) \{'

map_ = {
    "if": if_reg,
    "else": else_reg,
    "case": case_reg,
    "default_reg": default_reg
}


def getFile(filename):
    global lines
    with open(filename, 'r') as f:
        lines = f.read()


#找到返回类型为void的函数，不用判断
def del_void_func():
    global lines
    st = []
    end = -1
    while 1:
        res = re.search(void_reg, lines)
        if res == None:
            break
        else:
            for i in range(res.span()[1] - 1, len(lines)):
                if lines[i] == "{":
                    st.append(i)
                if lines[i] == "}":
                    if len(st) == 0:
                        exit()
                    st.pop()
                    if len(st) == 0:
                        end = i
                        break
            if len(st)!= 0:
                exit()
            lines = lines[0:res.span()[0]] + lines[end + 1:]


#index处开始括号匹配
def find_brace(msg, index):
    start = index
    end = -1
    st = []
    for i in range(index, len(msg)):
        if msg[i] == "{":
            st.append(i)
        if msg[i] == "}":
            if len(st) == 0:
                exit()
            st.pop()
            if len(st) == 0:
                end = i
                break
    if len(st) != 0:
        exit()
    return start, end


def process(msg):
    start = -1
    end = -1
    while 1:
        msg = msg[end + 1:]
        res = re.search(func_reg, msg)
        if res != None:
            start, end = find_brace(msg, res.span()[1] - 1)
            function = msg[res.span()[0]:end + 1]
            for i in map_.keys():
                helper(function, map_[i])
        else:
            break


#处理if、else、switch中的LOG(ERROR)
def helper(func, reg):
    start = -1
    end = -1
    while 1:
        func = func[end + 1:]
        res = re.search(reg, func)
        if res != None:
            start, end = find_brace(func, res.span()[1] - 1)
            temp = func[res.span()[0]:end + 1]
            res2 = re.search("LOG\(ERROR\)[^;]+;", temp)
            if res2 != None:
                #LOG(ERROR)后面没有非success状态,函数可能被外部调用，可能返回false等值
                if temp[res2.span()[1]:].find(
                        "return") == -1 and temp[res2.span()[1]:].find(
                            "MLUOP_STATUS_NOT_INITIALIZED"
                        ) == -1 and temp[res2.span()[1]:].find(
                            "MLUOP_STATUS_ALLOC_FAILED"
                        ) == -1 and temp[res2.span()[1]:].find(
                            "MLUOP_STATUS_BAD_PARAM") == -1 and temp[res2.span(
                            )[1]:].find("MLUOP_STATUS_INTERNAL_ERROR"
                                        ) == -1 and temp[res2.span()[1]:].find(
                                            "MLUOP_STATUS_ARCH_MISMATCH"
                                        ) == -1 and temp[res2.span()[1]:].find(
                                            "MLUOP_STATUS_EXECUTION_FAILED"
                                        ) == -1 and temp[res2.span()[1]:].find(
                                            "MLUOP_STATUS_NOT_SUPPORTED") == -1:
                    print(
                        '-- the LOG(ERROR) may be not legal, please make sure have the correct return value'
                    )
                    print(
                        '-- this is just a hint, if you confirm it is correct, you can ignore it'
                    )
                    print(temp)
        else:
            break


def check():
    #先处理返回类型为void的函数
    del_void_func()

    #匹配到LOG(ERROR) << "[cnnlWeightNormBackward] Only support axis == 0 or axis assign to"
    #            << " last dimension currently. But now axis is " << axis << ".";
    # return MLUOP_STATUS_SUCCESS;  肯定错误
    # res = re.search(sucess_regx, lines)
    res = re.findall(sucess_regx,lines)
    if len(res)!=0:
        print("-- the return value is wrong")
        for i in res:
            print(i)

    process(lines)


def main():
    if sys.argv[1].endswith(".mlu") or sys.argv[1].endswith(".cpp"):
        getFile(sys.argv[1])
        check()


if __name__ == "__main__":
    main()
