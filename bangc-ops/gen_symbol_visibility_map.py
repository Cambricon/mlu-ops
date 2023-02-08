import re
import sys

mluop_abi_version = "MLUOP_ABI_1.0 {"

def get_mluops(input_file):
    ops_str=""
    pattern = re.compile(r'(?P<api>mluOp\w+) *\(')
    with open(input_file,'r', encoding='utf8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                op = match.groupdict()['api'] + ';'
                ops_str += op
    return ops_str

def create_map_file(map_file,ops_str):
    with open(map_file,'w') as f:
        f.writelines(mluop_abi_version + "\n")
        global_str = "\t" + "global: " + ops_str + "\n"
        f.writelines(global_str)
        f.writelines("\t" + "local: *;\n")
        f.writelines("};")

if __name__ == '__main__':
    if len(sys.argv) > 2:
        map_file = sys.argv[1]
        ops_str = ""
        for arg in sys.argv[2:]:
            ops_str += get_mluops(arg)
        create_map_file(map_file,ops_str)
    else:
        print("[ERROR] Missing script parameter. Call this script in 'python gen_symbol_visibility_map.py dst_path src_path' way.")
