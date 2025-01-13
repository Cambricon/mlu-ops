import re
import sys

mluop_abi_version = "MLUOP_ABI_1.0 {"
kernels_header_map = []

def extract_include_paths(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#include"):
                start_index = line.find('"') + 1
                end_index = line.rfind('"')
                if start_index!= -1 and end_index!= -1:
                    include_path = line[start_index:end_index]
                    kernels_header_map.append(include_path)

def get_mluops(input_file):
    node_msg_cp_finished=""
    pattern = re.compile(r'(?P<api>mluOp\w+) *\(')
    pattern_lite = re.compile(r'(?P<api>mlu\w+) *\(')
    with open(input_file,'r', encoding='utf8') as f:
        for line in f:
            match = pattern.search(line)
            lite_match = pattern_lite.search(line)
            if match:
                op = match.groupdict()['api'] + ';'
                node_msg_cp_finished += op

            if lite_match:
                op = lite_match.groupdict()['api'] + '*;'
                node_msg_cp_finished += '*' + op
    return node_msg_cp_finished

def create_map_file(map_file,node_msg_cp_finished):
    with open(map_file,'w') as f:
        f.writelines(mluop_abi_version + "\n")
        global_str = "\t" + "global: " + node_msg_cp_finished + "\n"
        f.writelines(global_str)
        f.writelines("\t" + "local: *;\n")
        f.writelines("};")


if __name__ == '__main__':
    try:
        # argv: *, map_file, bangc_kernels_collection.h, mlu_op.h
        assert len(sys.argv) == 4, "len(sys.argv) should be equal to 4."
        map_file = sys.argv[1]
        kernels_header_map.append(sys.argv[2])
        extract_include_paths(sys.argv[3])
        print("gen symbol with path: ", kernels_header_map)

        node_msg_cp_finished = ""
        for path in kernels_header_map:
            node_msg_cp_finished += get_mluops(path)
        create_map_file(map_file, node_msg_cp_finished)

    except Exception as e:
        print(f"[scripts/gen_symbol_visibility_map.py] An error occurred: {e}") 