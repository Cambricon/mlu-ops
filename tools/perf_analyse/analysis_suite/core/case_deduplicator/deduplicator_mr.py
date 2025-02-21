"""
    Execute with mapreduce
"""

__all__ = (
    "cal_md5",
    "run_deduplicator_mr",
)

import sys
import io
from mrjob.job import MRJob
from mrjob.step import MRStep
import hashlib
from typing import List, Dict

def cal_md5(
        file_path: str,
    ) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    md5_hash = hashlib.md5(data).hexdigest()
    return md5_hash

class MRFileDeduplicator(MRJob):
    def configure_args(self):
        super(MRFileDeduplicator, self).configure_args()

    def mapper(self, _, line):
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        md5_hash = cal_md5(line_str)
        yield md5_hash, line_str

    def reducer(self, _, file_paths):
        file_list = list(file_paths)
        yield file_list[0], len(file_list)

    def step(self):
        return [
            MRStep(
                mapper = self.mapper,
                reducer = self.reducer,
                )
        ]

def run_deduplicator_mr(
        file_list: List[str],
        num_mappers: int = 3,
        num_reducers: int = 4,
    ) -> Dict[str, int]:
    # config
    mr_job = MRFileDeduplicator(args=[
            "-r", "local",
            "--jobconf", "mapred.map.tasks=" + str(num_mappers),
            "--jobconf", "mapred.reduce.tasks=" + str(num_reducers),
        ])

    # set input
    sys.stdin = io.BytesIO('\n'.join(file_list).encode('utf-8'))

    case_to_count_dict = {}
    with mr_job.make_runner() as runner:
        # run mr job
        runner.run()

        # set output
        for k, v in mr_job.parse_output(runner.cat_output()):
            case_to_count_dict[k] = v

    return case_to_count_dict

if __name__ == "__main__":
    MRFileDeduplicator.run()

