from benchmarks.benchmark_agent import v_T02
import json
s = json.dumps({"subgoals":[1,2,3], "owner":"agent"})
print("json:", s)
print("v_T02 ->", v_T02(s))