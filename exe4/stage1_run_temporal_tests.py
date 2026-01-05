# exe4/stage1_run_temporal_tests.py

from pathlib import Path
from contextlib import redirect_stdout
from .stage0_llm_rag import run_rag_with_multiple_configs
from .utils import get_queries



# =========================
# Main runner
# =========================

if __name__ == "__main__":
    queries=get_queries()
    queries=queries[1::]

    k_lst=[3,5,8]
    method_lst=["dense","bm25"]
    run_rag_with_multiple_configs(queries,k_list=k_lst,method_list=method_lst, chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/uk/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/uk/sources",nation="uk")
    run_rag_with_multiple_configs(queries,k_list=k_lst,method_list=method_lst, chunk_method="parent-son",answers_path_no_prefix="exe4/outputs/stage1/uk/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/uk/sources",nation="uk")
    print("finish uk")
    run_rag_with_multiple_configs(queries,k_list=k_lst,method_list=method_lst, chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/us/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/us/sources",nation="us")
    run_rag_with_multiple_configs(queries,k_list=k_lst,method_list=method_lst, chunk_method="parent-son",answers_path_no_prefix="exe4/outputs/stage1/us/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/us/sources",nation="us")
    


