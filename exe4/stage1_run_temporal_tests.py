# exe3/stage4_run_temporal_tests.py

from pathlib import Path
from contextlib import redirect_stdout
from .stage0_llm_rag import run_rag_with_multiple_configs
from .utils import get_queries



# =========================
# Main runner
# =========================

if __name__ == "__main__":
    queries=get_queries()
    queries=[queries[0],queries[1]]

    # print(len(queries))
    # exit()
    run_rag_with_multiple_configs(queries,k_list=[3,5,8],method_list=["dense","bm25"], chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/uk/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/uk/sources",nation="uk")
    run_rag_with_multiple_configs(queries,k_list=[3,5,8],method_list=["dense","bm25"], chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/uk/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/uk/sources",nation="uk")
    print("finish uk")
    run_rag_with_multiple_configs(queries,k_list=[3,5,8],method_list=["dense","bm25"], chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/us/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/us/sources",nation="us")
    run_rag_with_multiple_configs(queries,k_list=[3,5,8],method_list=["dense","bm25"], chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/us/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/us/sources",nation="us")
    


