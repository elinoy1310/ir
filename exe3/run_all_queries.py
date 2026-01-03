# exe3/stage4b_run_all_queries.py
from pathlib import Path
from datetime import datetime

FACTUAL_REQUIRED = [
    "On what dates did the British Prime Minister deliver his speech on the defense budget?",
    "What was the main argument regarding the immigration bill that was presented?",
    "What three industrial sectors were mentioned as the main victims of the new trade policy that was presented?",
    "What organizations were mentioned by the speakers as supporting the proposed reform of the health system?",
]

CONCEPTUAL_REQUIRED = [
    "How does the rhetoric on climate change vary between different speakers; is the emphasis on economic opportunity or existential crisis?",
    "What is the central tension that emerges from the speeches between the need for national security and the protection of citizens’ privacy in the digital age?",
    "How is the state’s moral responsibility towards refugees and asylum seekers described, and what are the ethical (rather than economic) arguments given for and against their absorption?",
    "In what ways did speakers link investment in education to reducing future crime, and was there consensus on this issue?",
]

MY_FACTUAL = [
    "Which renewable energy technologies were explicitly mentioned in the debates, and in what national contexts were they discussed?",
    "What specific funding amounts or investment figures were mentioned in relation to energy, education, or decarbonisation programs?",
    "Which government departments or public bodies were explicitly mentioned as responsible for energy or climate-related policies?",
    "What geographic locations (regions or nations) were explicitly referenced in discussions about infrastructure or energy projects?"
]

MY_CONCEPTUAL = [
    "How do speakers justify continued investment in both renewable energy and traditional energy sources, and what tensions arise between these approaches?",
    "How is energy security framed in the debates: primarily as an economic issue, a national security concern, or an environmental responsibility?",
    "How do speakers describe the role of government versus the private sector in driving technological innovation and infrastructure development?",
    "What arguments are used to link long-term public investment, such as in education or energy infrastructure, to future national resilience and prosperity?"
]


def build_queries():
    return FACTUAL_REQUIRED + CONCEPTUAL_REQUIRED + MY_FACTUAL + MY_CONCEPTUAL
    # all_queries = []
    # for q in FACTUAL_REQUIRED:
    #     all_queries.append(("FACTUAL_REQUIRED", q))
    # for q in CONCEPTUAL_REQUIRED:
    #     all_queries.append(("CONCEPTUAL_REQUIRED", q))
    # for q in MY_FACTUAL:
    #     all_queries.append(("FACTUAL_YOURS", q))
    # for q in MY_CONCEPTUAL:
    #     all_queries.append(("CONCEPTUAL_YOURS", q))
    # return all_queries

if __name__ == "__main__":
    queries = build_queries()