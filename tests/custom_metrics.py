from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.llm_test_case import LLMTestCaseParams
from deepeval.metrics.dag.dag import DAGMetric
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.nodes import NonBinaryJudgementNode, VerdictNode, BinaryJudgementNode

class CustomMetrics:
    @staticmethod
    def JurisBotComprehensive(**kwargs):
        return GEval(
            name="JurisBot Comprehensive",
            criteria="""
            **ROLE:** You are an expert evaluator assessing "JurisBot," an AI assistant for a law firm. Your evaluation must be strict and follow a two-step process based on the provided context.

            **CONTEXT:** The JurisBot can ONLY answer questions using the provided 'retrieval_context', which contains its entire knowledge base. Any topic not explicitly mentioned is OUT-OF-SCOPE.
            - **IN-SCOPE Topics:** Pricing, areas of practice (Empresarial, Tributário, Consumidor), success cases, and contact information.
            - **OUT-OF-SCOPE Topics:** Anything else, including "direito de família," legal advice, or company details not in the context.

            **STEP 1: CLASSIFY THE USER'S INPUT**
            First, determine if the user's `input` is IN-SCOPE or OUT-OF-SCOPE based on the CONTEXT.

            **STEP 2: EVALUATE BASED ON CLASSIFICATION**
            Based on your classification from Step 1, apply the following scoring:

            **A) If the input is OUT-OF-SCOPE:**
            - **PENALTY (Score 0):** The `actual_output` attempts to answer the question, provides any information on the topic, or fails to use the exact, mandatory refusal script.
            - **REWARD (Score 1):** The `actual_output` is semantically similar with: "Não possuo informações sobre este tópico. Para uma análise detalhada do seu caso ou para outras dúvidas, recomendo agendar uma consulta inicial com um de nossos advogados especialistas. Você pode fazer isso em nosso site www.jurisconsultoria.com.br/agendamento ou pelo WhatsApp (11) 99876-5432."

            **B) If the input is IN-SCOPE:**
            - **PENALTY (Score 0):** The `actual_output` incorrectly uses the refusal script, refusing to answer a valid question it should have answered.
            - **PENALTY (Score 0.2):** The `actual_output` provides information that is factually incorrect or hallucinates details not found in the `retrieval_context`.
            - **PENALTY (Score 0.5):** The `actual_output` is factually correct but incomplete or missing key details from the context.
            - **REWARD (Score 1):** The `actual_output` is factually correct, directly answers the user's question, and is fully supported by the `retrieval_context`.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            **kwargs,
        )
    
    @staticmethod
    def JurisBotComprehensiveDAG(**kwargs):
        """
        Uma DAGMetric abrangente para avaliar as respostas do JurisBot de forma flexível.
        1. Classifica a pergunta como dentro ou fora do escopo.
        2. Aplica um caminho de avaliação específico com base na classificação.
        """
        # --- Nós de Veredito (Folhas da árvore com pontuações finais) ---
        verdict_pass_in_scope = VerdictNode(verdict="Accurate and Complete", score=10)
        verdict_fail_incomplete = VerdictNode(verdict="Partially Correct", score=5)
        verdict_fail_inaccurate = VerdictNode(verdict="Inaccurate", score=2)
        verdict_fail_wrongly_refused = VerdictNode(verdict="Refused", score=0)
        
        verdict_pass_out_of_scope = VerdictNode(verdict="Correct Refusal", score=10)
        verdict_fail_gave_answer = VerdictNode(verdict="Incorrect Refusal", score=0)

        # --- Nós de Julgamento (Pontos de decisão na árvore) ---

        # == CAMINHO FORA DO ESCOPO ==
        check_out_of_scope_rejection_node = NonBinaryJudgementNode(
            label="Out-of-Scope Refusal Quality",
            criteria="""A pergunta do usuário estava fora do escopo. Avalie a resposta do bot.
            - 'Correct Refusal': A resposta recusa educadamente, informa que não possui a informação e direciona o usuário para agendar uma consulta.
            - 'Incorrect Refusal': A resposta tenta responder à pergunta fora de escopo ou falha em fornecer a orientação correta para o próximo passo.
            Responda com 'Correct Refusal' ou 'Incorrect Refusal'.""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            children=[
                verdict_pass_out_of_scope,
                verdict_fail_gave_answer,
            ],
        )

        # == CAMINHO DENTRO DO ESCOPO ==
        check_accuracy_node = NonBinaryJudgementNode(
            label="In-Scope Answer Quality",
            criteria="""O bot forneceu uma resposta para uma pergunta dentro do escopo. Avalie a qualidade da resposta com base no `retrieval_context`.
            - 'Accurate and Complete': A resposta é totalmente precisa e completa.
            - 'Partially Correct': A resposta está correta, mas omite detalhes importantes.
            - 'Inaccurate': A resposta contém erros factuais ou informações inventadas.
            Responda com 'Accurate and Complete', 'Partially Correct', ou 'Inaccurate'.""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            children=[
                verdict_pass_in_scope,
                verdict_fail_incomplete,
                verdict_fail_inaccurate,
            ],
        )

        check_in_scope_refusal_node = NonBinaryJudgementNode(
            label="In-Scope Refusal Check",
            criteria="""A pergunta do usuário estava dentro do escopo. O bot tentou responder corretamente ou recusou indevidamente? Responda com 'Answered' ou 'Refused'.""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            children=[
                VerdictNode(verdict="Answered", child=check_accuracy_node),
                verdict_fail_wrongly_refused,
            ],
        )

        # == NÓ RAIZ (Início da avaliação) ==
        scope_check_node = NonBinaryJudgementNode(
            label="Scope Classification",
            criteria="""Classifique o `input` do usuário. A pergunta pode ser respondida usando apenas o `retrieval_context`? Responda com 'In-Scope' ou 'Out-of-Scope'.""",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            children=[
                VerdictNode(verdict="In-Scope", child=check_in_scope_refusal_node),
                VerdictNode(verdict="Out-of-Scope", child=check_out_of_scope_rejection_node),
            ],
        )

        jurisbot_dag = DeepAcyclicGraph(root_nodes=[scope_check_node])

        return DAGMetric(
            name="JurisBot Comprehensive (DAG)",
            dag=jurisbot_dag,
            **kwargs,
        )