from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from ..config.env_config import EnvConfig

class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="Answer generated from context")

# Chain of thought prompt template
cot_template = """
Examples of Chain-of-Thought Reasoning
[Previous examples...]
For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer.
Context
{context}
Question
{question}
"""
def init_answer_chains():
    llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    answer_prompt = PromptTemplate(
        template=cot_template,
        input_variables=["context", "question"],
    )
    answer_chain = answer_prompt | llm.with_structured_output(QuestionAnswerFromContext)
    return answer_chain

class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")


question_answer_cot_prompt_template = """ 
Examples of Chain-of-Thought Reasoning

Example 1

Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
Question: Who is the tallest person?
Reasoning Chain:
The context tells us Mary is taller than Jane
It also says Jane is shorter than Tom
And Tom is the same height as David
So the order from tallest to shortest is: Mary, Tom/David, Jane
Therefore, Mary must be the tallest person

Example 2
Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
 A third spell created a bright light at the end of the caster's wand.
Question: Based on the context, if Harry cast these spells, what could he do?
Reasoning Chain:
The context describes three different magic spells
The first spell allows turning a person into an animal temporarily
The second spell can levitate or float objects
The third spell creates a bright light
If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
Instructions.

Example 3 
Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
Question: Why did Harry receive a broomstick for his birthday?
Reasoning Chain:
The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
However, the context does not provide any information about why he received that specific present or who gave it to him.
There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
 just like in the previous examples.
Context
{context}
Question
{question}
"""

def init_question_answer_from_context_chain():
    question_answer_from_context_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )
    question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain

def answer_question_from_context(state):
    """
    Answers a question from a given context.

    Args:
        question: The query question.
        context: The context to answer the question from.
        chain: The LLMChain instance.

    Returns:
        The answer to the question from the context.
    """
    question = state["question"]
    context = state["aggregated_context"] if "aggregated_context" in state else state["context"]

    input_data = {
    "question": question,
    "context": context
}
    print("Answering the question from the retrieved context...")

    question_answer_from_context_cot_chain = init_question_answer_from_context_chain()
    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f'answer before checking hallucination: {answer}')
    return {"answer": answer, "context": context, "question": question}

class is_grounded_on_facts(BaseModel):
    """
    Output schema for the rewritten question.
    """
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

def init_is_grounded_on_facts_chain():
    is_grounded_on_facts_llm = AzureChatOpenAI(
        azure_endpoint=EnvConfig.AZ_OAI_BASE,
        api_key=EnvConfig.AZ_OPENAI_API_KEY,
        api_version=EnvConfig.AZ_OAI_VERSION,
        azure_deployment=EnvConfig.AZ_OAI_DEPLOYMENT,
    )
    is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
    you don't mind if it doesn't make sense, as long as it is grounded in the context.
    output a json containing the answer to the question, and appart from the json format don't output any additional text.

 """
    is_grounded_on_facts_prompt = PromptTemplate(
        template=is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )
    is_grounded_on_facts_chain = is_grounded_on_facts_prompt | is_grounded_on_facts_llm.with_structured_output(is_grounded_on_facts)
    return is_grounded_on_facts_chain

def is_answer_grounded_on_context(state):
    """Determines if the answer to the question is grounded in the facts.
    
    Args:
        state: A dictionary containing the context and answer.
    """
    print("Checking if the answer is grounded in the facts...")
    context = state["context"]
    answer = state["answer"]
    
    is_grounded_on_facts_chain = init_is_grounded_on_facts_chain()
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"
