from typing import List, Optional
from pydantic import BaseModel
from typing import TypedDict

class PlanExecute(BaseModel):
    question: str
    curr_state: Optional[str] = None
    anonymized_question: Optional[str] = None
    query_to_retrieve_or_answer: Optional[str] = None
    plan: Optional[List[str]] = None
    past_steps: Optional[List[str]] = None
    mapping: Optional[dict] = None
    curr_context: Optional[str] = None
    aggregated_context: Optional[str] = None
    tool: Optional[str] = None
    response: Optional[str] = None
    
class QualitativeAnswerGraphState(TypedDict):
    """
    Represents the state of our graph.

    """

    question: str
    context: str
    answer: str
    
class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    question: str
    context: str
    relevant_context: str