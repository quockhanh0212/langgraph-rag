import langgraph
from .utils.helper_functions import text_wrap
from .workflows.agent_workflow import agent_workflow


def execute_plan_and_print_steps(inputs, recursion_limit=45):
    """
    Execute the plan and print the steps.
    Args:
        inputs: The inputs to the plan.
        recursion_limit: The recursion limit.
    Returns:
        The response and the final state.
    """
    print(f'inputs: {inputs}')
    config = {"recursion_limit": recursion_limit}
    plan_and_execute_app = agent_workflow.compile()
    try:    
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            for _, agent_state_value in plan_output.items():
                pass
                print(f' curr step: {agent_state_value}')
        response = agent_state_value['response']
    except langgraph.pregel.GraphRecursionError:
        response = "The answer wasn't found in the data."
    final_state = agent_state_value
    print(text_wrap(f' the final answer is: {response}'))
    return response, final_state

def main():
    input = {"question": "How many houses are there in Hogwarts?"}
    final_answer, final_state = execute_plan_and_print_steps(input)
    print(final_answer, final_state)

if __name__ == "__main__":
    main()