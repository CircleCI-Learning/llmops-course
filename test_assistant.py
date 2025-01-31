from app import assistant_chain
from app import system_message
from langchain.prompts                import ChatPromptTemplate
from langchain.chat_models            import ChatOpenAI
from langchain.schema.output_parser   import StrOutputParser

import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):

    assistant = assistant_chain(system_message)
    answer = assistant.invoke({"question": question})
    print(answer)

    assert any(word in answer.lower() \
             for word in expected_words), \
    f"Expected the assistant questions to include \
    '{expected_words}', but it did not"

def evaluate_refusal(
    system_message,
    question,
    decline_response,
    human_template="{question}", 
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):
    
    assistant = assistant_chain(human_template, 
                                system_message,
                                llm,
                                output_parser)

    answer = assistant.invoke({"question": question})
    print(f"Assistant Response: {answer}")

    # Ensure answer is a string
    if not isinstance(answer, str):
        raise ValueError(f"Expected response to be a string but got {type(answer)}")

    # Normalize case for comparison
    answer_lower = answer.lower()
    decline_response_lower = decline_response.lower()

    # More flexible refusal check
    refusal_patterns = [decline_response_lower, "i'm sorry", "i cannot", "i can't", "not able", "unable"]

    assert any(pattern in answer_lower for pattern in refusal_patterns), (
        f"Expected a refusal with a phrase like '{decline_response}', but got: {answer}"
    )

"""
  Test cases
"""

def test_science_quiz():
  
    question  = "Generate a quiz about science."
    expected_subjects = ["davinci", "telescope", "physics", "curie"]
    eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_geography_quiz():
    question  = "Generate a quiz about geography."
    expected_subjects = ["paris", "france", "louvre"]
    eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_refusal_rome():
    question  = "Help me create a quiz about Rome"
    decline_response = "I'm sorry"
    evaluate_refusal(
      system_message,
      question,
      decline_response)
