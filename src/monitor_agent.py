# -*- coding: utf-8 -*-
# @Time    : 3/5/2024 10:51 PM
# @Author  : yuzhn
# @File    : test_agent.py
# @Software: PyCharm
from langchain.chains import SequentialChain
from langchain_core.callbacks import StdOutCallbackHandler

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd

from src.monitor import Monitor

if __name__ == '__main__':
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature = 0.9, model = llm_model)
    df = pd.read_csv('Data.csv')

    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english, and if the review is already in english, then keep it as is:"
        "\n\n{Review}"
    )
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm = llm, prompt = first_prompt,
                         output_key = "English_Review"
                         )
    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:"
        "\n\n{English_Review}"
    )
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm = llm, prompt = second_prompt,
                         output_key = "summary"
                         )
    # prompt template 3: translate to english
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm = llm, prompt = third_prompt, output_key = "language")

    # prompt template 4: follow up message
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )

    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm = llm, prompt = fourth_prompt,
                          output_key = "followup_message"
                          )

    # overall_chain: input= Review
    # and output= English_Review,summary, followup_message
    # use this chain for monitoring and logging
    overall_chain = SequentialChain(
        chains = [chain_one, chain_two, chain_three, chain_four],
        input_variables = ["Review"],
        output_variables = ["English_Review", "summary", "followup_message"],
        verbose = True
    )
    review = ('This is the best throw pillow fillers on Amazon. I’ve tried several others, and they’re all cheap and '
              'flat no matter how much fluffing you do. Once you toss these in the dryer after you remove them from '
              'the vacuum sealed shipping material, they fluff up great')

    overall_chain = Monitor(overall_chain, pre_checking = True, check_toxicity = True, check_existing_attack = True)
    overall_chain.display_info()
    overall_chain.display_data()
    handler = StdOutCallbackHandler()
    overall_chain.invoke({"Review": review}, {"callbacks": [handler]})
