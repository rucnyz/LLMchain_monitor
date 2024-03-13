import pandas as pd
from IPython.core.display_functions import display
from langchain.chains.base import Chain
from tabulate import tabulate

from checking.existing_attack import ExistingAttackChecker
from checking.toxicity import ToxicityChecker
from utils import check_is_ipython


class Monitor:
    def __init__(self, chain: Chain, pre_checking = True, check_toxicity = False, check_existing_attack = False):
        # create a dataframe
        pd.set_option('display.max_columns', None)
        self.pre_checking = pre_checking
        self.df_info = pd.DataFrame()
        self.df_data = pd.DataFrame()
        self.chain = chain
        self.input_variable = chain.input_variables
        self.chain_list = chain.chains
        self.output_variable = chain.output_variables

        # toxicity checker
        if check_toxicity:
            self.check_toxicity = ToxicityChecker(model_path = 'martin-ha/toxic-comment-model', device = 'cuda')
        if check_existing_attack:
            self.check_existing_attack = ExistingAttackChecker("Salesforce/SFR-Embedding-Mistral")
        for i in range(len(self.chain_list)):
            llm_chain = self.chain_list[i].prompt.messages
            output_key = self.chain_list[i].output_key
            column = pd.MultiIndex.from_tuples([(f'chain_{i}', f'output_key'),
                                                (f'chain_{i}', f'model_name')])

            self.df_info = pd.concat(
                [self.df_info, pd.DataFrame([[output_key, self.chain_list[i].llm.model_name]], columns = column)],
                axis = 1)

            for j in range(len(llm_chain)):
                columns = pd.MultiIndex.from_tuples([(f'chain_{i}', f'prompt_{j}'),
                                                     (f'chain_{i}', f'input_variable_{j}')])
                template = llm_chain[j].prompt.template
                input_variable = llm_chain[j].input_variables
                column_data = pd.MultiIndex.from_tuples(
                    [(f'chain_{i}', f'input.{input_name}') for input_name in input_variable])
                self.df_data = pd.concat([self.df_data, pd.DataFrame(columns = column_data)], axis = 1)
                self.df_info = pd.concat([self.df_info, pd.DataFrame([[template, input_variable]], columns = columns)],
                                         axis = 1)

            column_data = pd.MultiIndex.from_tuples([(f'chain_{i}', f'output.{output_key}')])
            self.df_data = pd.concat([self.df_data, pd.DataFrame(columns = column_data)], axis = 1)
        # get all columns
        self.last_columns = self.df_data.columns.get_level_values(self.df_data.columns.nlevels - 1).tolist()
        self.last_columns = [col.split('.')[-1] for col in self.last_columns]
        # self.df_info.to_csv('monitor.csv', index = False)

    def __call__(self, *args, **kwargs):
        return self.chain(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def invoke(self, input, config = None, **kwargs):
        # differenct handler situation
        if self.pre_checking:
            for values in input.values():
                if self.check_toxicity:
                    toxicity = self.check_toxicity.check(values)
                    if toxicity > 0.6:
                        # Currently, we raise a ValueError if the toxicity is greater than 0.5
                        raise ValueError(f'The toxicity of the input is {toxicity}')
                if self.check_existing_attack:
                    attack = self.check_existing_attack.check(values)
                    if attack:
                        raise ValueError(f'The input similar to an existing attack')

        output = self.chain.invoke(input, config, **kwargs)
        self.df_data.loc[len(self.df_data)] = ['new_input', 'new_output']
        return output

    def display_info(self):
        if check_is_ipython():
            display(self.df_info)
        else:
            print(tabulate(self.df_info, headers = 'keys', tablefmt = 'psql'))

    def display_data(self):
        if check_is_ipython():
            display(self.df_data)
        else:
            print(tabulate(self.df_data, headers = 'keys', tablefmt = 'psql'))
