传入agent，（langchain），使用数据集对其进行测试（不同的agent可能使用不同数据集），得到输出。进行monitor，测试各种性能指标。
只能把不同的agent分开作为不同的情况考虑，但是可以提供统一的入口
langkit中，提供的统一入口就是，why.log()，里面需要手动传入prompt和response。
考虑更进一步，可以传进一个LLM chain，然后我们自动来处理这个chain的所有agent，然后得到结果。 

可以先从langchain入手，将一个chain传入进行考虑






## Others

打印dataframe不好看，使用tabulate，但存在问题，无法合并一级索引

两种情况，用户在constructer里传入handler，或者在invoke里传入handler