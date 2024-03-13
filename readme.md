## How to use

```shell
export OPENAI_API_KEY=your-api-key
python monitor_agent.py
```

## Current features

- Monitor the input and output of a langchain agent. Users don't need to pass input and response to a monitor, instead,
  they just need to pass a langchain agent to our monitor.
- Check the input by examing existing attacks and toxicity
- Attack Dataset:
    - [x] jailbreak
    - [x] toxicity
    - [ ] stereotype
    - [ ] robustness
    - [ ] fairness
    - [ ] Ethics
    - [x] Privacy
      Details can be found in `dataset/readme.md`

## Future features

- Monitor all the variables in the chain
- Support more types of agent (now only support `langchain.chains.Chain`)
- Support more types of dataset
- Considering performance, the Monitor should have a `fast` feature, which means not monitoring before and after
  the chain execution but instead monitoring the chain asynchronously and notifying the user if a problem arises.
  It avoids impacting the chain's performance. However, the drawback is that it cannot prevent harmful
  content from happening, but only monitor it.
- Even when `fast=False`, parallel monitoring should still be adopted.

