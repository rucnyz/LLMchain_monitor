class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from langchain."""

    def on_llm_start(
            self, serialized, prompts, **kwargs
    ):
        """Run when LLM starts running."""
        pass

    def on_chat_model_start(
            self, serialized, messages, **kwargs
    ):
        """Run when Chat Model starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        pass

    def on_llm_error(
            self, error, **kwargs
    ):
        """Run when LLM errors."""
        pass

    def on_chain_start(
            self, serialized, inputs, **kwargs
    ):
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs, **kwargs):
        """Run when chain ends running."""
        pass

    def on_chain_error(
            self, error, **kwargs
    ):
        """Run when chain errors."""
        pass

    def on_tool_start(
            self, serialized, input_str, **kwargs
    ):
        """Run when tool starts running."""
        pass

    def on_tool_end(self, output, **kwargs):
        """Run when tool ends running."""
        pass

    def on_tool_error(
            self, error, **kwargs
    ):
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs):
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action, **kwargs):
        """Run on agent action."""
        pass

    def on_agent_finish(self, finish, **kwargs):
        """Run on agent end."""
        pass
