from autogen.agentchat import Agent, AssistantAgent, UserProxyAgent
from typing import Dict, Optional, Union, Callable
import chainlit as cl

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

class ChainlitAssistantAgent(AssistantAgent):
    """
    Wrapper for AutoGens Assistant Agent
    """
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                author=self.name, # "AssistantAgent"
            ).send()
        )
        super(ChainlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )
        
class ChainlitUserProxyAgent(UserProxyAgent):
    """
    Wrapper for AutoGens UserProxy Agent. Simplifies the UI by adding CL Actions.
    """
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to chat_manager. Press enter to skip and use auto-reply"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or Exit Conversation?",
                    actions=[
                        cl.Action( name="continue", value="continue", label="✅ Continue" ),
                        cl.Action( name="feedback",value="feedback", label="💬 Provide feedback"),
                        cl.Action( name="exit",value="exit", label="🔚 Exit Conversation" )
                    ],
                )
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"
            if res.get("value")=="feedback":
                feedback_reply = cl.run_sync(ask_helper(cl.AskUserMessage, content="Please provide your feedback:", timeout=60))
                return feedback_reply["output"].strip()

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=60))

        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        cl.run_sync(
            cl.Message(
               content=f'{message}',# *Sending message to "{recipient.name}"*:\n\n
               author=self.name,
            ).send()
        )
        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )