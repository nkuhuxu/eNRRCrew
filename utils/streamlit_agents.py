from autogen.agentchat import Agent, AssistantAgent, UserProxyAgent
from typing import Dict, Optional, Union
import streamlit as st

# Define a helper function to handle user inputs in Streamlit
def ask_helper(func, **kwargs):
    res = func(**kwargs)
    while not res:
        res = func(**kwargs)
    return res

# Define the StreamlitAssistantAgent class
class StreamlitAssistantAgent(AssistantAgent):
    """
    Wrapper for AutoGen's AssistantAgent using Streamlit for UI
    """
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        # Display the message being sent in Streamlit
        st.write(f'*{self.name} is sending a message to "{recipient.name}":*')
        st.write(message)
        # Call the parent class's send method
        return super(StreamlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

# Define the StreamlitUserProxyAgent class
class StreamlitUserProxyAgent(UserProxyAgent):
    """
    Wrapper for AutoGen's UserProxyAgent using Streamlit to simplify the UI by adding interactions
    """
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to chat_manager. Press enter to skip and use auto-reply"
        ):
            st.write("Continue or provide feedback?")
            action = st.radio("Select an action:", ["✅ Continue", "💬 Provide feedback", "🔚 Exit Conversation"])
            if action == "✅ Continue":
                return ""
            elif action == "🔚 Exit Conversation":
                return "exit"
            elif action == "💬 Provide feedback":
                feedback = st.text_area("Please provide your feedback:")
                return feedback.strip()
        else:
            # Get user input with a timeout (Note: Streamlit doesn't support timeout directly)
            reply = st.text_input(prompt)
            return reply.strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # Display the message being sent in Streamlit
        st.write(f'*{self.name} is sending a message to "{recipient.name}":*')
        st.write(message)
        # Call the parent class's send method
        super(StreamlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )
