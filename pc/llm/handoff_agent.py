class HandoffAgent:
    def __init__(self, client, agents):
        self.client = client
        self.agents = agents

    def route(self, user_input: str):
        agent_definitions = {
            "TargetAgent": "Handles tasks that involve identifying or extracting the names of objects or goals, such as picking up or delivering items.",
            "DirectionAgent": "Handles spatial direction commands, such as turning left, moving forward, or interpreting spatial references like 'to your right'."
        }

        definitions_text = "\n".join([f"{name}: {desc}" for name, desc in agent_definitions.items()])

        prompt = f"""
You are a triage agent for a quadruped robot assistant system.

Your job is to choose which agent should handle the user's instruction, based on the instruction's intent.

Only choose **one** of the following agents:
- TargetAgent
- DirectionAgent

Agent Descriptions:
{definitions_text}

User instruction:
"{user_input}"

Respond with the agent name only: "TargetAgent" or "DirectionAgent".
Do not explain your answer. Do not include any other text.
"""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "text"}
        )

        return response.choices[0].message.content.strip()
