# 更新 direction_agent.py

class DirectionAgent:
    def __init__(self, llm_client):
        self.client = llm_client
        self.name = "DirectionAgent"

    def extract_direction(self, user_input):
        prompt = f"""
        You are controlling a quadruped robot based on natural language commands.

        From the user's instruction, extract the intended direction for the robot to turn or move. 
        You must strictly choose one of the following options and return it in a JSON object:

        ["forward", "backward", "left", "right",
        "front-left", "front-right", "back-left", "back-right"]

        Instruction: "{user_input}"

        Only return a valid JSON object with a single field named 'direction'.
        For example:
        {{ "direction": "back-right" }}

        Do not return any explanation or extra text.
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

