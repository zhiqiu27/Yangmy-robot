class TargetAgent:
    def __init__(self, llm_client):
        self.client = llm_client
        self.name = "TargetAgent"

    def extract_targets(self, user_input):
        prompt = f"""
You are a perception-planning assistant for a quadruped robot.

Your task is to extract **two vision-compatible target entity names** from the user's instruction.
These should be object categories that can be recognized by a vision-language model (VLM), such as "book", "bottle", "human", "trash can", etc.

Requirements:
- Replace ambiguous language like "me", "you", or "myself" with "human".
- Only return generalizable object classes (not names or specific people).
- Respond only with a JSON object like: {{ "target_entities": ["object", "destination"] }}

User instruction:
"{user_input}"
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content
