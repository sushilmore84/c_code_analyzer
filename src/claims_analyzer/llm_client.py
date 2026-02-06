import os
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self):
        load_dotenv()
        
        self.anthropic_client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def call_llm(
        self,
        provider: str,      # "anthropic" | "openai"
        model: str,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> dict:
        """
        Normalized response structure for easy comparison
        """

        if provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            output_text = response.content[0].text

        elif provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            output_text = response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return {
            "provider": provider,
            "model": model,
            "text": output_text,
            "raw_response": response
        }


if __name__ == '__main__':
    llm = LLMClient()


    prompt = "Explain DRG in simple language with a real example."

    claude = llm.call_llm(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt=prompt
    )

    # gpt = llm.call_llm(
    #     provider="openai",
    #     model="gpt-4.1",
    #     prompt=prompt
    # )

    print("Claude:\n", claude["text"])
    #print("\nGPT:\n", gpt["text"])
