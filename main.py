import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure APIs
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)


class SalesAgentsOrchestrator:
    def __init__(self):
        # Default preference: OpenAI first
        self.primary_model = "OpenAI GPT-4o"

        self.agents = {
            "discovery": {
                "product": self.product_discovery_ai,
                "discount": self.discount_adviser_agent,
                "profile": self.profile_matcher_agent,
                "segment": self.segment_recommendations,
            },
            "engagement": {
                "upsell": self.upsell_cross_sell_agent,
                "prompter": self.upsell_prompter,
                "outreach": self.proactive_outreach_engagement_agent,
            },
            "retention": {
                "followup": self.follow_up_agent,
                "loyalty": self.customer_loyalty_agent,
            }
        }

    def ai_reply(self, prompt):
        """Try OpenAI first, then fallback to Gemini"""
        try:
            # --- Try OpenAI ---
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            try:
                # --- Fallback to Gemini ---
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as ge:
                return f"❌ Both OpenAI and Gemini failed: {ge}"

    def handle_query(self, query):
        query_lower = query.lower()
        matched_agents = []

        for category, agents_dict in self.agents.items():
            for keyword, agent_func in agents_dict.items():
                if keyword in query_lower:
                    matched_agents.append((keyword, agent_func))

        if matched_agents:
            responses = []
            for keyword, agent in matched_agents:
                response = agent(query)
                responses.append(f"**{keyword.capitalize()} Agent:** {response}")
            return "\n\n".join(responses)

        fallback_response = self.product_discovery_ai(query)
        if fallback_response != "No products matched your search.":
            return f"**Product Discovery Agent:** {fallback_response}"

        return "Sorry, I couldn't find an answer to your query. Please try rephrasing."

    # === Agents ===
    def product_discovery_ai(self, data):
        return self.ai_reply(f"Find products related to: {data}")

    def discount_adviser_agent(self, data):
        return self.ai_reply(f"Suggest any current discounts or offers for: {data}")

    def profile_matcher_agent(self, data):
        return self.ai_reply(f"Match the best product profile for: {data}")

    def segment_recommendations(self, data):
        return self.ai_reply(f"Give fashion recommendations for segment: {data}")

    def upsell_cross_sell_agent(self, data):
        return self.ai_reply(f"Suggest upsell or cross-sell ideas for: {data}")

    def upsell_prompter(self, data):
        return self.ai_reply(f"Create a promotional upsell message for: {data}")

    def proactive_outreach_engagement_agent(self, data):
        return self.ai_reply(f"Draft proactive outreach engagement content for: {data}")

    def follow_up_agent(self, data):
        return self.ai_reply(f"Write a follow-up message for: {data}")

    def customer_loyalty_agent(self, data):
        return self.ai_reply(f"Check loyalty benefits for: {data}")


# === Streamlit App ===
def main():
    st.title("🛒 AI Sales Agent")

    orchestrator = SalesAgentsOrchestrator()
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if not user_query.strip():
            st.warning("Query cannot be empty!")
        else:
            with st.spinner("Thinking..."):
                response = orchestrator.handle_query(user_query)
                st.markdown(response)


if __name__ == "__main__":
    main()
