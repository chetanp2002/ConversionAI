from groq import Groq

class EmailAgent:
    def __init__(self, api_key=None):
        self.client = None
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except:
                pass # Fail silently
    
    def write_email(self, profile):
        """
        Uses Llama-3 to write a personalized email based on user profile.
        """
        # Fallback if no API key
        if not self.client:
            return "⚠️ Please enter a valid Groq API Key to generate real AI emails."
        
        prompt = f"""
        Act as a senior marketing copywriter. Write a 1-sentence subject line and a 3-sentence email body.
        
        Target Customer Profile:
        - Age: {profile['Age']}
        - Spending Habit: ${int(profile['Amount'])} avg spend
        - Status: Hasn't visited in {profile['Recency']} days.
        
        Goal: Persuade them to buy using a 'Secret 20% Off' coupon.
        Tone: Exclusive and warm.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to AI: {str(e)}"