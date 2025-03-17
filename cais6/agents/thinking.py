import google.generativeai as genai
import json
import time

# Setup Gemini API (Replace with your API Key)
genai.configure(api_key="<YOURGEMINIAPIKEY>")
#model = genai.GenerativeModel('gemini-2.0-flash')

class Agent:
    """Base AI Agent with structured step-by-step reasoning."""
    def __init__(self, role):
        self.role = role
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.chat_history = None
    
    def chat(self, chat , prompt):
        """Sends a structured prompt to Gemini and returns response."""
        try:
            response = chat.send_message(prompt)
            self.chat_history = chat.history
            return response.text
        except Exception as e:
            print(f"Failed to send prompt to Gemini: {str(e)}")
            time.sleep(100)
            chat = self.model.start_chat()
            chat.history = self.chat_history
            return self.chat(chat , prompt)


class StepByStepIdeaGenerationAgent(Agent):
    """Generates research ideas step by step."""
    def generate_idea(self,chat ,  topic):
        prompt = f"""
        You are a reasoning-based  Scientist.  
        Think **step by step** to generate a novel research idea for {topic}.  
        
        Follow this strict reasoning process:  
        
        **Step 1: Identify a research gap**  
        - What is a major limitation in current research?  
        
        **Step 2: Propose a novel method**  
        - What is a new way to overcome this limitation?  
        
        **Step 3: Predict impact**  
        - If this method works, how will it change the field?  
        
        **Output Format**:  
        - **Research Gap:** [Your Answer]  
        - **Novel Method:** [Your Answer]  
        - **Expected Impact:** [Your Answer]  
        """
        return self.chat(chat , prompt)

class StepByStepCritiqueAgent(Agent):
    """Critiques an idea in structured steps."""
    def critique_idea(self, chat , idea):
        prompt = f"""
        You are a strict research critic.  
        Analyze the research idea **step by step** and then assign scores:  
        
        
        Follow this strict reasoning process:  
        
        Step 1: Novelty Check  
        - Is this idea truly new? Has similar work been done?  
        
        Step 2: Feasibility Check  
        - Is this idea practical? What are potential challenges?  
        - Is this idea mathematically better? or on paper actually.
        
        Step 3: Impact Check  
        - If the idea works, how significant is the impact?  
        
        Step 4: Suggest Refinements  
        - What changes would make the idea stronger?  
        
        Scoring (0-10 scale):  
        - Novelty Score: [Give a number from 0-10]  
        - Feasibility Score: [Give a number from 0-10]  
        - Impact Score: [Give a number from 0-10]  
        - **Total Score:** [Sum of the three scores]  
        
        
        **Response Format**
        The response must be a valid JSON object with this structure:
        {{
            "novelty": <score>,
            "feasibility": <score>,
            "impact": <score>,
            "overall": <weighted_score>,
            "weaknesses": "Brief explanation of the idea's flaws.",
            "strong points" : " show where it is strong"
            "refinements" : " show where it needs to change"
        }}
        ensure that the output can be loading by json after stripping
        Use ONLY double quotes for keys and string values.
        """
        try:
            response = self.chat(chat , prompt)
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip().replace("'", '')
            critique_json = json.loads(json_str)
        except:
            response = self.chat(chat , prompt+ "error in json loading afterwards , ensure proper output")
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip().replace("'", '')
            critique_json = json.loads(json_str)
        
        total_score = critique_json.get("overall", 0)
        feedback = critique_json.get("weaknesses", "")
        pros = critique_json.get("strong points", "")
        #print(critique_json)
        return total_score, feedback, pros

class RefinedIdeaGenerationAgent(Agent):
    """Generates research ideas step by step."""
    def refine_idea(self , chat ):
        prompt = f"""
        You are a reasoning-based Scientist.  
        generate refined idea   
        """
        return self.chat(chat , prompt)

class MultiAgentStepByStepSystem:
    """Runs step-by-step reasoning for research idea generation and refinement."""
    def __init__(self,threshold_score=23):
        self.threshold_score = threshold_score
        self.idea_agent = StepByStepIdeaGenerationAgent("Idea Generator")
        self.critique_agent = StepByStepCritiqueAgent("Critique & Refinement")
        self.refine_agent = RefinedIdeaGenerationAgent("Refined Idea Generator")
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def run(self ,chat_history , topic):
        """Main loop for generating and refining ideas step by step."""
        self.topic = topic
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
        idea = self.idea_agent.generate_idea(self.chat , topic)
        print(f"\n🔵 Iteration {1}: Proposed Idea 🔵\n{idea}\n")


        while True:
            
            total_score, feedback ,pros = self.critique_agent.critique_idea(self.chat , idea)
            #print(f"\n🟠 Step-by-Step Critique 🟠\n{critique}\n")

            print(f"📊 **Total Score:** {total_score}/30\n")
            print(f"Feedback: {feedback}\n")
            print(f"Pros: {pros}\n")
            #print(f"\n🟠 Step-by-Step Critique 🟠\n{refined_idea}\n")

            if total_score >= self.threshold_score:
                print("✅ Idea is strong. Stopping refinement.")
                break
            print(f"\n 🟢 Final Refined Idea 🟢\n\n")
            idea = self.refine_agent.refine_idea(self.chat)
            print(idea)
        
        return idea , self.chat.history

# Example Usage
'''
research_system = MultiAgentStepByStepSystem()
chat_history = None
while True:
    topic = input("Enter the topic: ")
    
    final_idea , chat_history = research_system.run(None , topic)

    print("\n🚀 Final Refined Research Idea 🚀\n", final_idea)
    '''

