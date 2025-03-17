import logging
from typing import Dict, Any
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CheckingResultsAgent:
    """
    Agent responsible for evaluating the feasibility and flaws of research ideas.
    """

    def __init__(self,config: Dict[str, Any]):
        """
        Initializes the CritiquingAgent with a configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration,
                                     including the Gemini API key.
        """
        try:
            self.api_key = config['gemini_api_key']
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            logger.info("CritiquingAgent initialized successfully.")
        except KeyError:
            logger.error("Gemini API key not found in configuration.")
            raise
        except Exception as e:
            logger.error(f"Error initializing CritiquingAgent: {e}")
            raise
    def query_gemini(self, prompt: str) -> str:
        """
        Sends a prompt to Gemini and returns the response.

        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The response from Gemini.
        """
        try:
            response = self.chat.send_message(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            self.chat_history = self.chat.history
            return response.text
        except :
            logger.error(f"Error querying Gemini: {e}")
            time.sleep(100)
            self.chat = self.model.start_chat()
            self.chat.history = self.chat_history
            return self.query_gemini(prompt)

    def critique(self, chat_history ,results: str ) -> str:
        """
        Critiques the given research idea and suggests refinements using the Gemini API.

        Args:
            research_idea (str): The research idea to critique.

        Returns:
            str: A string containing the refined research idea.
        """        
        self.chat = self.model.start_chat()
        if chat_history is not None:
            self.chat.history = chat_history
        self.chat_history = chat_history
        try:
            prompt = f"""
            For now you are going to remark.
            this is the output we have got from running the project: {results} , 
            analyze the result and tell me if the novel approach we implemented is successful and better than the others
            give output only 1 if we are successful(our crreated novel approach is performing better than others)
            else return 0
            only return it nothing else as integer value
            your answer should come from reasons but do not give them in output
            **your output should only be the integer value no quotes  , nothing


            """

            response = self.query_gemini(prompt)
            logger.info(f"Results upto satisfaction: {response}")
            return response.strip() , self.chat.history

        except Exception as e:
            logger.exception(f"Error critiquing and refining research idea: {e}")
            return f"Error critiquing and refining research idea: {e}"
        

'''
if __name__ == '__main__':
    # Example Usage:
    # Assuming you have a config.yaml file with 'gemini_api_key'
    import yaml

    try:
        with open('/Users/krisanusarkar/Documents/ML/unt/generated/cais6/configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.  Make sure it exists and is in the correct location.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit()

    critiquing_agent = CheckingResultsAgent(config)
    research_idea = """create a better optimizer than rmsprop
    """
    results = """Requirement already satisfied: scikit-learn in ./venv/lib/python3.11/site-packages (from -r requirements.txt (line 1)) (1.6.1), Requirement already satisfied:
torch in ./venv/lib/python3.11/site-packages (from -r requirements.txt (line 2)) (2.6.0), Requirement already satisfied: numpy in 
./venv/lib/python3.11/site-packages (from -r requirements.txt (line 3)) (2.2.3), Requirement already satisfied: PyYAML in ./venv/lib/python3.11/site-packages 
(from -r requirements.txt (line 4)) (6.0.2), Requirement already satisfied: tqdm in ./venv/lib/python3.11/site-packages (from -r requirements.txt (line 5)) 
(4.67.1), Requirement already satisfied: matplotlib in ./venv/lib/python3.11/site-packages (from -r requirements.txt (line 6)) (3.10.1), Requirement already 
satisfied: scipy>=1.6.0 in ./venv/lib/python3.11/site-packages (from scikit-learn->-r requirements.txt (line 1)) (1.15.2), Requirement already satisfied: 
joblib>=1.2.0 in ./venv/lib/python3.11/site-packages (from scikit-learn->-r requirements.txt (line 1)) (1.4.2), Requirement already satisfied: 
threadpoolctl>=3.1.0 in ./venv/lib/python3.11/site-packages (from scikit-learn->-r requirements.txt (line 1)) (3.6.0), Requirement already satisfied: filelock
in ./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (3.18.0), Requirement already satisfied: typing-extensions>=4.10.0 in 
./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (4.12.2), Requirement already satisfied: networkx in 
./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (3.4.2), Requirement already satisfied: jinja2 in 
./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (3.1.6), Requirement already satisfied: fsspec in 
./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (2025.3.0), Requirement already satisfied: sympy==1.13.1 in 
./venv/lib/python3.11/site-packages (from torch->-r requirements.txt (line 2)) (1.13.1), Requirement already satisfied: mpmath<1.4,>=1.1.0 in 
./venv/lib/python3.11/site-packages (from sympy==1.13.1->torch->-r requirements.txt (line 2)) (1.3.0), Requirement already satisfied: contourpy>=1.0.1 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (1.3.1), Requirement already satisfied: cycler>=0.10 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (0.12.1), Requirement already satisfied: fonttools>=4.22.0 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (4.56.0), Requirement already satisfied: kiwisolver>=1.3.1 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (1.4.8), Requirement already satisfied: packaging>=20.0 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (24.2), Requirement already satisfied: pillow>=8 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (11.1.0), Requirement already satisfied: pyparsing>=2.3.1 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (3.2.1), Requirement already satisfied: python-dateutil>=2.7 in 
./venv/lib/python3.11/site-packages (from matplotlib->-r requirements.txt (line 6)) (2.9.0.post0), Requirement already satisfied: six>=1.5 in 
./venv/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib->-r requirements.txt (line 6)) (1.17.0), Requirement already satisfied: 
MarkupSafe>=2.0 in ./venv/lib/python3.11/site-packages (from jinja2->torch->-r requirements.txt (line 2)) (3.0.2), Using device: cpu, Training 
DRMS_SSBSTRGS..., Epoch 1/50, Train Accuracy: 0.7850, Test Accuracy: 0.8800, Time: 0.3824s, Epoch 2/50, Train Accuracy: 0.6325, Test Accuracy: 0.4300, Time: 
0.1485s, Epoch 3/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.1127s, Epoch 4/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.1194s, 
Epoch 5/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0667s, Epoch 6/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0350s, Epoch 
7/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0526s, Epoch 8/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0461s, Epoch 9/50, 
Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0404s, Epoch 10/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0528s, Epoch 11/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0367s, Epoch 12/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0326s, Epoch 13/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0374s, Epoch 14/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0381s, Epoch 15/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0337s, Epoch 16/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0564s, Epoch 17/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0419s, Epoch 18/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0243s, Epoch 19/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0262s, Epoch 20/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0312s, Epoch 21/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0484s, Epoch 22/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0436s, Epoch 23/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0418s, Epoch 24/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0353s, Epoch 25/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0274s, Epoch 26/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0320s, Epoch 27/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0509s, Epoch 28/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0509s, Epoch 29/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0396s, Epoch 30/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0323s, Epoch 31/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0301s, Epoch 32/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0404s, Epoch 33/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0407s, Epoch 34/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0335s, Epoch 35/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0270s, Epoch 36/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0372s, Epoch 37/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0454s, Epoch 38/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0369s, Epoch 39/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0455s, Epoch 40/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0458s, Epoch 41/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0328s, Epoch 42/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0322s, Epoch 43/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0393s, Epoch 44/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0354s, Epoch 45/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0631s, Epoch 46/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0398s, Epoch 47/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0243s, Epoch 48/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0332s, Epoch 49/50, Train 
Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0409s, Epoch 50/50, Train Accuracy: 0.5175, Test Accuracy: 0.4300, Time: 0.0369s, Training RMSprop..., Epoch 
1/50, Train Accuracy: 0.8100, Test Accuracy: 0.8500, Time: 0.0291s, Epoch 2/50, Train Accuracy: 0.8650, Test Accuracy: 0.8500, Time: 0.0504s, Epoch 3/50, 
Train Accuracy: 0.8825, Test Accuracy: 0.8600, Time: 0.0163s, Epoch 4/50, Train Accuracy: 0.8800, Test Accuracy: 0.8700, Time: 0.0275s, Epoch 5/50, Train 
Accuracy: 0.8875, Test Accuracy: 0.8800, Time: 0.0630s, Epoch 6/50, Train Accuracy: 0.8975, Test Accuracy: 0.8700, Time: 0.0207s, Epoch 7/50, Train Accuracy: 
0.9100, Test Accuracy: 0.8700, Time: 0.0273s, Epoch 8/50, Train Accuracy: 0.9075, Test Accuracy: 0.9000, Time: 0.0435s, Epoch 9/50, Train Accuracy: 0.9150, 
Test Accuracy: 0.9100, Time: 0.0314s, Epoch 10/50, Train Accuracy: 0.9300, Test Accuracy: 0.9200, Time: 0.0142s, Epoch 11/50, Train Accuracy: 0.9375, Test 
Accuracy: 0.9400, Time: 0.0148s, Epoch 12/50, Train Accuracy: 0.9425, Test Accuracy: 0.9600, Time: 0.0159s, Epoch 13/50, Train Accuracy: 0.9575, Test 
Accuracy: 0.9500, Time: 0.0332s, Epoch 14/50, Train Accuracy: 0.9550, Test Accuracy: 0.9500, Time: 0.0238s, Epoch 15/50, Train Accuracy: 0.9650, Test 
Accuracy: 0.9600, Time: 0.0211s, Epoch 16/50, Train Accuracy: 0.9675, Test Accuracy: 0.9700, Time: 0.0203s, Epoch 17/50, Train Accuracy: 0.9700, Test 
Accuracy: 0.9600, Time: 0.0416s, Epoch 18/50, Train Accuracy: 0.9700, Test Accuracy: 0.9800, Time: 0.0236s, Epoch 19/50, Train Accuracy: 0.9725, Test 
Accuracy: 0.9800, Time: 0.0367s, Epoch 20/50, Train Accuracy: 0.9725, Test Accuracy: 0.9900, Time: 0.0209s, Epoch 21/50, Train Accuracy: 0.9725, Test 
Accuracy: 0.9800, Time: 0.0211s, Epoch 22/50, Train Accuracy: 0.9775, Test Accuracy: 0.9800, Time: 0.0328s, Epoch 23/50, Train Accuracy: 0.9725, Test 
Accuracy: 0.9800, Time: 0.0367s, Epoch 24/50, Train Accuracy: 0.9775, Test Accuracy: 0.9900, Time: 0.0171s, Epoch 25/50, Train Accuracy: 0.9750, Test 
Accuracy: 0.9600, Time: 0.0227s, Epoch 26/50, Train Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0456s, Epoch 27/50, Train Accuracy: 0.9800, Test 
Accuracy: 0.9600, Time: 0.0339s, Epoch 28/50, Train Accuracy: 0.9775, Test Accuracy: 0.9800, Time: 0.0286s, Epoch 29/50, Train Accuracy: 0.9775, Test 
Accuracy: 0.9700, Time: 0.0206s, Epoch 30/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0251s, Epoch 31/50, Train Accuracy: 0.9800, Test 
Accuracy: 0.9800, Time: 0.0186s, Epoch 32/50, Train Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0200s, Epoch 33/50, Train Accuracy: 0.9825, Test 
Accuracy: 0.9900, Time: 0.0298s, Epoch 34/50, Train Accuracy: 0.9825, Test Accuracy: 0.9700, Time: 0.0244s, Epoch 35/50, Train Accuracy: 0.9800, Test 
Accuracy: 0.9700, Time: 0.0231s, Epoch 36/50, Train Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0304s, Epoch 37/50, Train Accuracy: 0.9825, Test 
Accuracy: 0.9800, Time: 0.0204s, Epoch 38/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0163s, Epoch 39/50, Train Accuracy: 0.9825, Test 
Accuracy: 0.9800, Time: 0.0183s, Epoch 40/50, Train Accuracy: 0.9775, Test Accuracy: 0.9900, Time: 0.0208s, Epoch 41/50, Train Accuracy: 0.9825, Test 
Accuracy: 0.9700, Time: 0.0211s, Epoch 42/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0235s, Epoch 43/50, Train Accuracy: 0.9825, Test 
Accuracy: 0.9700, Time: 0.0189s, Epoch 44/50, Train Accuracy: 0.9850, Test Accuracy: 0.9800, Time: 0.0208s, Epoch 45/50, Train Accuracy: 0.9850, Test 
Accuracy: 0.9900, Time: 0.0513s, Epoch 46/50, Train Accuracy: 0.9825, Test Accuracy: 0.9900, Time: 0.0810s, Epoch 47/50, Train Accuracy: 0.9850, Test 
Accuracy: 0.9700, Time: 0.0341s, Epoch 48/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0574s, Epoch 49/50, Train Accuracy: 0.9775, Test 
Accuracy: 0.9700, Time: 0.0462s, Epoch 50/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0334s, Training Adam..., Epoch 1/50, Train Accuracy: 
0.4750, Test Accuracy: 0.8900, Time: 0.0164s, Epoch 2/50, Train Accuracy: 0.8475, Test Accuracy: 0.8700, Time: 0.0196s, Epoch 3/50, Train Accuracy: 0.8575, 
Test Accuracy: 0.8600, Time: 0.0211s, Epoch 4/50, Train Accuracy: 0.8675, Test Accuracy: 0.8600, Time: 0.0295s, Epoch 5/50, Train Accuracy: 0.8675, Test 
Accuracy: 0.8600, Time: 0.0171s, Epoch 6/50, Train Accuracy: 0.8850, Test Accuracy: 0.8800, Time: 0.0380s, Epoch 7/50, Train Accuracy: 0.8950, Test Accuracy: 
0.8800, Time: 0.0194s, Epoch 8/50, Train Accuracy: 0.9025, Test Accuracy: 0.8900, Time: 0.0179s, Epoch 9/50, Train Accuracy: 0.9150, Test Accuracy: 0.9000, 
Time: 0.0138s, Epoch 10/50, Train Accuracy: 0.9225, Test Accuracy: 0.9000, Time: 0.0191s, Epoch 11/50, Train Accuracy: 0.9300, Test Accuracy: 0.9100, Time: 
0.0441s, Epoch 12/50, Train Accuracy: 0.9425, Test Accuracy: 0.9400, Time: 0.0246s, Epoch 13/50, Train Accuracy: 0.9450, Test Accuracy: 0.9400, Time: 0.0228s,
Epoch 14/50, Train Accuracy: 0.9550, Test Accuracy: 0.9600, Time: 0.0270s, Epoch 15/50, Train Accuracy: 0.9625, Test Accuracy: 0.9600, Time: 0.0266s, Epoch 
16/50, Train Accuracy: 0.9650, Test Accuracy: 0.9600, Time: 0.0229s, Epoch 17/50, Train Accuracy: 0.9625, Test Accuracy: 0.9600, Time: 0.0227s, Epoch 18/50, 
Train Accuracy: 0.9700, Test Accuracy: 0.9800, Time: 0.0292s, Epoch 19/50, Train Accuracy: 0.9700, Test Accuracy: 0.9700, Time: 0.0241s, Epoch 20/50, Train 
Accuracy: 0.9725, Test Accuracy: 0.9700, Time: 0.0181s, Epoch 21/50, Train Accuracy: 0.9725, Test Accuracy: 0.9800, Time: 0.0200s, Epoch 22/50, Train 
Accuracy: 0.9700, Test Accuracy: 0.9800, Time: 0.0222s, Epoch 23/50, Train Accuracy: 0.9725, Test Accuracy: 0.9800, Time: 0.0196s, Epoch 24/50, Train 
Accuracy: 0.9750, Test Accuracy: 0.9800, Time: 0.0184s, Epoch 25/50, Train Accuracy: 0.9750, Test Accuracy: 0.9800, Time: 0.0161s, Epoch 26/50, Train 
Accuracy: 0.9775, Test Accuracy: 0.9800, Time: 0.0289s, Epoch 27/50, Train Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0544s, Epoch 28/50, Train 
Accuracy: 0.9775, Test Accuracy: 0.9800, Time: 0.0249s, Epoch 29/50, Train Accuracy: 0.9800, Test Accuracy: 0.9700, Time: 0.0205s, Epoch 30/50, Train 
Accuracy: 0.9775, Test Accuracy: 0.9800, Time: 0.0208s, Epoch 31/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0204s, Epoch 32/50, Train 
Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0154s, Epoch 33/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0385s, Epoch 34/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0139s, Epoch 35/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0198s, Epoch 36/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0235s, Epoch 37/50, Train Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0181s, Epoch 38/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0243s, Epoch 39/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0330s, Epoch 40/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0219s, Epoch 41/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0220s, Epoch 42/50, Train 
Accuracy: 0.9850, Test Accuracy: 0.9800, Time: 0.0768s, Epoch 43/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0324s, Epoch 44/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.1397s, Epoch 45/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0755s, Epoch 46/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0440s, Epoch 47/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0527s, Epoch 48/50, Train 
Accuracy: 0.9800, Test Accuracy: 0.9800, Time: 0.0287s, Epoch 49/50, Train Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0175s, Epoch 50/50, Train 
Accuracy: 0.9825, Test Accuracy: 0.9800, Time: 0.0539s, Final DRMS_SSBSTRGS Testing Accuracy: 0.4300, Final RMSprop Testing Accuracy: 0.9800, Final Adam 
Testing Accuracy: 0.9800, """

    results = """Collecting numpy (from -r requirements.txt (line 1))
Using cached numpy-2.2.4-cp311-cp311-macosx_14_0_arm64.whl.metadata (62 kB)
Collecting scikit-learn (from -r requirements.txt (line 2))
Using cached scikit_learn-1.6.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (31 kB)
Collecting matplotlib (from -r requirements.txt (line 3))
Using cached matplotlib-3.10.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (11 kB)
Collecting pyyaml (from -r requirements.txt (line 4))
Using cached PyYAML-6.0.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (2.1 kB)
Collecting tqdm (from -r requirements.txt (line 5))
Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting torch (from -r requirements.txt (line 6))
Using cached torch-2.6.0-cp311-none-macosx_11_0_arm64.whl.metadata (28 kB)
Collecting scipy (from -r requirements.txt (line 7))
Using cached scipy-1.15.2-cp311-cp311-macosx_14_0_arm64.whl.metadata (61 kB)
Collecting joblib>=1.2.0 (from scikit-learn->-r requirements.txt (line 2))
Using cached joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn->-r requirements.txt (line 2))
Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting contourpy>=1.0.1 (from matplotlib->-r requirements.txt (line 3))
Using cached contourpy-1.3.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (5.4 kB)
Collecting cycler>=0.10 (from matplotlib->-r requirements.txt (line 3))
Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib->-r requirements.txt (line 3))
Using cached fonttools-4.56.0-cp311-cp311-macosx_10_9_universal2.whl.metadata (101 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib->-r requirements.txt (line 3))
Using cached kiwisolver-1.4.8-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.2 kB)
Collecting packaging>=20.0 (from matplotlib->-r requirements.txt (line 3))
Using cached packaging-24.2-py3-none-any.whl.metadata (3.2 kB)
Collecting pillow>=8 (from matplotlib->-r requirements.txt (line 3))
Using cached pillow-11.1.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (9.1 kB)
Collecting pyparsing>=2.3.1 (from matplotlib->-r requirements.txt (line 3))
Using cached pyparsing-3.2.1-py3-none-any.whl.metadata (5.0 kB)
Collecting python-dateutil>=2.7 (from matplotlib->-r requirements.txt (line 3))
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting filelock (from torch->-r requirements.txt (line 6))
Using cached filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)
Collecting typing-extensions>=4.10.0 (from torch->-r requirements.txt (line 6))
Using cached typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)
Collecting networkx (from torch->-r requirements.txt (line 6))
Using cached networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch->-r requirements.txt (line 6))
Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec (from torch->-r requirements.txt (line 6))
Using cached fsspec-2025.3.0-py3-none-any.whl.metadata (11 kB)
Collecting sympy==1.13.1 (from torch->-r requirements.txt (line 6))
Using cached sympy-1.13.1-py3-none-any.whl.metadata (12 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy==1.13.1->torch->-r requirements.txt (line 6))
Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib->-r requirements.txt (line 3))
Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch->-r requirements.txt (line 6))
Using cached MarkupSafe-3.0.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (4.0 kB)
Using cached numpy-2.2.4-cp311-cp311-macosx_14_0_arm64.whl (5.4 MB)
Using cached scikit_learn-1.6.1-cp311-cp311-macosx_12_0_arm64.whl (11.1 MB)
Using cached matplotlib-3.10.1-cp311-cp311-macosx_11_0_arm64.whl (8.0 MB)
Using cached PyYAML-6.0.2-cp311-cp311-macosx_11_0_arm64.whl (172 kB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached torch-2.6.0-cp311-none-macosx_11_0_arm64.whl (66.5 MB)
Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)
Using cached scipy-1.15.2-cp311-cp311-macosx_14_0_arm64.whl (22.4 MB)
Using cached contourpy-1.3.1-cp311-cp311-macosx_11_0_arm64.whl (254 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Using cached fonttools-4.56.0-cp311-cp311-macosx_10_9_universal2.whl (2.8 MB)
Using cached joblib-1.4.2-py3-none-any.whl (301 kB)
Using cached kiwisolver-1.4.8-cp311-cp311-macosx_11_0_arm64.whl (65 kB)
Using cached packaging-24.2-py3-none-any.whl (65 kB)
Using cached pillow-11.1.0-cp311-cp311-macosx_11_0_arm64.whl (3.1 MB)
Using cached pyparsing-3.2.1-py3-none-any.whl (107 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Using cached typing_extensions-4.12.2-py3-none-any.whl (37 kB)
Using cached filelock-3.18.0-py3-none-any.whl (16 kB)
Using cached fsspec-2025.3.0-py3-none-any.whl (193 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached networkx-3.4.2-py3-none-any.whl (1.7 MB)
Using cached MarkupSafe-3.0.2-cp311-cp311-macosx_11_0_arm64.whl (12 kB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: mpmath, typing-extensions, tqdm, threadpoolctl, sympy, six, pyyaml, pyparsing, pillow, packaging, numpy, networkx, MarkupSafe, 
kiwisolver, joblib, fsspec, fonttools, filelock, cycler, scipy, python-dateutil, jinja2, contourpy, torch, scikit-learn, matplotlib
Successfully installed MarkupSafe-3.0.2 contourpy-1.3.1 cycler-0.12.1 filelock-3.18.0 fonttools-4.56.0 fsspec-2025.3.0 jinja2-3.1.6 joblib-1.4.2 kiwisolver-1.4.8 
matplotlib-3.10.1 mpmath-1.3.0 networkx-3.4.2 numpy-2.2.4 packaging-24.2 pillow-11.1.0 pyparsing-3.2.1 python-dateutil-2.9.0.post0 pyyaml-6.0.2 scikit-learn-1.6.1 
scipy-1.15.2 six-1.17.0 sympy-1.13.1 threadpoolctl-3.6.0 torch-2.6.0 tqdm-4.67.1 typing-extensions-4.12.2
Training with RMSprop...
RMSprop Accuracy: 0.9733
Training with Adam...
Adam Accuracy: 0.9767
Training with AdamW...
AdamW Accuracy: 0.9767
Training with ALRC_RMSprop...
ALRC_RMSprop Accuracy: 0.9767

Results Table:
--------------------------------------------------
Optimizer       Accuracy   Sharpness
--------------------------------------------------
RMSprop         0.9733 0.0004
Adam            0.9767 0.0005
AdamW           0.9767 0.0006
ALRC_RMSprop    0.9767 0.0006
--------------------------------------------------
Project run successfully."""
    refined_idea = critiquing_agent.critique(None ,results )
    print(f"Refined Research Idea:\n{refined_idea}")

    '''