# controller/integration_loop.py

from models.hybrid.neuro_symbolic_bridge import NeuroSymbolicBridge

class IntegrationLoop:
    def __init__(self):
        self.bridge = NeuroSymbolicBridge()

    def run_once(self, input_text):
        print(f"[Input] {input_text}")
        result = self.bridge.process_text_and_reason(input_text)
        print("[Output]")
        print(result)

    def run_repl(self):
        print("Cogmenta Reasoning REPL — type 'exit' to quit")
        while True:
            user_input = input("\n> ")
            if user_input.lower() in ("exit", "quit"):
                break
            self.run_once(user_input)

if __name__ == "__main__":
    loop = IntegrationLoop()
    loop.run_repl()