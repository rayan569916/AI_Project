from services.llm_service import generate_response

def create_app():
    while True:
        user_input=input("you: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print ("bot:", generate_response(user_input))

create_app()