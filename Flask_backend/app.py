from services.llm_service import generate_response
from flask import Flask

def create_app():
    app=Flask(__name__)
    history=[]
    while True:
        user_input=input("you: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append(f"""user: {user_input} AI: {generate_response(user_input,history)}""")
        print (history[-1].split("AI: ")[-1].strip())

    return app

if __name__=="__main__":
    app=create_app()
    app.run()