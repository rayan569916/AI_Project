from services.llm_service import generate_response
from flask import Flask

def create_app():
    app=Flask(__name__)
    while True:
        user_input=input("you: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print ("bot:", generate_response(user_input))

    return app

if __name__=="__main__":
    app=create_app()
    app.run()